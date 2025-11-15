# -*- coding: utf-8 -*-
"""
Przewidywanie opadów per STACJA (bez województw)
- Wczytuje dane IMGW SYNOP z lokalnych CSV (WORK_DIR/<rok>/*.csv)
- Buduje cechy czasowe dla KAŻDEJ stacji
- Cel: czy jutro (t+1) wystąpi opad >= 0.1 mm w tej stacji
- Walidacja: TimeSeriesSplit na train
- Podział: train/test po latach
- Zapisuje metryki, raporty, macierz pomyłek i model

Zależności:
    pip install pandas numpy scikit-learn joblib matplotlib seaborn
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import (
    classification_report, roc_auc_score, confusion_matrix,
    f1_score, precision_score, recall_score, average_precision_score
)
from sklearn.ensemble import HistGradientBoostingClassifier
from joblib import dump

warnings.filterwarnings("ignore")

# ====== KONFIGURACJA ======
WORK_DIR = "work_simple"      # wejście: WORK_DIR/<rok>/*.csv
OUT_DIR  = "out_simple"       # wyjście: raporty, wykresy, model
TRAIN_YEARS = [2019, 2020, 2021, 2022, 2023]
TEST_YEARS  = [2024]

# ====== WCZYTANIE DANYCH ======
def load_csv_files(years):
    """
    Wczytuje pliki CSV z danymi meteorologicznymi IMGW SYNOP.
    
    Oczekiwany układ kolumn w plikach CSV (bez nagłówków):
      0: station_id - kod stacji
      1: station_name - nazwa stacji
      2: rok
      3: miesiąc
      4: dzień
      14: opad [mm] - suma dobowa opadów
       5: tmax, 7: tmin, 9: tavg - temperatury (max, min, średnia)
    """
    # Listy do przechowywania wczytanych danych
    all_data = []  # Wszystkie wiersze danych z plików CSV
    all_stations = {}  # Słownik: kod_stacji -> nazwa_stacji

    # Przechodzimy przez każdy rok
    for year in years:
        year_dir = os.path.join(WORK_DIR, str(year))
        # Sprawdzamy czy katalog z danym rokiem istnieje
        if not os.path.exists(year_dir):
            continue
        # Znajdujemy wszystkie pliki CSV w katalogu
        csv_files = [f for f in os.listdir(year_dir) if f.endswith('.csv')]
        # Wczytujemy każdy plik CSV
        for csv_file in csv_files:
            path = os.path.join(year_dir, csv_file)
            try:
                # Wczytujemy CSV bez nagłówków, kodowanie latin1 (typowe dla danych IMGW)
                df = pd.read_csv(path, header=None, encoding='latin1')
                # Sprawdzamy czy plik ma wystarczającą liczbę kolumn
                if df.shape[1] < 15:
                    continue

                # Ekstrakcja ID i nazwy stacji (usuwamy spacje i cudzysłowy)
                df['station_id']   = df[0].astype(str).str.strip().str.replace('"', '')
                df['station_name'] = df[1].astype(str).str.strip().str.replace('"', '')

                # Składamy datę z 3 kolumn: rok-miesiąc-dzień
                # str.zfill(2) dodaje zero wiodące dla miesięcy/dni < 10
                df['date'] = pd.to_datetime(
                    df[2].astype(str) + '-' + df[3].astype(str).str.zfill(2) + '-' + df[4].astype(str).str.zfill(2),
                    errors='coerce'  # Błędne daty zamieniane na NaN
                )

                # Ekstrahujemy dane meteorologiczne
                df['precip'] = pd.to_numeric(df[14], errors='coerce').fillna(0.0)  # Opady, brak danych = 0
                df['tmax']   = pd.to_numeric(df[5],  errors='coerce')  # Temperatura maksymalna
                df['tmin']   = pd.to_numeric(df[7],  errors='coerce')  # Temperatura minimalna
                df['tavg']   = pd.to_numeric(df[9],  errors='coerce')  # Temperatura średnia


                # Zostawiamy tylko kolumny, których potrzebujemy
                df = df[['station_id','station_name','date','precip','tmax','tmin','tavg']]
                # Usuwamy rekordy z błędnymi datami
                df = df.dropna(subset=['date'])
                # Dodajemy DataFrame do listy
                all_data.append(df)
            except Exception:
                # Jeśli plik jest uszkodzony, pomijamy go
                continue

    # Sprawdzamy czy udało się wczytać jakiekolwiek dane
    if not all_data:
        raise RuntimeError("Nie znaleziono żadnych danych wejściowych w WORK_DIR.")


    # Łączymy wszystkie DataFrames w jeden
    data = pd.concat(all_data, ignore_index=True)
    # Sortujemy i usuwamy duplikaty (gdyby ta sama stacja i data występowała wielokrotnie)
    data = data.sort_values(['station_id','date']).drop_duplicates(subset=['station_id','date'])
    return data, all_stations

# ====== CECHY (PER STACJA) ======
def build_features_per_station(df: pd.DataFrame) -> pd.DataFrame:

    df = df.sort_values(['station_id','date']).copy()

    # bazowe
    df['precip'] = df['precip'].fillna(0.0)
    df['tmax']   = df['tmax'].astype(float)
    df['tmin']   = df['tmin'].astype(float)
    df['tavg']   = df['tavg'].astype(float)

    # dzisiejszy opad binarny + target jutro
    df['rain_today'] = (df['precip'] >= 0.1).astype(int)
    df['rain_tomorrow'] = df.groupby('station_id')['rain_today'].shift(-1)

    # Tworzymy cechy opóźnione (lag features) - dane z przeszłości
    # Pomagają modelowi zauważyć trendy czasowe i zależności
    for col in ['precip','tmax','tmin','tavg','rain_today']:
        # lag1 = wartość z wczoraj (shift(1) = przesunięcie o 1 dzień w przeszłość)
        df[f'{col}_lag1'] = df.groupby('station_id')[col].shift(1)
        # lag7 = wartość sprzed tygodnia (shift(7) = przesunięcie o 7 dni)
        df[f'{col}_lag7'] = df.groupby('station_id')[col].shift(7)

    # Dodajemy cechy związane z sezonowością (pora roku wpływa na opady)
    df['month'] = df['date'].dt.month              # Miesiąc (1-12)
    df['day_of_year'] = df['date'].dt.dayofyear    # Dzień roku (1-365/366)
    df['year'] = df['date'].dt.year                # Rok (do podziału train/test)

    # Usuwamy rekordy z brakującymi wartościami
    # - Pierwsze 7 dni każdej stacji nie mają lag7, więc je pomijamy
    # - Ostatni dzień nie ma rain_tomorrow (nie wiemy co będzie jutro)
    df = df.dropna(subset=['rain_tomorrow','precip_lag1','tmax_lag1','tmin_lag1','tavg_lag1']).copy()
    df['rain_tomorrow'] = df['rain_tomorrow'].astype(int)  # Upewniamy się że target to int (0 lub 1)
    return df

def feature_cols(df):
    """Zwraca listę kolumn do użycia jako cechy (features) w modelu."""
    # Pomijamy kolumny, które nie powinny być cechami:
    # - 'date', 'year' - informacje o czasie (mamy już month, day_of_year)
    # - 'station_id', 'station_name' - identyfikatory, nie cechy numeryczne
    # - 'rain_tomorrow' - to jest target (to co przewidujemy), nie cecha!
    skip = {'date','year','station_id','station_name','rain_tomorrow'}
    return [c for c in df.columns if c not in skip]

# ====== TRENING / WALIDACJA ======
def train_and_evaluate(data: pd.DataFrame):
    """
    Główna funkcja trenowania i oceny modelu.
    
    Proces:
    1. Podział danych na train/test według lat
    2. Optymalizacja hiperparametrów (GridSearchCV + TimeSeriesSplit na train)
    3. Trening finalnego modelu na całym zbiorze treningowym
    4. Ewaluacja na zbiorze testowym
    5. Zapisanie wyników, wykresów i modelu
    """
    # KROK 1: Podział danych na zbiór treningowy i testowy według lat
    train = data[data['year'].isin(TRAIN_YEARS)].copy()  # np. 2019-2023
    test  = data[data['year'].isin(TEST_YEARS)].copy()   # np. 2024

    # Przygotowanie cech (X) i targetów (y)
    feats = feature_cols(data)  # Lista kolumn-cech do użycia
    # X = cechy (features), y = target (rain_tomorrow)
    Xtr, ytr = train[feats].fillna(0.0), train['rain_tomorrow'].values  # Dane treningowe
    Xte, yte = test [feats].fillna(0.0),  test ['rain_tomorrow'].values  # Dane testowe
    
    # KROK 2: Optymalizacja hiperparametrów za pomocą GridSearchCV
    # Definiujemy siatkę parametrów do przeszukania (2×3×2×2×2 = 48 kombinacji)
    param_grid = {
        'max_iter': [100, 150],              # Liczba drzew (iteracji boosting)
        'learning_rate': [0.08, 0.1, 0.15],  # Tempo uczenia (im mniejsze, tym ostrożniejsze)
        'max_depth': [5, 6],                 # Maksymalna głębokość drzewa
        'min_samples_leaf': [20, 40],        # Min. próbek w liściu (zapobiega przeuczeniu)
        'l2_regularization': [0.0, 0.1]      # Regularyzacja L2 (kontroluje złożoność)
    }
    
    # Model bazowy do optymalizacji
    base_model = HistGradientBoostingClassifier(random_state=42)  # random_state dla powtarzalności
    # TimeSeriesSplit - walidacja czasowa (szanuje kolejność danych w czasie)
    tss = TimeSeriesSplit(n_splits=3)  # 3 podziały train/validation
    
    # GridSearchCV - automatycznie przeszukuje wszystkie kombinacje parametrów
    grid_search = GridSearchCV(
        estimator=base_model,          # Model do optymalizacji
        param_grid=param_grid,         # Siatka parametrów
        cv=tss,                        # Strategia walidacji krzyżowej
        scoring='roc_auc',             # Metryka do optymalizacji (pole pod krzywą ROC)
        n_jobs=-1,                     # Użyj wszystkich dostępnych rdzeni CPU
        verbose=1,                     # Wyświetlaj postęp
        return_train_score=True        # Zwracaj też wyniki na zbiorze treningowym
    )
    
    # Uruchamiamy przeszukiwanie - to może potrwać kilka minut
    grid_search.fit(Xtr, ytr)
    
    # Wyświetlamy najlepsze znalezione parametry
    print("\n✅ Najlepsze parametry:")
    for param, value in grid_search.best_params_.items():
        print(f"   {param}: {value}")
    print(f"\n🎯 Najlepszy wynik CV (ROC AUC): {grid_search.best_score_:.4f}")
    
    # KROK 3: Obliczamy dodatkowe metryki dla najlepszego modelu
    # (GridSearchCV domyślnie zwraca tylko ROC AUC, my chcemy więcej metryk)
    best_model = grid_search.best_estimator_
    cv_scores = {"roc_auc":[], "pr_auc":[], "f1":[], "precision":[], "recall":[]}  # Słownik na wyniki
    
    print("\n📊 Ocena najlepszego modelu na CV (5-fold)...")
    # Przeprowadzamy 5-krotną walidację krzyżową z najlepszymi parametrami
    for tr_idx, val_idx in TimeSeriesSplit(n_splits=5).split(Xtr):
        # Trenujemy model na części treningowej foldów
        m = HistGradientBoostingClassifier(**grid_search.best_params_, random_state=42)
        m.fit(Xtr.iloc[tr_idx], ytr[tr_idx])
        # Predykcje na części walidacyjnej
        p = m.predict_proba(Xtr.iloc[val_idx])[:,1]  # Prawdopodobieństwa klasy 1 (opad)
        pred = (p>=0.5).astype(int)                   # Konwersja prawdopodobieństw na klasy (próg 0.5)
        # Obliczamy różne metryki
        cv_scores["roc_auc"].append(roc_auc_score(ytr[val_idx], p))              # ROC AUC
        cv_scores["pr_auc"].append(average_precision_score(ytr[val_idx], p))     # Precision-Recall AUC
        cv_scores["f1"].append(f1_score(ytr[val_idx], pred))                     # F1-score
        cv_scores["precision"].append(precision_score(ytr[val_idx], pred))       # Precyzja
        cv_scores["recall"].append(recall_score(ytr[val_idx], pred))             # Czułość (recall)


    # KROK 4: Trenujemy finalny model na CAŁYM zbiorze treningowym
    print("\n🔧 Trenowanie finalnego modelu z najlepszymi parametrami...")
    model = HistGradientBoostingClassifier(**grid_search.best_params_, random_state=42)
    model.fit(Xtr, ytr)  # Tym razem używamy wszystkich danych treningowych

    # KROK 5: Ewaluacja na zbiorze testowym (dane nieużywane w treningu!)
    p_test = model.predict_proba(Xte)[:,1]  # Prawdopodobieństwa dla klasy 1 (opad)
    y_pred = (p_test>=0.5).astype(int)      # Konwersja na predykcje binarne (próg 0.5)

    # Wyświetlamy szczegółowy raport klasyfikacji
    print("\n📈 WYNIKI TEST:")
    # classification_report pokazuje precision, recall, f1-score dla każdej klasy
    print(classification_report(yte, y_pred, target_names=['Brak opadu','Opad']))
    print(f"🎯 ROC AUC (test): {roc_auc_score(yte, p_test):.3f}")

    # Macierz pomyłek (confusion matrix) - pokazuje błędy modelu
    cm = confusion_matrix(yte, y_pred)
    # Tworzymy wizualizację jako heatmapę
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Brak opadu','Opad'], yticklabels=['Brak opadu','Opad'])
    plt.title('Macierz pomyłek - Test'); plt.ylabel('Rzeczywiste'); plt.xlabel('Przewidziane')
    os.makedirs(OUT_DIR, exist_ok=True)
    cm_path = os.path.join(OUT_DIR, 'confusion_matrix.png')
    plt.tight_layout(); plt.savefig(cm_path, dpi=300, bbox_inches='tight'); plt.close()
    print(f"💾 Macierz pomyłek zapisana: {cm_path}")

    # KROK 6: Analiza wyników per stacja (dla każdej stacji osobno)
    # Dodajemy predykcje do danych testowych
    test_results = test.copy()
    test_results["predicted"] = y_pred          # Przewidywania modelu (0 lub 1)
    test_results["probability"] = p_test        # Prawdopodobieństwa klasy 1
    test_results["correct"] = (test_results["rain_tomorrow"] == test_results["predicted"]).astype(int)  # Czy poprawne?

    # Grupujemy wyniki według stacji i obliczamy statystyki
    per_station = (
        test_results.groupby(["station_id","station_name"])
        .agg(
            n=("correct","size"),              # Liczba przewidywań dla tej stacji
            acc=("correct","mean"),            # Dokładność (accuracy) dla tej stacji
            pos_rate=("rain_tomorrow","mean")  # Odsetek dni z opadem (balans klas)
        ).reset_index().sort_values("acc", ascending=False)  # Sortujemy od najlepszej dokładności
    )
    # Zaokrąglamy dla czytelności
    per_station["acc"] = per_station["acc"].round(4)
    per_station["pos_rate"] = per_station["pos_rate"].round(4)

    # KROK 7: Zapisywanie wyników do plików
    
    # 1. Metryki w formacie JSON (najlepsze parametry + wyniki CV i TEST)
    with open(os.path.join(OUT_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({
            "best_params": grid_search.best_params_,  # Optymalne hiperparametry znalezione przez GridSearch
            "cv_mean": cv_summary,                    # Średnie wyniki z walidacji krzyżowej
            "test": {                                 # Wyniki na zbiorze testowym
                "roc_auc": float(roc_auc_score(yte, p_test)),
                "pr_auc": float(average_precision_score(yte, p_test)),
                "f1": float(f1_score(yte, y_pred)),
                "precision": float(precision_score(yte, y_pred)),
                "recall": float(recall_score(yte, y_pred)),
                "accuracy": float((y_pred == yte).mean())
            }
        }, f, indent=2, ensure_ascii=False)

    # 2. Szczegółowy raport klasyfikacji (txt)
    with open(os.path.join(OUT_DIR, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(classification_report(yte, y_pred, target_names=['Brak opadu','Opad'], digits=4))

    # 3. Statystyki per stacja (CSV)
    per_station.to_csv(os.path.join(OUT_DIR, "per_station_summary.csv"), index=False, encoding="utf-8")
    
    # 4. Wszystkie szczegółowe predykcje (CSV) - każdy dzień każdej stacji
    test_results[['date','station_id','station_name','rain_tomorrow','predicted','probability','correct']].to_csv(
        os.path.join(OUT_DIR, "predictions_full.csv"), index=False, encoding="utf-8"
    )

    # 5. Wytrenowany model (można go później załadować i używać)
    dump(model, os.path.join(OUT_DIR, "model.joblib"))

    # Wyświetlamy podsumowanie zapisanych plików
    print("\n📁 Zapisano w:", OUT_DIR)
    print("  - model.joblib (wytrenowany model)")
    print("  - metrics.json (CV + TEST + najlepsze parametry)")
    print("  - classification_report.txt (szczegółowy raport klasyfikacji)")
    print("  - confusion_matrix.png (wizualizacja macierzy pomyłek)")
    print("  - per_station_summary.csv (skuteczność per stacja)")
    print("  - predictions_full.csv (szczegółowe predykcje)")

    return model


def main():
    # Tworzymy katalog na wyniki (jeśli nie istnieje)
    os.makedirs(OUT_DIR, exist_ok=True)

    # ETAP 1: Wczytanie danych z plików CSV
    print("\n📂 Wczytywanie danych...")
    raw, _stations = load_csv_files(TRAIN_YEARS + TEST_YEARS)

    # ETAP 2: Budowa cech dla każdej stacji (lagi, sezonowość, target)
    print("\n🔨 Budowa cech per stacja...")
    data = build_features_per_station(raw)
    print(f"  ✓ Zbudowano {len(data)} rekordów ze stacji={data['station_id'].nunique()}")

    # ETAP 3: Trening, walidacja, optymalizacja i ewaluacja modelu
    _ = train_and_evaluate(data)


# Punkt wejścia programu - uruchamia funkcję main() gdy skrypt jest uruchomiony bezpośrednio
if __name__ == "__main__":
    main()
