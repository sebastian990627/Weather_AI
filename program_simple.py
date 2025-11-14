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
    Oczekiwany układ (jak w Twoim kodzie):
      0: station_id
      1: station_name
      2: rok
      3: miesiąc
      4: dzień
      14: opad [mm]
       5: tmax, 7: tmin, 9: tavg  (częsty układ IMGW)
    """
    all_data = []
    all_stations = {}

    for year in years:
        year_dir = os.path.join(WORK_DIR, str(year))
        if not os.path.exists(year_dir):
            continue
        csv_files = [f for f in os.listdir(year_dir) if f.endswith('.csv')]
        for csv_file in csv_files:
            path = os.path.join(year_dir, csv_file)
            try:
                df = pd.read_csv(path, header=None, encoding='latin1')
                if df.shape[1] < 15:
                    continue

                df['station_id']   = df[0].astype(str).str.strip().str.replace('"', '')
                df['station_name'] = df[1].astype(str).str.strip().str.replace('"', '')

                # data z kolumn 2/3/4
                df['date'] = pd.to_datetime(
                    df[2].astype(str) + '-' + df[3].astype(str).str.zfill(2) + '-' + df[4].astype(str).str.zfill(2),
                    errors='coerce'
                )

                # meteo
                df['precip'] = pd.to_numeric(df[14], errors='coerce').fillna(0.0)
                df['tmax']   = pd.to_numeric(df[5],  errors='coerce')
                df['tmin']   = pd.to_numeric(df[7],  errors='coerce')
                df['tavg']   = pd.to_numeric(df[9],  errors='coerce')

                # słownik stacji (info)
                for sid, sname in zip(df['station_id'].unique(), df['station_name'].unique()):
                    if sid not in all_stations:
                        all_stations[sid] = sname

                df = df[['station_id','station_name','date','precip','tmax','tmin','tavg']]
                df = df.dropna(subset=['date'])
                all_data.append(df)
            except Exception:
                continue

    if not all_data:
        raise RuntimeError("Nie znaleziono żadnych danych wejściowych w WORK_DIR.")

    print(f"  ✓ Znaleziono {len(all_stations)} unikalnych stacji (pierwsze 10):")
    for sid, sname in list(sorted(all_stations.items()))[:10]:
        print(f"    - {sid}: {sname}")
    if len(all_stations) > 10:
        print(f"    ... i {len(all_stations)-10} więcej")

    data = pd.concat(all_data, ignore_index=True)
    # porządek i deduplikacja (gdyby były duplikaty w plikach)
    data = data.sort_values(['station_id','date']).drop_duplicates(subset=['station_id','date'])
    return data, all_stations

# ====== CECHY (PER STACJA) ======
def build_features_per_station(df: pd.DataFrame) -> pd.DataFrame:
    """
    Budujemy cechy dla KAŻDEJ stacji osobno, bez mieszania z innymi stacjami.
    Target: rain_tomorrow (czy jutro opad >= 0.1 mm w TEJ stacji).
    """
    df = df.sort_values(['station_id','date']).copy()

    # bazowe
    df['precip'] = df['precip'].fillna(0.0)
    df['tmax']   = df['tmax'].astype(float)
    df['tmin']   = df['tmin'].astype(float)
    df['tavg']   = df['tavg'].astype(float)

    # dzisiejszy opad binarny + target jutro
    df['rain_today'] = (df['precip'] >= 0.1).astype(int)
    df['rain_tomorrow'] = df.groupby('station_id')['rain_today'].shift(-1)

    # lags per stacja
    for col in ['precip','tmax','tmin','tavg','rain_today']:
        df[f'{col}_lag1'] = df.groupby('station_id')[col].shift(1)
        df[f'{col}_lag7'] = df.groupby('station_id')[col].shift(7)

    # sezonowość
    df['month'] = df['date'].dt.month
    df['day_of_year'] = df['date'].dt.dayofyear
    df['year'] = df['date'].dt.year

    # usuń rekordy bez targetu/lagów
    df = df.dropna(subset=['rain_tomorrow','precip_lag1','tmax_lag1','tmin_lag1','tavg_lag1']).copy()
    df['rain_tomorrow'] = df['rain_tomorrow'].astype(int)
    return df

def feature_cols(df):
    skip = {'date','year','station_id','station_name','rain_tomorrow'}
    return [c for c in df.columns if c not in skip]

# ====== TRENING / WALIDACJA ======
def train_and_evaluate(data: pd.DataFrame):
    """
    - Walidacja czasowa na TRAIN (TimeSeriesSplit)
    - Trening na całym TRAIN, ocena na TEST (lata z TEST_YEARS)
    """
    train = data[data['year'].isin(TRAIN_YEARS)].copy()
    test  = data[data['year'].isin(TEST_YEARS)].copy()

    if train.empty or test.empty:
        raise RuntimeError("Za mało danych w train/test (sprawdź TRAIN_YEARS/TEST_YEARS).")

    feats = feature_cols(data)
    Xtr, ytr = train[feats].fillna(0.0), train['rain_tomorrow'].values
    Xte, yte = test [feats].fillna(0.0),  test ['rain_tomorrow'].values

    print(f"\n📊 Rozmiar danych: train={len(train)}, test={len(test)}, stacji={data['station_id'].nunique()}")

    # Optymalizacja hiperparametrów z GridSearchCV
    print("\n🔍 Optymalizacja hiperparametrów (GridSearchCV + TimeSeriesSplit)...")
    print("   Testowanie ~36 kombinacji...")
    
    # Siatka parametrów - zoptymalizowana pod szybkość
    param_grid = {
        'max_iter': [100, 150],
        'learning_rate': [0.08, 0.1, 0.15],
        'max_depth': [5, 6],
        'min_samples_leaf': [20, 40],
        'l2_regularization': [0.0, 0.1]
    }
    
    base_model = HistGradientBoostingClassifier(random_state=42)
    tss = TimeSeriesSplit(n_splits=3)
    
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=tss,
        scoring='roc_auc',
        n_jobs=-1,  # Użyj wszystkich dostępnych rdzeni
        verbose=1,
        return_train_score=True
    )
    
    grid_search.fit(Xtr, ytr)
    
    print("\n✅ Najlepsze parametry:")
    for param, value in grid_search.best_params_.items():
        print(f"   {param}: {value}")
    print(f"\n🎯 Najlepszy wynik CV (ROC AUC): {grid_search.best_score_:.4f}")
    
    # Oblicz dodatkowe metryki dla najlepszego modelu
    best_model = grid_search.best_estimator_
    cv_scores = {"roc_auc":[], "pr_auc":[], "f1":[], "precision":[], "recall":[]}
    
    print("\n📊 Ocena najlepszego modelu na CV...")
    for tr_idx, val_idx in TimeSeriesSplit(n_splits=5).split(Xtr):
        m = HistGradientBoostingClassifier(**grid_search.best_params_, random_state=42)
        m.fit(Xtr.iloc[tr_idx], ytr[tr_idx])
        p = m.predict_proba(Xtr.iloc[val_idx])[:,1]
        pred = (p>=0.5).astype(int)
        cv_scores["roc_auc"].append(roc_auc_score(ytr[val_idx], p))
        cv_scores["pr_auc"].append(average_precision_score(ytr[val_idx], p))
        cv_scores["f1"].append(f1_score(ytr[val_idx], pred))
        cv_scores["precision"].append(precision_score(ytr[val_idx], pred))
        cv_scores["recall"].append(recall_score(ytr[val_idx], pred))

    cv_summary = {k: float(np.mean(v)) for k, v in cv_scores.items()}
    print("  Średnie wyniki CV:", json.dumps(cv_summary, ensure_ascii=False, indent=2))

    # Trenuj na całym train z najlepszymi parametrami
    print("\n🔧 Trenowanie finalnego modelu z najlepszymi parametrami...")
    model = HistGradientBoostingClassifier(**grid_search.best_params_, random_state=42)
    model.fit(Xtr, ytr)

    p_test = model.predict_proba(Xte)[:,1]
    y_pred = (p_test>=0.5).astype(int)

    print("\n📈 WYNIKI TEST:")
    print(classification_report(yte, y_pred, target_names=['Brak opadu','Opad']))
    print(f"🎯 ROC AUC (test): {roc_auc_score(yte, p_test):.3f}")

    # Macierz pomyłek
    cm = confusion_matrix(yte, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Brak opadu','Opad'], yticklabels=['Brak opadu','Opad'])
    plt.title('Macierz pomyłek - Test'); plt.ylabel('Rzeczywiste'); plt.xlabel('Przewidziane')
    os.makedirs(OUT_DIR, exist_ok=True)
    cm_path = os.path.join(OUT_DIR, 'confusion_matrix.png')
    plt.tight_layout(); plt.savefig(cm_path, dpi=300, bbox_inches='tight'); plt.close()
    print(f"💾 Macierz pomyłek zapisana: {cm_path}")

    # Raport per stacja (ile próbek i skuteczność)
    test_results = test.copy()
    test_results["predicted"] = y_pred
    test_results["probability"] = p_test
    test_results["correct"] = (test_results["rain_tomorrow"] == test_results["predicted"]).astype(int)

    per_station = (
        test_results.groupby(["station_id","station_name"])
        .agg(
            n=("correct","size"),
            acc=("correct","mean"),
            pos_rate=("rain_tomorrow","mean")
        ).reset_index().sort_values("acc", ascending=False)
    )
    per_station["acc"] = per_station["acc"].round(4)
    per_station["pos_rate"] = per_station["pos_rate"].round(4)

    # Zapisy
    with open(os.path.join(OUT_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({
            "best_params": grid_search.best_params_,
            "cv_mean": cv_summary,
            "test": {
                "roc_auc": float(roc_auc_score(yte, p_test)),
                "pr_auc": float(average_precision_score(yte, p_test)),
                "f1": float(f1_score(yte, y_pred)),
                "precision": float(precision_score(yte, y_pred)),
                "recall": float(recall_score(yte, y_pred)),
                "accuracy": float((y_pred == yte).mean())
            }
        }, f, indent=2, ensure_ascii=False)

    with open(os.path.join(OUT_DIR, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(classification_report(yte, y_pred, target_names=['Brak opadu','Opad'], digits=4))

    per_station.to_csv(os.path.join(OUT_DIR, "per_station_summary.csv"), index=False, encoding="utf-8")
    test_results[['date','station_id','station_name','rain_tomorrow','predicted','probability','correct']].to_csv(
        os.path.join(OUT_DIR, "predictions_full.csv"), index=False, encoding="utf-8"
    )

    # Model
    dump(model, os.path.join(OUT_DIR, "model.joblib"))

    print("\n📁 Zapisano w:", OUT_DIR)
    print("  - model.joblib")
    print("  - metrics.json (CV + TEST)")
    print("  - classification_report.txt")
    print("  - confusion_matrix.png")
    print("  - per_station_summary.csv (skuteczność per stacja)")
    print("  - predictions_full.csv (szczegółowe predykcje)")

    return model


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("\n📂 Wczytywanie danych...")
    raw, _stations = load_csv_files(TRAIN_YEARS + TEST_YEARS)

    print("\n🔨 Budowa cech per stacja...")
    data = build_features_per_station(raw)
    print(f"  ✓ Zbudowano {len(data)} rekordów ze stacji={data['station_id'].nunique()}")

    _ = train_and_evaluate(data)



if __name__ == "__main__":
    main()
