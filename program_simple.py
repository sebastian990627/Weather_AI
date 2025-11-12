# -*- coding: utf-8 -*-
"""
Prosty program: przewidywanie opadów w województwach Polski
- Wczytuje dane z lokalnych plików CSV (IMGW SYNOP)
- Mapuje stacje na województwa
- Trenuje model klasyfikacji (czy jutro będzie padać?)
- Ocenia wyniki na zbiorze testowym
"""

import os
import pandas as pd
import numpy as np
import geopandas as gpd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from joblib import dump
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ====== KONFIGURACJA ======
WORK_DIR = "work_simple"
OUT_DIR = "out_simple"
GEOJSON_PATH = "wojewodztwa-min.geojson"
TRAIN_YEARS = [2019, 2020, 2021, 2022, 2023]
TEST_YEARS = [2024]


def load_csv_files(years):
    """Wczytuje pliki CSV z podanych lat"""
    all_data = []
    all_stations = {}  # Do zbierania współrzędnych
    
    for year in years:
        year_dir = os.path.join(WORK_DIR, str(year))
        if not os.path.exists(year_dir):
            continue
            
        csv_files = [f for f in os.listdir(year_dir) if f.endswith('.csv')]
        
        for csv_file in csv_files:
            path = os.path.join(year_dir, csv_file)
            try:
                # Pliki bez nagłówków, format: kod, nazwa, rok, miesiąc, dzień, dane...
                df = pd.read_csv(path, header=None, encoding='latin1')
                
                if len(df.columns) < 15:
                    continue
                
                # Podstawowe kolumny
                df['station_id'] = df[0].astype(str).str.strip().str.replace('"', '')
                df['station_name'] = df[1].astype(str).str.strip().str.replace('"', '')
                df['date'] = pd.to_datetime(
                    df[2].astype(str) + '-' + 
                    df[3].astype(str).str.zfill(2) + '-' + 
                    df[4].astype(str).str.zfill(2),
                    errors='coerce'
                )
                df['precip'] = pd.to_numeric(df[14], errors='coerce').fillna(0)
                df['tmax'] = pd.to_numeric(df[5], errors='coerce')
                df['tmin'] = pd.to_numeric(df[7], errors='coerce')
                df['tavg'] = pd.to_numeric(df[9], errors='coerce')
                
                # Zbierz unikalne stacje
                for sid, sname in zip(df['station_id'].unique(), df['station_name'].unique()):
                    if sid not in all_stations:
                        all_stations[sid] = sname
                
                # Zachowaj tylko potrzebne kolumny
                df = df[['station_id', 'station_name', 'date', 'precip', 'tmax', 'tmin', 'tavg']]
                df = df.dropna(subset=['date'])
                
                all_data.append(df)
                
            except Exception:
                continue
    
    if not all_data:
        raise Exception("Nie znaleziono żadnych danych!")
    
    print(f"  ✓ Znaleziono {len(all_stations)} unikalnych stacji:")
    for sid, sname in sorted(all_stations.items())[:10]:
        print(f"    - {sid}: {sname}")
    if len(all_stations) > 10:
        print(f"    ... i {len(all_stations)-10} więcej")
    
    return pd.concat(all_data, ignore_index=True), all_stations


def map_stations_to_voivodeships(data, all_stations, geojson_path):
    """Mapuje stacje meteorologiczne na województwa - rozdziela równomiernie"""
    # Wczytaj granice województw
    gdf_woj = gpd.read_file(geojson_path)
    
    # Znajdź kolumnę z nazwą województwa
    name_col = None
    for col in ['name', 'nazwa', 'jpt_nazwa_', 'JPT_NAZWA_']:
        if col in gdf_woj.columns:
            name_col = col
            break
    if name_col is None:
        name_col = [c for c in gdf_woj.columns if c != 'geometry'][0]
    
    wojewodztwa = sorted(gdf_woj[name_col].tolist())
    print(f"\n  📍 Mapowanie {len(all_stations)} stacji na {len(wojewodztwa)} województw...")
    print(f"  📍 Stacje zostaną rozdzielone równomiernie między województwa")
    
    # Przypisz stacje równomiernie do województw
    station_to_woj = {}
    sorted_stations = sorted(all_stations.keys())
    
    for i, sid in enumerate(sorted_stations):
        woj = wojewodztwa[i % len(wojewodztwa)]
        station_to_woj[sid] = woj
    
    # Pokaż statystyki
    from collections import Counter
    woj_counts = Counter(station_to_woj.values())
    print(f"\n  ✓ Rozkład stacji po województwach:")
    for woj in sorted(wojewodztwa):
        count = woj_counts.get(woj, 0)
        print(f"    - {woj}: {count} stacji")
    
    return station_to_woj


def create_features(data, station_to_woj):
    """Tworzy cechy dla modelu"""
    # Dodaj województwo
    data['voivodeship'] = data['station_id'].map(station_to_woj)
    data = data.dropna(subset=['voivodeship'])
    
    # Czy padało (>= 0.1mm)
    data['rain_today'] = (data['precip'] >= 0.1).astype(int)
    
    # Agregacja do poziomu województwa (średnia dzienna)
    daily = data.groupby(['voivodeship', 'date']).agg({
        'precip': 'mean',
        'tmax': 'mean',
        'tmin': 'mean',
        'tavg': 'mean',
        'rain_today': 'max'  # czy padało w JAKIEJKOLWIEK stacji w województwie
    }).reset_index()
    
    # Target: czy jutro będzie padać
    daily = daily.sort_values(['voivodeship', 'date'])
    daily['rain_tomorrow'] = daily.groupby('voivodeship')['rain_today'].shift(-1)
    daily = daily.dropna(subset=['rain_tomorrow'])
    
    # Cechy z przeszłości
    for col in ['precip', 'tmax', 'tmin', 'tavg', 'rain_today']:
        daily[f'{col}_lag1'] = daily.groupby('voivodeship')[col].shift(1)
        daily[f'{col}_lag7'] = daily.groupby('voivodeship')[col].shift(7)
    
    # Usuń wiersze z brakującymi wartościami
    daily = daily.dropna()
    
    # Dodaj informacje o dacie
    daily['month'] = daily['date'].dt.month
    daily['day_of_year'] = daily['date'].dt.dayofyear
    daily['year'] = daily['date'].dt.year
    
    return daily


def train_and_evaluate(data):
    """Trenuje model i ocenia wyniki"""
    # Podział train/test
    train = data[data['year'].isin(TRAIN_YEARS)]
    test = data[data['year'].isin(TEST_YEARS)]
    
    print(f"\n📊 Rozmiar danych:")
    print(f"  Train: {len(train)} rekordów")
    print(f"  Test:  {len(test)} rekordów")
    print(f"  Województw: {data['voivodeship'].nunique()}")
    
    wojewodztwa = sorted(data['voivodeship'].unique())
    print(f"\n  Województwa w danych:")
    for woj in wojewodztwa:
        n_train = len(train[train['voivodeship'] == woj])
        n_test = len(test[test['voivodeship'] == woj])
        print(f"    - {woj}: train={n_train}, test={n_test}")
    
    # Kolumny do trenowania
    feature_cols = [
        'precip', 'tmax', 'tmin', 'tavg', 'rain_today',
        'precip_lag1', 'tmax_lag1', 'tmin_lag1', 'tavg_lag1', 'rain_today_lag1',
        'precip_lag7', 'tmax_lag7', 'tmin_lag7', 'tavg_lag7', 'rain_today_lag7',
        'month', 'day_of_year'
    ]
    
    X_train = train[feature_cols]
    y_train = train['rain_tomorrow']
    X_test = test[feature_cols]
    y_test = test['rain_tomorrow']
    
    # Trenowanie
    print("\n🔧 Trenowanie modelu...")
    model = HistGradientBoostingClassifier(
        max_iter=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Predykcje
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    y_proba_test = model.predict_proba(X_test)[:, 1]
    
    # Metryki
    print("\n📈 WYNIKI TRAIN:")
    print(classification_report(y_train, y_pred_train, target_names=['Brak opadu', 'Opad']))
    
    print("\n📈 WYNIKI TEST:")
    print(classification_report(y_test, y_pred_test, target_names=['Brak opadu', 'Opad']))
    
    # Dodatkowe metryki
    print(f"\n🎯 Dodatkowe metryki TEST:")
    print(f"  ROC AUC: {roc_auc_score(y_test, y_proba_test):.3f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred_test)
    print(f"\n📊 Macierz pomyłek TEST:")
    print(f"                Predykcja")
    print(f"                Brak  Opad")
    print(f"  Rzecz. Brak    {cm[0,0]:4d}  {cm[0,1]:4d}")
    print(f"         Opad    {cm[1,0]:4d}  {cm[1,1]:4d}")
    
    # Wizualizacja macierzy pomyłek
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Brak opadu', 'Opad'],
                yticklabels=['Brak opadu', 'Opad'])
    plt.title('Macierz pomyłek - Test', fontsize=14, fontweight='bold')
    plt.ylabel('Rzeczywiste', fontsize=12)
    plt.xlabel('Przewidziane', fontsize=12)
    plt.tight_layout()
    confusion_path = os.path.join(OUT_DIR, 'confusion_matrix.png')
    plt.savefig(confusion_path, dpi=300, bbox_inches='tight')
    print(f"\n� Macierz pomyłek zapisana: {confusion_path}")
    plt.close()
    
    # Zapisz szczegółowe przewidywania
    test_results = test.copy()
    test_results['predicted'] = y_pred_test
    test_results['probability'] = y_proba_test
    test_results['correct'] = (test_results['rain_tomorrow'] == test_results['predicted']).astype(int)
    
    # Raport po województwach
    report_path = os.path.join(OUT_DIR, 'predictions_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("RAPORT PRZEWIDYWAŃ OPADÓW - WOJEWÓDZTWA POLSKI\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Okres testowy: {test['date'].min()} - {test['date'].max()}\n")
        f.write(f"Liczba predykcji: {len(test_results)}\n")
        f.write(f"Dokładność ogólna: {(test_results['correct'].sum() / len(test_results) * 100):.2f}%\n")
        f.write(f"ROC AUC: {roc_auc_score(y_test, y_proba_test):.3f}\n\n")
        
        f.write("-" * 80 + "\n")
        f.write("STATYSTYKI PO WOJEWÓDZTWACH\n")
        f.write("-" * 80 + "\n\n")
        
        for woj in sorted(test_results['voivodeship'].unique()):
            woj_data = test_results[test_results['voivodeship'] == woj]
            accuracy = (woj_data['correct'].sum() / len(woj_data) * 100)
            
            f.write(f"\n{'='*60}\n")
            f.write(f"WOJEWÓDZTWO: {woj}\n")
            f.write(f"{'='*60}\n")
            f.write(f"Liczba dni: {len(woj_data)}\n")
            f.write(f"Dokładność: {accuracy:.2f}%\n")
            f.write(f"Dni z opadem (rzeczywiste): {woj_data['rain_tomorrow'].sum()}\n")
            f.write(f"Dni z opadem (przewidziane): {woj_data['predicted'].sum()}\n\n")
            
            f.write("Przykładowe predykcje:\n")
            f.write("-" * 60 + "\n")
            f.write(f"{'Data':<12} {'Rzeczyw.':<10} {'Predykc.':<10} {'Pewność':<10} {'Status'}\n")
            f.write("-" * 60 + "\n")
            
            for _, row in woj_data.head(20).iterrows():
                rzecz = "OPAD" if row['rain_tomorrow'] == 1 else "BRAK"
                pred = "OPAD" if row['predicted'] == 1 else "BRAK"
                status = "✓" if row['correct'] == 1 else "✗"
                f.write(f"{str(row['date'])[:10]:<12} {rzecz:<10} {pred:<10} {row['probability']:.3f}      {status}\n")
            
            if len(woj_data) > 20:
                f.write(f"... i {len(woj_data)-20} więcej dni\n")
    
    print(f"💾 Raport szczegółowy zapisany: {report_path}")
    
    # Zapisz pełne przewidywania do CSV
    predictions_csv = os.path.join(OUT_DIR, 'predictions_full.csv')
    test_results[['date', 'voivodeship', 'rain_tomorrow', 'predicted', 'probability', 'correct']].to_csv(
        predictions_csv, index=False, encoding='utf-8'
    )
    print(f"💾 Pełne przewidywania zapisane: {predictions_csv}")
    
    # Zapisz model
    model_path = os.path.join(OUT_DIR, 'model.joblib')
    dump(model, model_path)
    print(f"💾 Model zapisany: {model_path}")
    
    # Zapisz metryki
    metrics = {
        'train_size': len(train),
        'test_size': len(test),
        'test_roc_auc': float(roc_auc_score(y_test, y_proba_test)),
        'test_accuracy': float((y_pred_test == y_test).mean()),
        'voivodeships': len(wojewodztwa),
    }
    
    metrics_path = os.path.join(OUT_DIR, 'metrics.json')
    import json
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"💾 Metryki zapisane: {metrics_path}")
    
    return model


def main():
    print("=" * 60)
    print("🌧️  PRZEWIDYWANIE OPADÓW W WOJEWÓDZTWACH POLSKI")
    print("=" * 60)
    
    os.makedirs(OUT_DIR, exist_ok=True)
    
    # 1. Wczytaj dane
    print("\n📂 Wczytywanie danych...")
    data, all_stations = load_csv_files(TRAIN_YEARS + TEST_YEARS)
    print(f"\n  ✓ Wczytano {len(data)} rekordów")
    print(f"  ✓ Zakres dat: {data['date'].min()} - {data['date'].max()}")
    
    # 2. Mapowanie na województwa
    print("\n🗺️  Mapowanie stacji na województwa...")
    station_to_woj = map_stations_to_voivodeships(data, all_stations, GEOJSON_PATH)
    
    # 3. Tworzenie cech
    print("\n🔨 Tworzenie cech...")
    features = create_features(data, station_to_woj)
    print(f"  ✓ Przygotowano {len(features)} rekordów")
    
    # 4. Trenowanie i ocena
    model = train_and_evaluate(features)
    
    print("\n" + "=" * 60)
    print("✅ GOTOWE!")
    print("=" * 60)
    print(f"\n📁 Pliki wyjściowe w katalogu: {OUT_DIR}/")
    print("  - model.joblib - wytrenowany model")
    print("  - confusion_matrix.png - wizualizacja macierzy pomyłek")
    print("  - predictions_report.txt - szczegółowy raport po województwach")
    print("  - predictions_full.csv - pełne przewidywania")
    print("  - metrics.json - metryki modelu")


if __name__ == "__main__":
    main()
