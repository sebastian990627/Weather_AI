# -*- coding: utf-8 -*-
"""
Prosty pipeline: przewidywanie wystąpienia opadu (tak/nie) dla WOJEWÓDZTW w PL
na podstawie dobowych danych SYNOP z IMGW.

Co robi:
- Pobiera CSV z katalogów IMGW dla stałej listy lat
- Parsuje i normalizuje kolumny (tolerancyjnie)
- Mapuje stacje -> województwa (GeoJSON, ścieżka w stałej GEOJSON_WOJ)
- Buduje lekkie cechy + agreguje do poziomu województwa
- Cel: czy jutro (t+1) w województwie wystąpi opad >= 0.1 mm
- Walidacja: TimeSeriesSplit (5-split) na train, raport średnich metryk
- Podział czasowy: train <= TRAIN_UNTIL_YEAR, test > TRAIN_UNTIL_YEAR
- Model: HistGradientBoostingClassifier (bez grid-search)
- Zapis: model.joblib, metrics.json, classification_report.txt, latest_region_forecast.csv

Zależności:
    pip install pandas numpy scikit-learn joblib geopandas shapely requests
"""

import os, io, re, zipfile, json
from datetime import timedelta
import requests
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from sklearn.experimental import enable_hist_gradient_boosting  # noqa: F401
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, classification_report
)
from joblib import dump

# ====== STAŁE (USTAW RAZ I NIE DOTYKAJ) =====================================

YEARS = [2019, 2020, 2021, 2022, 2023, 2024]  # jakie lata pobieramy
TRAIN_UNTIL_YEAR = 2023                         # train: <= ten rok, test: > ten rok
WORK_DIR = "work_simple"                        # tu zapiszą się CSV-y
OUT_DIR = "out_simple"                          # tu zapiszemy model i metryki

# GeoJSON z granicami województw (np. oficjalny plik administracyjny PL)
GEOJSON_WOJ = "wojewodztwa-min.geojson"    # <<< UZUPEŁNIJ TĘ ŚCIEŻKĘ

BASE_URL = "https://danepubliczne.imgw.pl/data/dane_pomiarowo_obserwacyjne/dane_meteorologiczne/dobowe/synop/"
STATIONS_META_URL = "https://danepubliczne.imgw.pl/data/dane_pomiarowo_obserwacyjne/dane_meteorologiczne/dobowe/synop/s_d_format.txt"

# Współrzędne stacji SYNOP (backup - jeśli nie da się pobrać z IMGW)
# Kluczowe stacje meteorologiczne w Polsce (kod stacji: (lat, lon))
STATION_COORDS = {
    "354150100": (54.188, 15.583),  # Kołobrzeg
    "354120300": (54.383, 18.4),    # Gdańsk
    "353230295": (53.417, 14.55),   # Szczecin
    "352230140": (52.167, 20.967),  # Warszawa
    "351210576": (51.117, 17.05),   # Wrocław
    "350160449": (50.067, 19.95),   # Kraków
    "349180120": (49.817, 22.967),  # Rzeszów
    "354240135": (54.352, 18.467),  # Gdynia
    # Dodatkowe stacje z analizy plików
    "354150105": (54.533, 18.533),  # Hel
    "354150115": (54.167, 16.15),   # Koszalin
    "354150120": (54.1, 15.767),    # Świnoujście
    "354150125": (53.917, 14.233),  # Kamień Pomorski
    "354150135": (53.767, 15.55),   # Kołobrzeg-Dźwirzyno
    "353150155": (53.583, 14.617),  # Szczecin
    "353150160": (53.183, 16.667),  # Chojnice
    "353150185": (53.533, 21.933),  # Suwałki
    "353150195": (53.1, 23.15),     # Białystok
    "353150200": (53.567, 19.4),    # Olsztyn
    "352150205": (52.4, 16.917),    # Poznań
    "352150210": (52.167, 22.283),  # Siedlce
    "352150230": (52.4, 16.9),      # Piła
    "352150235": (52.433, 20.717),  # Legionowo
    "352150250": (52.167, 21.0),    # Warszawa-Okęcie
    "351150270": (51.75, 19.4),     # Łódź
    "351150272": (51.817, 15.533),  # Zielona Góra
    "351150280": (51.65, 16.217),   # Leszno
    "351150295": (51.383, 21.55),   # Lublin
    "351150300": (51.25, 22.55),    # Zamość
    "351150310": (51.767, 19.45),   # Sulejów
    "350150330": (50.317, 19.0),    # Częstochowa
    "350150345": (50.25, 19.0),     # Katowice
    "350150360": (50.067, 19.8),    # Kraków-Balice
    "349150375": (49.633, 18.817),  # Bielsko-Biała
    "349150385": (49.417, 20.3),    # Nowy Sącz
    "349150399": (49.0, 22.683),    # Lesko
    "349150400": (49.817, 22.75),   # Rzeszów-Jasionka
    "349150415": (49.633, 22.05),   # Krosno
    "349150418": (49.833, 19.133),  # Kasprowy Wierch
    "349150424": (49.417, 20.433),  # Zakopane
    "349150435": (49.4, 20.9),      # Tarnów
    "350150455": (50.75, 15.75),    # Jelenia Góra
    "351150465": (51.4, 21.617),    # Puławy
    "351150469": (51.283, 22.383),  # Radzyń Podlaski
    "351150488": (51.467, 21.55),   # Kozienice
    "351150495": (51.717, 16.217),  # Kalisz
    "351150497": (51.817, 15.567),  # Gorzów Wielkopolski
    "350150500": (50.25, 19.483),   # Sosnowiec
    "351150510": (51.933, 18.217),  # Koło
    "351150520": (51.417, 21.25),   # Puławy
    "351150530": (51.583, 18.5),    # Łęczyca
    "351150540": (51.867, 20.45),   # Skierniewice
    "351150550": (51.25, 22.517),   # Włodawa
    "352150560": (52.267, 21.05),   # Warszawa-Bielany
    "353150566": (53.417, 20.917),  # Działdowo
    "353150570": (53.833, 20.45),   # Elbląg
    "353150575": (53.017, 18.25),   # Toruń
    "353150580": (53.217, 23.167),  # Augustów
    "352150585": (52.45, 23.883),   # Biała Podlaska
    "353150595": (53.567, 18.067),  # Grudziądz
    "352150600": (52.467, 17.05),   # Konin
    "351150625": (51.2, 16.183),    # Oława
    "351150628": (51.683, 17.917),  # Wieluń
    "350150650": (50.333, 18.533),  # Racibórz
    "350150660": (50.417, 18.95),   # Gliwice
    "350150670": (50.617, 17.05),   # Opole
    "349150690": (49.283, 19.95),   # Zakopane-Kasprowy
}

# ====== POMOCNICY ============================================================

COLUMN_ALIASES = {
    "station_id": ["id_stacji", "id stacji", "id", "kod_stacji", "kod stacji"],
    "station_name": ["stacja", "nazwa_stacji", "nazwa stacji"],
    "lat": ["szerokość_geograficzna", "szerokość geograficzna", "latitude", "lat", "szer"],
    "lon": ["długość_geograficzna", "długość geograficzna", "longitude", "lon", "dlug"],
    "date": ["data", "data_obserwacji", "data pomiaru", "termin", "czas"],
    "precip": ["rr", "rrr", "opad", "opad_mm", "suma_opadu", "opady", "rr [mm]"],
    "tavg": ["tavg", "tśr", "t_sr", "t średnia"],
    "tmin": ["tmin", "t_min"],
    "tmax": ["tmax", "t_max"],
    "pres": ["pp0", "cisnienie", "ciśnienie", "p0", "pres"],
    "rh":   ["wilgotność", "wilgotnosc", "rh"],
    "wind": ["wiatr", "ff", "wind"]
}

def list_zip_urls(year: int):
    idx = requests.get(f"{BASE_URL}{year}/", timeout=60)
    idx.raise_for_status()
    hrefs = re.findall(r'href="([^"]+)"', idx.text)
    urls = []
    for h in hrefs:
        if h.lower().endswith(".zip") or h.lower().endswith(".csv"):
            urls.append(f"{BASE_URL}{year}/{h}")
    return urls

def download_csvs(url: str, to_dir: str):
    os.makedirs(to_dir, exist_ok=True)
    got = []
    r = requests.get(url, timeout=120); r.raise_for_status()
    if url.lower().endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
            for n in zf.namelist():
                if n.lower().endswith(".csv"):
                    p = os.path.join(to_dir, os.path.basename(n))
                    with zf.open(n) as src, open(p, "wb") as dst: dst.write(src.read())
                    got.append(p)
    else:
        p = os.path.join(to_dir, os.path.basename(url))
        with open(p, "wb") as f: f.write(r.content)
        got.append(p)
    return got

def enrich_with_coords(df: pd.DataFrame) -> pd.DataFrame:
    """Uzupełnia brakujące współrzędne geograficzne stacji."""
    if "lat" not in df.columns or "lon" not in df.columns:
        df["lat"] = np.nan
        df["lon"] = np.nan
    
    # Uzupełnij z mapowania backup
    for sid, (lat, lon) in STATION_COORDS.items():
        mask = (df["station_id"].astype(str) == str(sid)) & (df["lat"].isna() | df["lon"].isna())
        df.loc[mask, "lat"] = lat
        df.loc[mask, "lon"] = lon
    
    return df

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols_map = {c.lower().strip(): c for c in df.columns}
    rename = {}
    for std, aliases in COLUMN_ALIASES.items():
        for a in aliases:
            if a.lower().strip() in cols_map:
                rename[cols_map[a.lower().strip()]] = std
                break
    df = df.rename(columns=rename)

    # daty
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["date"] = df["date"].dt.normalize()

    # liczby
    for c in ["precip","tavg","tmin","tmax","pres","rh","wind","lat","lon"]:
        if c in df.columns:
            df[c] = pd.to_numeric(
                df[c].astype(str).str.replace(",", ".", regex=False), errors="coerce"
            )

    # ID stacji fallback
    if "station_id" not in df.columns:
        df["station_id"] = (df.get("station_name") or pd.Series(range(len(df)))).astype("category").cat.codes

    if "station_name" not in df.columns:
        df["station_name"] = None

    df = df.dropna(subset=["date"]).copy()
    keep = ["station_id","station_name","lat","lon","date","precip","tavg","tmin","tmax","pres","rh","wind"]
    for k in list(keep):
        if k not in df.columns: keep.remove(k)
    return df[keep].drop_duplicates(subset=["station_id","date"])

def read_csv_any(path: str):
    # Najpierw sprawdź czy plik ma nagłówki
    try:
        # Spróbuj odczytać z nagłówkiem
        for sep in [",", ";", "; "]:
            for enc in ["utf-8", "cp1250", "latin1"]:
                try:
                    df = pd.read_csv(path, sep=sep, encoding=enc, low_memory=False, nrows=5)
                    if len(df.columns) > 5:  # Jeśli ma sporo kolumn, może ma nagłówki
                        df_full = pd.read_csv(path, sep=sep, encoding=enc, low_memory=False)
                        df_std = standardize_columns(df_full)
                        if "date" in df_std.columns and len(df_std) > 0:
                            return df_std
                except Exception:
                    continue
    except Exception:
        pass
    
    # Jeśli nie ma nagłówków, użyj standardowych nazw kolumn IMGW SYNOP dobowe
    # Format: Kod stacji, Nazwa stacji, Rok, Miesiąc, Dzień, [dane meteorologiczne...]
    try:
        for enc in ["utf-8", "cp1250", "latin1"]:
            try:
                df = pd.read_csv(path, sep=",", encoding=enc, low_memory=False, header=None)
                if len(df) == 0:
                    continue
                # Standardowe kolumny dla IMGW dobowych SYNOP
                # Pierwsze 5 kolumn: kod_stacji, nazwa_stacji, rok, miesiąc, dzień
                if df.shape[1] >= 5:
                    df.columns = [f"col_{i}" for i in range(len(df.columns))]
                    df["station_id"] = df["col_0"]
                    df["station_name"] = df["col_1"]
                    # Buduj datę z roku, miesiąca, dnia
                    df["date"] = pd.to_datetime(
                        df["col_2"].astype(str) + "-" + 
                        df["col_3"].astype(str).str.zfill(2) + "-" + 
                        df["col_4"].astype(str).str.zfill(2),
                        errors="coerce"
                    )
                    # Kolumny z danymi meteorologicznymi (różne w zależności od pliku)
                    # Zazwyczaj: tmax, status, tmin, status, tavg, status, tmin_gruntu, status, suma_opadu, status...
                    # Dla uproszczenia szukamy kolumn numerycznych
                    df["tmax"] = pd.to_numeric(df.get("col_5"), errors="coerce") if df.shape[1] > 5 else np.nan
                    df["tmin"] = pd.to_numeric(df.get("col_7"), errors="coerce") if df.shape[1] > 7 else np.nan
                    df["tavg"] = pd.to_numeric(df.get("col_9"), errors="coerce") if df.shape[1] > 9 else np.nan
                    # Suma opadów jest zazwyczaj w okolicach kolumny 13-14
                    df["precip"] = pd.to_numeric(df.get("col_14"), errors="coerce") if df.shape[1] > 14 else np.nan
                    
                    # Latitude/Longitude - zazwyczaj brak w tych plikach, będzie None
                    df["lat"] = np.nan
                    df["lon"] = np.nan
                    df["pres"] = np.nan
                    df["rh"] = np.nan
                    df["wind"] = np.nan
                    
                    keep = ["station_id","station_name","lat","lon","date","precip","tavg","tmin","tmax","pres","rh","wind"]
                    df = df[keep].dropna(subset=["date"]).copy()
                    if len(df) > 0:
                        return df
            except Exception as e:
                continue
    except Exception:
        pass
    
    print(f"[WARN] Nie można odczytać pliku: {path}")
    return None

def map_to_voivodeships(stations_df: pd.DataFrame, geojson_path: str) -> pd.DataFrame:
    gdf = gpd.read_file(geojson_path)
    name_col = next((c for c in gdf.columns if c.lower() in
                     ["name","nazwa","województwo","wojewodztwo","jpt_nazwa_","jpt_nazwa"]), None)
    if name_col is None:
        name_col = [c for c in gdf.columns if c != "geometry"][0]
    pts = gpd.GeoDataFrame(
        stations_df.copy(),
        geometry=gpd.points_from_xy(stations_df["lon"], stations_df["lat"]),
        crs="EPSG:4326"
    )
    if gdf.crs is None: gdf = gdf.set_crs("EPSG:4326")
    if gdf.crs != pts.crs: gdf = gdf.to_crs(pts.crs)
    joined = gpd.sjoin(pts, gdf[[name_col, "geometry"]], how="left", predicate="within")
    out = pd.DataFrame(joined.drop(columns="geometry")).rename(columns={name_col:"voivodeship"})
    return out[["station_id","voivodeship"]]

# ====== CECHY I AGREGACJA ====================================================

def make_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["station_id","date"]).copy()
    df["precip_mm"] = df["precip"].fillna(0)
    df["rain_today"] = (df["precip_mm"] >= 0.1).astype(int)
    # lag wczoraj
    df["rain_yest"] = df.groupby("station_id")["rain_today"].shift(1)
    # proste rollingi (3/7) dla kilku zmiennych
    for c in ["tavg","tmin","tmax","pres","rh","wind","precip_mm"]:
        for w in [3,7]:
            df[f"{c}_r{w}"] = df.groupby("station_id")[c].transform(lambda s: s.rolling(w, min_periods=1).mean().shift(1))
    # cechy kalendarzowe
    df["doy"] = df["date"].dt.dayofyear
    df["doy_sin"] = np.sin(2*np.pi*df["doy"]/366.0)
    df["doy_cos"] = np.cos(2*np.pi*df["doy"]/366.0)
    df = df.dropna(subset=["rain_yest"])
    return df

def aggregate_to_region(df_feat: pd.DataFrame) -> pd.DataFrame:
    # agregacje do poziomu (voivodeship, date)
    group_cols = ["voivodeship","date"]
    # target: czy jutro popada (max po stacjach) — przesuwamy o -1 dzień
    day_rain = (
        df_feat.assign(r=(df_feat["precip_mm"] >= 0.1).astype(int))
               .groupby(group_cols)["r"].max().reset_index()
    )
    day_rain["date"] = day_rain["date"] - pd.Timedelta(days=1)
    day_rain = day_rain.rename(columns={"r":"rain_tomorrow"})

    # cechy: średnie i maksima z podstawowych + rollingów
    agg = {}
    for c in df_feat.columns:
        if c.startswith(("tavg","tmin","tmax","pres","rh","wind","precip_mm")):
            agg[c] = ["mean","max","min"]
    for c in list(df_feat.columns):
        if "_r3" in c or "_r7" in c: agg[c] = ["mean","max","min"]

    feats = df_feat.groupby(group_cols).agg(agg)
    feats.columns = ["__".join(col) for col in feats.columns]
    feats = feats.reset_index()
    # dodaj cechy kalendarzowe
    feats["doy"] = feats["date"].dt.dayofyear
    feats["doy_sin"] = np.sin(2*np.pi*feats["doy"]/366.0)
    feats["doy_cos"] = np.cos(2*np.pi*feats["doy"]/366.0)

    data = feats.merge(day_rain, on=["voivodeship","date"], how="left").dropna(subset=["rain_tomorrow"])
    data["rain_tomorrow"] = data["rain_tomorrow"].astype(int)
    data["year"] = data["date"].dt.year
    return data

def feature_cols(df: pd.DataFrame):
    skip = {"voivodeship","date","year","rain_tomorrow"}
    return [c for c in df.columns if c not in skip]

# ====== GŁÓWNY PRZEPŁYW ======================================================

def main():
    os.makedirs(WORK_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1) Pobranie i wczytanie CSV
    csv_paths = []
    
    # Najpierw sprawdź czy mamy już pobrane pliki lokalnie
    print(f"[INFO] Sprawdzam lokalne pliki w {WORK_DIR}...")
    for y in YEARS:
        year_dir = os.path.join(WORK_DIR, str(y))
        if os.path.exists(year_dir):
            local_csvs = [os.path.join(year_dir, f) for f in os.listdir(year_dir) if f.endswith('.csv')]
            if local_csvs:
                print(f"[INFO] Znaleziono {len(local_csvs)} lokalnych plików dla roku {y}")
                csv_paths.extend(local_csvs)
                continue  # Pomiń pobieranie dla tego roku
        
        # Jeśli nie ma lokalnych plików, pobierz
        try:
            print(f"[INFO] Pobieranie danych dla roku {y}...")
            urls = list_zip_urls(y)
            print(f"[INFO] Znaleziono {len(urls)} plików dla roku {y}")
            for u in urls:
                try:
                    downloaded = download_csvs(u, os.path.join(WORK_DIR, str(y)))
                    csv_paths += downloaded
                    print(f"[INFO] Pobrano {len(downloaded)} plików z {u}")
                except Exception as e2:
                    print(f"[WARN] Nie można pobrać {u}: {e2}")
        except Exception as e:
            print(f"[WARN] rok {y}: {e}")
    
    if not csv_paths:
        print("[ERROR] Nie pobrano żadnych CSV z IMGW.")
        print(f"[INFO] Sprawdź połączenie internetowe i dostępność: {BASE_URL}")
        print(f"[INFO] Katalog roboczy: {WORK_DIR}")
        raise SystemExit("Brak danych do przetworzenia. Program nie może kontynuować bez danych z IMGW.")

    print(f"[INFO] Łącznie pobrano {len(csv_paths)} plików CSV")

    frames = []
    for p in csv_paths:
        df = read_csv_any(p)
        if df is not None and len(df): 
            frames.append(df)
            print(f"[INFO] Wczytano {len(df)} rekordów z {os.path.basename(p)}")
    
    if not frames:
        print("[ERROR] Nie udało się wczytać żadnych danych z pobranych CSV.")
        raise SystemExit("Wszystkie pliki CSV są puste lub uszkodzone.")
    
    print(f"[INFO] Łączenie {len(frames)} ramek danych...")
    data = pd.concat(frames, ignore_index=True)
    print(f"[INFO] Łączna liczba rekordów: {len(data)}")
    
    # Uzupełnij współrzędne geograficzne
    data = enrich_with_coords(data)
    
    # Sprawdź ile stacji ma współrzędne
    has_coords = data[["station_id", "lat", "lon"]].dropna().drop_duplicates()
    print(f"[INFO] Stacji z współrzędnymi: {len(has_coords)} / {data['station_id'].nunique()}")
    
    if len(has_coords) == 0:
        print("[ERROR] Żadna stacja nie ma współrzędnych geograficznych.")
        print("[INFO] Aby program działał, potrzebne są współrzędne stacji do mapowania na województwa.")
        print("[INFO] Dodaj więcej stacji do STATION_COORDS w kodzie lub pobierz metadane z IMGW.")
        raise SystemExit("Brak współrzędnych geograficznych stacji.")
    
    # Odfiltruj tylko rekordy bez koordynatów
    data_before = len(data)
    data = data.dropna(subset=["lat","lon"]).copy()
    print(f"[INFO] Pozostało {len(data)} rekordów po filtrowaniu stacji bez współrzędnych (było {data_before})")
    
    if len(data) == 0:
        print("[ERROR] Po odfiltrowaniu stacji bez współrzędnych nie pozostały żadne dane.")
        raise SystemExit("Brak danych do przetworzenia po filtrowaniu.")

    # 2) Mapowanie stacji -> województwo
    stations = data.groupby("station_id")[["station_id","station_name","lat","lon"]].agg("first").reset_index()
    st2woj = map_to_voivodeships(stations, GEOJSON_WOJ)  # <<< wymaga poprawnej ścieżki
    data = data.merge(st2woj, on="station_id", how="left").dropna(subset=["voivodeship"])

    # 3) Cechy + agregacja do województw
    feat = make_features(data)
    reg = aggregate_to_region(feat)

    # 4) Split czasowy
    train = reg[reg["year"] <= TRAIN_UNTIL_YEAR].copy()
    test  = reg[reg["year"] >  TRAIN_UNTIL_YEAR].copy()
    assert not train.empty and not test.empty, "Zbyt mało danych w train/test — zmień TRAIN_UNTIL_YEAR lub YEARS."

    Xtr, ytr = train[feature_cols(train)].fillna(0.0), train["rain_tomorrow"].values
    Xte, yte = test [feature_cols(test )].fillna(0.0), test ["rain_tomorrow"].values

    # 5) Walidacja modeli na train (TimeSeriesSplit)
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08)
    tss = TimeSeriesSplit(n_splits=5)
    cv_scores = {"roc_auc":[], "pr_auc":[], "f1":[], "precision":[], "recall":[]}
    for tr_idx, val_idx in tss.split(Xtr):
        X_tr, X_val = Xtr.iloc[tr_idx], Xtr.iloc[val_idx]
        y_tr, y_val = ytr[tr_idx], ytr[val_idx]
        clf.fit(X_tr, y_tr)
        p = clf.predict_proba(X_val)[:,1]
        pred = (p>=0.5).astype(int)
        cv_scores["roc_auc"].append(roc_auc_score(y_val, p))
        cv_scores["pr_auc"].append(average_precision_score(y_val, p))
        cv_scores["f1"].append(f1_score(y_val, pred))
        cv_scores["precision"].append(precision_score(y_val, pred))
        cv_scores["recall"].append(recall_score(y_val, pred))

    cv_summary = {k: float(np.mean(v)) for k,v in cv_scores.items()}

    # 6) Trening na całym train + test na przyszłości
    clf.fit(Xtr, ytr)
    p_test = clf.predict_proba(Xte)[:,1]
    y_pred = (p_test>=0.5).astype(int)

    metrics = {
        "cv_mean": cv_summary,
        "test": {
            "roc_auc": float(roc_auc_score(yte, p_test)),
            "pr_auc": float(average_precision_score(yte, p_test)),
            "f1": float(f1_score(yte, y_pred)),
            "precision": float(precision_score(yte, y_pred)),
            "recall": float(recall_score(yte, y_pred)),
        }
    }

    # 7) Zapisy
    with open(os.path.join(OUT_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    with open(os.path.join(OUT_DIR, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(classification_report(yte, y_pred, digits=4))

    dump(clf, os.path.join(OUT_DIR, "model.joblib"))

    # Prognoza dla najnowszej daty (p(t+1)) dla wszystkich województw
    latest_day = reg["date"].max()
    sample = reg[reg["date"] == latest_day].copy()
    proba = clf.predict_proba(sample[feature_cols(sample)].fillna(0.0))[:,1]
    out = sample[["voivodeship","date"]].copy()
    out["rain_tomorrow_prob"] = proba
    out.sort_values("rain_tomorrow_prob", ascending=False).to_csv(
        os.path.join(OUT_DIR, "latest_region_forecast.csv"), index=False, encoding="utf-8"
    )

    print("OK. Wyniki zapisane w", OUT_DIR)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
