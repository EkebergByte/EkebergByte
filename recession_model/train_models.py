import os
import sys
import joblib
import pandas as pd
from dotenv import load_dotenv
from fredapi import Fred
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Ładuje zmienne z lokalnego pliku .env
load_dotenv()

API_KEY = os.getenv("FRED_API_KEY")
if not API_KEY:
    sys.exit(
        "BŁĄD: Brak klucza FRED_API_KEY! Upewnij się, że masz plik .env z wpisem FRED_API_KEY=twój_klucz."
    )

fred = Fred(api_key=API_KEY)


def train_and_persist():
    print("⏳ [TRENING] Pobieranie historii 1968–2026 i trenowanie wag...")

    series = {
        "gs10": "GS10",
        "tb3m": "TB3MS",
        "baa10y": "BAA10Y",
        "unrate": "UNRATE",
        "claims_4w": "IC4WSA",
        "permits": "PERMIT",
        "usrec": "USREC",
    }
    raw = {k: fred.get_series(v) for k, v in series.items()}

    # Resampling każdej serii z osobna
    s_gs10 = raw["gs10"].resample("MS").last().dropna()
    s_tb3m = raw["tb3m"].resample("MS").last().dropna()
    s_baa = raw["baa10y"].resample("MS").last().dropna()
    s_unrate = raw["unrate"].resample("MS").last().dropna()
    s_claims = raw["claims_4w"].resample("MS").last().dropna()
    s_permits = raw["permits"].resample("MS").last().dropna()
    s_usrec = raw["usrec"].resample("MS").last().dropna()

    # Wyliczamy cechy na czystych szeregach
    df = pd.DataFrame(index=s_gs10.index)
    df["yield_curve_10y3m"] = s_gs10 - s_tb3m
    df["yield_curve_delta6m"] = df["yield_curve_10y3m"] - df[
        "yield_curve_10y3m"
    ].shift(6)
    df["yield_curve_lag6m"] = df["yield_curve_10y3m"].shift(6)
    df["baa10y"] = s_baa
    df["baa_spread_lag6m"] = s_baa.shift(6)
    df["claims_yoy"] = s_claims.pct_change(12) * 100
    df["permits_yoy"] = s_permits.pct_change(12) * 100
    df["sahm_rule"] = s_unrate.rolling(3).mean() - s_unrate.rolling(12).min()
    df["usrec"] = s_usrec

    # Uczymy się tylko na zsynchronizowanych danych historycznych
    df = df.dropna()

    configs = {
        3: ["sahm_rule", "claims_yoy", "baa10y", "yield_curve_10y3m"],
        6: [
            "yield_curve_10y3m",
            "yield_curve_delta6m",
            "baa10y",
            "claims_yoy",
        ],
        12: [
            "yield_curve_lag6m",
            "permits_yoy",
            "baa_spread_lag6m",
            "yield_curve_10y3m",
        ],
    }

    bundle = {"models": {}, "scalers": {}, "configs": configs}

    for h, feats in configs.items():
        y = (
            df["usrec"]
            .rolling(window=h)
            .max()
            .shift(-h)
            .dropna()
            .astype(int)
        )
        X = df.loc[y.index, feats]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = LogisticRegression(penalty="l2", C=1.0, random_state=42)
        model.fit(X_scaled, y)

        bundle["models"][h] = model
        bundle["scalers"][h] = scaler

    joblib.dump(bundle, "model_bundle.joblib")
    print(
        f"✓ [SUKCES] Modele wytrenowane na {len(df)} miesiącach i zapisane do 'model_bundle.joblib'."
    )


if __name__ == "__main__":
    train_and_persist()