import os
import sys
import joblib
import pandas as pd
from dotenv import load_dotenv
from fredapi import Fred
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Ładowanie zmiennych z ukrytego pliku .env
load_dotenv()

API_KEY = os.getenv("FRED_API_KEY")
if not API_KEY:
    sys.exit(
        "BŁĄD: Brak klucza FRED_API_KEY w pliku .env! Upewnij się, że plik istnieje."
    )

fred = Fred(api_key=API_KEY)


def train_and_persist():
    series = {
        "gs10": "GS10",  # 10-Year Treasury Yield
        "tb3m": "TB3MS",  # 3-Month Treasury Bill Yield
        "baa10y": "BAA10Y",  # Moody's Baa Corporate Spread
        "unrate": "UNRATE",  # Stopa bezrobocia
        "claims_4w": "IC4WSA",  # Wnioski o zasiłek (średnia 4-tyg)
        "permits": "PERMIT",  # Pozwolenia na budowę domów
        "usrec": "USREC",  # Oficjalne recesje NBER (0 lub 1)
    }

    print("\n" + "=" * 88)
    print(
        "📡 POBIERANIE SUROWYCH DANYCH Z FEDERAL RESERVE BANK OF ST. LOUIS (FRED)..."
    )
    print("=" * 88)
    print(
        f"{'WSKAŹNIK':<12} | {'KOD FRED':<10} | {'NAJSTARSZA DATA':<15} | {'NAJNOWSZA DATA':<15} | {'LICZBA PUNKTÓW'}"
    )
    print("-" * 88)

    raw = {}
    for name, code in series.items():
        # Pobranie i odrzucenie braków
        s = fred.get_series(code).dropna()
        raw[name] = s

        # Prawdziwe daty bezpośrednio z indeksu pobranej serii
        start_date = s.index[0].strftime("%Y-%m-%d")
        end_date = s.index[-1].strftime("%Y-%m-%d")
        count = len(s)

        print(
            f"{name:<12} | {code:<10} | {start_date:<15} | {end_date:<15} | {count:>6} odczytów"
        )

    print("=" * 88)

    # 1. Resampling każdej czystej serii do siatki miesięcznej (Month Start)
    s_gs10 = raw["gs10"].resample("MS").last().dropna()
    s_tb3m = raw["tb3m"].resample("MS").last().dropna()
    s_baa = raw["baa10y"].resample("MS").last().dropna()
    s_unrate = raw["unrate"].resample("MS").last().dropna()
    s_claims = raw["claims_4w"].resample("MS").last().dropna()
    s_permits = raw["permits"].resample("MS").last().dropna()
    s_usrec = raw["usrec"].resample("MS").last().dropna()

    # 2. Wyliczanie cech na nieskażonych szeregach
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

    # 3. Do treningu bierzemy wyłącznie pełne, zsynchronizowane miesiące (Balanced Panel)
    df = df.dropna()

    # Dynamiczny audyt wspólnej macierzy treningowej
    recessions_count = (
        (df["usrec"].diff() == 1).sum() + (1 if df["usrec"].iloc[0] == 1 else 0)
    )
    first_month = df.index[0].strftime("%Y-%m-%d")
    last_month = df.index[-1].strftime("%Y-%m-%d")
    total_months = len(df)
    total_years = total_months / 12

    print(
        "\n🔍 WERYFIKACJA WSPÓLNEJ MACIERZY TRENINGOWEJ PO POŁĄCZENIU I SYNCHRONIZACJI:"
    )
    print(f"  • Pierwszy wspólny miesiąc : {first_month}")
    print(f"  • Ostatni wspólny miesiąc  : {last_month}")
    print(
        f"  • Łączna długość próby     : {total_months} miesięcy ({total_years:.1f} lat historii)"
    )
    print(f"  • Zarejestrowane recesje   : {recessions_count} kryzysów w próbie")
    print("-" * 88)

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

    print("🧠 TRENOWANIE MODELI REGRESJI LOGISTYCZNEJ (BEZSTRONNY LOGIT):")
    for h, feats in configs.items():
        # Zmienna celu: czy w ciągu kolejnych h miesięcy wystąpi recesja
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

        # Zgodnie z wytycznymi scikit-learn 1.8+:
        # Używamy l1_ratio=0.0 zamiast penalty="l2", unikając FutureWarning
        model = LogisticRegression(
            l1_ratio=0.0, C=1.0, random_state=42, solver="lbfgs"
        )
        model.fit(X_scaled, y)

        bundle["models"][h] = model
        bundle["scalers"][h] = scaler
        print(f"  ✓ Model {h:>2}M wytrenowany poprawnie na cechach: {feats}")

    # Zapis wag do pliku
    joblib.dump(bundle, "model_bundle.joblib")
    print(
        f"\n💾 Pomyślnie zaktualizowano plik 'model_bundle.joblib'. Zero ostrzeżeń."
    )
    print("=" * 88 + "\n")


if __name__ == "__main__":
    train_and_persist()