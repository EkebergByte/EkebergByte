import json
import os
import sys
import joblib
import pandas as pd
import requests
from dotenv import load_dotenv
from fredapi import Fred

# Ładuje zmienne z lokalnego pliku .env
load_dotenv()

API_KEY = os.getenv("FRED_API_KEY")
if not API_KEY:
    sys.exit(
        "BŁĄD: Brak klucza FRED_API_KEY! Upewnij się, że masz plik .env z wpisem FRED_API_KEY=twój_klucz."
    )

TG_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TG_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

fred = Fred(api_key=API_KEY)
STATE_FILE = "fred_state.json"
BUNDLE_FILE = "model_bundle.joblib"

SERIES = {
    "gs10": "GS10",
    "tb3m": "TB3MS",
    "baa10y": "BAA10Y",
    "unrate": "UNRATE",
    "claims_4w": "IC4WSA",
    "permits": "PERMIT",
}


def send_alert(message):
    if TG_TOKEN and TG_CHAT_ID:
        url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
        try:
            requests.post(
                url,
                json={
                    "chat_id": TG_CHAT_ID,
                    "text": message,
                    "parse_mode": "Markdown",
                },
                timeout=10,
            )
        except Exception as e:
            print(f"Błąd wysyłki Telegram: {e}")
    print(f"\n[RAPORT SYSTEMOWY]:\n{message}")


def check_and_run():
    if not os.path.exists(BUNDLE_FILE):
        sys.exit(
            f"Brak pliku '{BUNDLE_FILE}'! Uruchom najpierw: python train_models.py"
        )

    # 1. Sprawdzanie metadanych w FRED (Lekkie zapytania JSON)
    last_state = (
        json.load(open(STATE_FILE, encoding="utf-8"))
        if os.path.exists(STATE_FILE)
        else {}
    )
    current_state = {}
    changed_series = []

    for name, s_id in SERIES.items():
        info = fred.get_series_info(s_id)
        last_up = info.get("last_updated")
        current_state[s_id] = last_up
        if last_state.get(s_id) != last_up:
            changed_series.append((name, s_id, info.get("title")))

    if not changed_series and "last_probabilities" in last_state:
        print("☕ [FRED] Brak nowych danych od ostatniego sprawdzenia.")
        return

    updated_names = (
        ", ".join([item[0] for item in changed_series])
        if changed_series
        else "Inicjalizacja systemu"
    )
    print(f"⚡ Wykryto nowe dane: {updated_names}")

    # 2. Pobieranie surowych serii
    raw = {k: fred.get_series(v) for k, v in SERIES.items()}

    # 3. Resamplujemy każdą serię osobno na jej WŁASNEJ osi czasu
    s_gs10 = raw["gs10"].resample("MS").last().dropna()
    s_tb3m = raw["tb3m"].resample("MS").last().dropna()
    s_baa = raw["baa10y"].resample("MS").last().dropna()
    s_unrate = raw["unrate"].resample("MS").last().dropna()
    s_claims = raw["claims_4w"].resample("MS").last().dropna()
    s_permits = raw["permits"].resample("MS").last().dropna()

    # 4. Wyliczamy cechy na czystych danych (BEZ zniekształceń ffill)
    yc_10y3m = s_gs10 - s_tb3m
    yc_delta6m = yc_10y3m - yc_10y3m.shift(6)
    yc_lag6m = yc_10y3m.shift(6)
    baa_lag6m = s_baa.shift(6)
    permits_yoy = s_permits.pct_change(12) * 100
    claims_yoy = s_claims.pct_change(12) * 100
    sahm_rule = s_unrate.rolling(3).mean() - s_unrate.rolling(12).min()

    # 5. Łączymy w ramkę i DOPIERO TERAZ stosujemy ffill dla poszarpanej krawędzi (Ragged-Edge)
    features_df = pd.DataFrame(
        {
            "yield_curve_10y3m": yc_10y3m,
            "yield_curve_delta6m": yc_delta6m,
            "yield_curve_lag6m": yc_lag6m,
            "baa10y": s_baa,
            "baa_spread_lag6m": baa_lag6m,
            "permits_yoy": permits_yoy,
            "claims_yoy": claims_yoy,
            "sahm_rule": sahm_rule,
        }
    ).ffill()

    latest_row = features_df.iloc[[-1]]
    latest_date = features_df.index[-1].strftime("%Y-%m-%d")

    # 6. Predykcja
    bundle = joblib.load(BUNDLE_FILE)
    new_probs = {}
    for h, feats in bundle["configs"].items():
        vec = latest_row[feats]
        vec_scaled = bundle["scalers"][h].transform(vec)
        prob = bundle["models"][h].predict_proba(vec_scaled)[0][1] * 100
        new_probs[str(h)] = round(prob, 2)

    old_probs = last_state.get("last_probabilities", {})

    # 7. Konstrukcja raportu (Delta Alert)
    msg = f"📊 *RAPORT MAKROEKONOMICZNY USA* | Stan: `{latest_date}`\n"
    msg += f"Zaktualizowano: `{updated_names}`\n\n"
    msg += f"🏛️ *Szacowane Ryzyko Recesji (Nowcast):*\n"

    for h in [3, 6, 12]:
        p_new = new_probs[str(h)]
        p_old = old_probs.get(str(h), p_new)
        delta = p_new - p_old
        sign = f"+{delta:.2f}%" if delta > 0 else f"{delta:.2f}%"
        delta_str = f" (Zmiana: {sign})" if p_old != p_new else ""

        if p_new < 15.0:
            ico = "🟢"
        elif p_new < 35.0:
            ico = "🟡"
        elif p_new < 55.0:
            ico = "🟠"
        else:
            ico = "🔴"

        msg += f"{ico} **Horyzont {h}M:** `{p_new}%`{delta_str}\n"

    yc = latest_row["yield_curve_10y3m"].values[0]
    sahm = latest_row["sahm_rule"].values[0]
    claims = latest_row["claims_yoy"].values[0]
    baa = latest_row["baa10y"].values[0]

    msg += f"\n🔍 *Kluczowe Odczyty:* Sahm: `{sahm:.2f} pkt`, Krzywa 10Y-3M: `{yc:.2f}%`, Zasiłki YoY: `{claims:.1f}%`, Baa Spread: `{baa:.2f}%`"

    send_alert(msg)

    # 8. Zapis stanu
    current_state["last_probabilities"] = new_probs
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(current_state, f, indent=2)


if __name__ == "__main__":
    check_and_run()