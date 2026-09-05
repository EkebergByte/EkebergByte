import os
import yaml
from dotenv import load_dotenv
from src.data_fetcher import DataFetcher
from src.valuation_engine import ValuationEngine
from src.notifier import Notifier

load_dotenv()


def main():
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    print("=" * 80)
    print("🚀 START SKANERA ALPHAMOAT-RADAR + EARNINGS & CAPEX SENTINEL")
    print("=" * 80)

    rf_rate = DataFetcher.get_10y_treasury_yield()
    print(f"Bieżąca stopa wolna od ryzyka (US 10Y Yield): {rf_rate * 100:.2f}%\n")

    engine = ValuationEngine(config, rf_rate)
    notifier = Notifier()

    results = []
    watchlist = config.get("watchlist", [])

    print("🔍 SKANOWANIE FUNDAMENTALNE WATCHLISTY:")
    for ticker in watchlist:
        print(f"  Analizuję: {ticker:<6} ...", end="", flush=True)
        raw_data = DataFetcher.fetch_stock_data(ticker)
        if not raw_data:
            print(" [POMINIĘTO: Brak danych]")
            continue

        metrics = engine.evaluate_stock(raw_data)
        results.append(metrics)
        flags = f" [FLAGI: {len(metrics['danger_signals'])}]" if metrics["danger_signals"] else ""
        print(f" AlphaScore: {metrics['alpha_score']:4.1f} | RSI: {metrics['rsi_14']:4.1f} | FCF Yield: {metrics['real_fcf_yield_pct']:4.1f}%{flags}")

    results.sort(key=lambda x: x["alpha_score"], reverse=True)
    min_score = config["thresholds"]["min_alpha_score"]
    top_opportunities = [r for r in results if r["alpha_score"] >= min_score]

    # === 1. KALENDARZ NADCHODZĄCYCH WYNIKÓW DLA CAŁEJ WATCHLISTY (SORTOWANIE CHRONOLOGICZNE) ===
    print("\n" + "=" * 80)
    print("📅 KALENDARZ NADCHODZĄCYCH RAPORTÓW (CAŁA WATCHLISTA - WG KOLEJNOŚCI):")
    print("=" * 80)
    print(f"{'TICKER':<7} | {'DATA RAPORTU':<14} | {'POZOSTAŁO DNI':<15} | {'STATUS RYZYKA'}")
    print("-" * 80)

    all_calendar = []
    for r in results:
        e_info = r.get("earnings_info")
        if e_info and e_info.get("days_left") is not None:
            days = e_info["days_left"]
            d_str = e_info["date"]
        else:
            days = None
            d_str = "Brak daty (TBD)"

        all_calendar.append({
            "ticker": r["ticker"],
            "date": d_str,
            "days_left": days,
        })

    # Sortowanie: najpierw te najbliższe (np. za 5 dni), na końcu te bez daty (TBD)
    all_calendar.sort(key=lambda x: x["days_left"] if x["days_left"] is not None else 9999)

    for item in all_calendar:
        days = item["days_left"]
        if days is not None:
            if days <= 7:
                status = "🔴 BLISKO (Ruletka - Nie kupować!)"
            elif days <= 21:
                status = "🟡 Za 2-3 tygodnie"
            else:
                status = "🟢 Bezpieczny dystans"
        else:
            status = "⚪ Oczekuje na ogłoszenie"

        days_display = str(days) if days is not None else "-"
        print(f"{item['ticker']:<7} | {item['date']:<14} | {days_display:<15} | {status}")

    # === 2. AUDYT CAPEX HYPERSCALERÓW AI ===
    # Sprawdzamy wydatki kluczowych gigantów AI obecnych na liście
    hyperscalers = [t for t in ["NVDA", "MSFT", "AAPL", "GOOGL", "AMZN", "META", "AVGO", "TSM"] if t in watchlist]

    print("\n" + "=" * 80)
    print("🏗️  AUDYT OSTATNICH WYNIKÓW I CAPEX GIGANTÓW AI (CZY CYKL ZWALNIA?):")
    print("=" * 80)
    print(f"{'TICKER':<7} | {'CAPEX (Q0)':<11} | {'ZMIANA KW/KW':<13} | {'ZYSK EPS (BEAT/MISS)':<22} | {'STATUS CYKLU'}")
    print("-" * 80)

    capex_history = []
    for ticker in hyperscalers:
        h = DataFetcher.get_mag8_previous_earnings_and_capex(ticker)
        capex_history.append(h)
        sign = "+" if h["capex_qoq_pct"] > 0 else ""
        qoq_str = f"{sign}{h['capex_qoq_pct']}%"
        capex_str = f"${h['last_capex_billions']}B"
        print(f"{ticker:<7} | {capex_str:<11} | {qoq_str:<13} | {h['eps_verdict']:<22} | {h['capex_status']}")

    print("=" * 80)

    # Wysłanie raportu
    report_html = notifier.format_report(top_opportunities, results, all_calendar, capex_history, rf_rate)

    has_threats = any(len(r.get("danger_signals", [])) > 0 for r in results)
    if top_opportunities or has_threats:
        notifier.send_telegram(report_html)


if __name__ == "__main__":
    main()