import datetime
import logging
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import yfinance as yf

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class DataFetcher:
    @staticmethod
    def get_10y_treasury_yield() -> float:
        """Pobiera aktualną rentowność 10-letnich obligacji USA (^TNX)."""
        try:
            tnx = yf.Ticker("^TNX")
            hist = tnx.history(period="5d")
            if not hist.empty:
                return float(hist["Close"].iloc[-1]) / 100.0
        except Exception as e:
            logging.warning(f"Błąd pobierania ^TNX: {e}. Przyjmuję domyślne 4.25%.")
        return 0.0425

    @staticmethod
    def _find_row(df: Optional[pd.DataFrame], keywords: List[str]) -> Optional[pd.Series]:
        """Wyszukuje wiersz w sprawozdaniu po słowach kluczowych."""
        if df is None or df.empty:
            return None
        for idx in df.index:
            idx_clean = str(idx).lower().replace("_", " ").strip()
            for kw in keywords:
                if kw in idx_clean:
                    res = df.loc[idx]
                    return res.iloc[0] if isinstance(res, pd.DataFrame) else res
        return None

    @classmethod
    def get_upcoming_earnings_date(cls, ticker: str) -> Optional[Dict[str, Any]]:
        """Pobiera z serwerów giełdowych najbliższą datę publikacji wyników."""
        try:
            t = yf.Ticker(ticker)
            today = datetime.date.today()

            # 1. Próba przez get_earnings_dates
            try:
                ed = t.get_earnings_dates(limit=6)
                if ed is not None and not ed.empty:
                    now = pd.Timestamp.now(tz=ed.index.dtype.tz)
                    future = ed[ed.index >= now].sort_index()
                    if not future.empty:
                        target_dt = future.index[0].to_pydatetime().date()
                        days_left = (target_dt - today).days
                        return {"date": target_dt.strftime("%Y-%m-%d"), "days_left": days_left}
            except Exception:
                pass

            # 2. Fallback przez calendar
            cal = t.calendar
            if cal is not None:
                ed_val = None
                if isinstance(cal, dict):
                    ed_val = cal.get("Earnings Date")
                elif isinstance(cal, pd.DataFrame):
                    if "Earnings Date" in cal.index:
                        ed_val = cal.loc["Earnings Date"].values
                    elif "Earnings Date" in cal.columns:
                        ed_val = cal["Earnings Date"].values

                if ed_val is not None and len(ed_val) > 0:
                    val = ed_val[0]
                    target_dt = val.date() if hasattr(val, "date") else pd.to_datetime(val).date()
                    days_left = (target_dt - today).days
                    if days_left >= 0:
                        return {"date": target_dt.strftime("%Y-%m-%d"), "days_left": days_left}
        except Exception as e:
            logging.debug(f"Błąd pobierania daty earnings dla {ticker}: {e}")
        return None

    @classmethod
    def get_mag8_previous_earnings_and_capex(cls, ticker: str) -> Dict[str, Any]:
        """
        Pobiera zrealizowane dane z OSTATNIEGO kwartału:
        - Faktyczny CapEx w mld USD i dynamikę kw/kw oraz r/r
        - Zaskoczenie zyskiem EPS (Beat/Miss)
        - Klasyfikację trendu inwestycyjnego
        """
        res = {
            "ticker": ticker,
            "last_capex_billions": 0.0,
            "capex_qoq_pct": 0.0,
            "capex_yoy_pct": 0.0,
            "capex_status": "⚪ Brak danych",
            "last_eps_surprise_pct": None,
            "eps_verdict": "Brak danych",
            "last_reported_date": "Niedawno",
        }
        try:
            t = yf.Ticker(ticker)
            q_cf = t.quarterly_cashflow

            # 1. Analiza CapEx z przepływów pieniężnych
            capex_series = cls._find_row(q_cf, ["capital expenditure", "capital expenditures"])
            if capex_series is not None and len(capex_series) >= 2:
                c0 = abs(float(capex_series.iloc[0]))
                c1 = abs(float(capex_series.iloc[1]))
                res["last_capex_billions"] = round(c0 / 1e9, 2)
                
                if c1 > 0:
                    qoq = ((c0 - c1) / c1) * 100.0
                    res["capex_qoq_pct"] = round(qoq, 1)

                if len(capex_series) >= 5:
                    c4 = abs(float(capex_series.iloc[4]))
                    if c4 > 0:
                        res["capex_yoy_pct"] = round(((c0 - c4) / c4) * 100.0, 1)

                # Klasyfikacja trendu CapEx
                if res["capex_qoq_pct"] > 8.0:
                    res["capex_status"] = "🟢 Agresywna rozbudowa AI"
                elif res["capex_qoq_pct"] < -8.0:
                    res["capex_status"] = "🚨 CIĘCIE WYDATKÓW (Chłodzenie)"
                else:
                    res["capex_status"] = "🟡 Ustabilizowane wydatki"

            # 2. Wyniki finansowe i Zaskoczenie EPS
            try:
                ed = t.get_earnings_dates(limit=8)
                if ed is not None and not ed.empty:
                    now = pd.Timestamp.now(tz=ed.index.dtype.tz)
                    past = ed[ed.index < now].sort_index(ascending=False)
                    if not past.empty:
                        last_q = past.iloc[0]
                        res["last_reported_date"] = past.index[0].strftime("%Y-%m-%d")
                        
                        # Sprawdzamy kolumny Surprise(%) lub Reported/Estimate
                        for col in last_q.index:
                            if "surprise" in str(col).lower():
                                val = last_q[col]
                                if pd.notna(val):
                                    surp = float(val) * 100.0 if abs(val) < 1.0 else float(val)
                                    res["last_eps_surprise_pct"] = round(surp, 1)
                                    res["eps_verdict"] = f"🟢 Pobicie (+{surp:.1f}%)" if surp >= 0 else f"🔴 Wpadka ({surp:.1f}%)"
                                break
            except Exception:
                pass

        except Exception as e:
            logging.debug(f"Błąd analizy poprzednich wyników dla {ticker}: {e}")

        return res

    @classmethod
    def fetch_stock_data(cls, ticker: str) -> Optional[Dict[str, Any]]:
        """Pobiera komplet danych TTM z obsługą walut ADR."""
        try:
            t = yf.Ticker(ticker)
            info = t.info
            hist = t.history(period="1y")

            if hist.empty or len(hist) < 200:
                logging.warning(f"Zbyt krótka historia notowań dla {ticker}.")
                return None

            q_cf = t.quarterly_cashflow
            q_is = t.quarterly_financials
            q_bs = t.quarterly_balance_sheet

            # Korekta walutowa dla spółek zagranicznych (np. TSM w TWD, ASML w EUR)
            fin_currency = info.get("financialCurrency", "USD")
            fx_rate = 1.0
            if ticker == "TSM" and fin_currency == "TWD":
                fx_rate = 0.031  # 1 TWD ≈ 0.031 USD
            elif ticker == "ASML" and fin_currency == "EUR":
                fx_rate = 1.08   # 1 EUR ≈ 1.08 USD

            # 1. Cash Flow TTM
            sbc_series = cls._find_row(q_cf, ["stock based compensation", "share based compensation"])
            ocf_series = cls._find_row(q_cf, ["operating cash flow", "cash flow from continuing operating"])
            capex_series = cls._find_row(q_cf, ["capital expenditure", "capital expenditures"])

            capex_qoq_change_pct = 0.0
            latest_capex = 0.0

            if capex_series is not None and len(capex_series) >= 2:
                c_now = abs(float(capex_series.iloc[0])) * fx_rate
                c_prev = abs(float(capex_series.iloc[1])) * fx_rate
                latest_capex = c_now
                if c_prev > 0:
                    capex_qoq_change_pct = ((c_now - c_prev) / c_prev) * 100.0

            if ocf_series is not None and len(ocf_series) >= 4:
                ocf_ttm = float(ocf_series.iloc[:4].fillna(0).sum()) * fx_rate
                capex_ttm = (abs(float(capex_series.iloc[:4].fillna(0).sum())) if capex_series is not None else 0.0) * fx_rate
                sbc_ttm = (float(sbc_series.iloc[:4].fillna(0).sum()) if sbc_series is not None else 0.0) * fx_rate
            else:
                annual_cf = t.cashflow
                ocf_row = cls._find_row(annual_cf, ["operating cash flow", "cash flow from continuing operating"])
                capex_row = cls._find_row(annual_cf, ["capital expenditure", "capital expenditures"])
                sbc_row = cls._find_row(annual_cf, ["stock based compensation", "share based compensation"])

                ocf_ttm = (float(ocf_row.iloc[0]) if ocf_row is not None else float(info.get("operatingCashflow") or 0.0)) * fx_rate
                capex_ttm = (abs(float(capex_row.iloc[0])) if capex_row is not None else 0.0) * fx_rate
                sbc_ttm = (float(sbc_row.iloc[0]) if sbc_row is not None else 0.0) * fx_rate

            reported_fcf_ttm = ocf_ttm - capex_ttm
            real_fcf_ttm = reported_fcf_ttm - sbc_ttm

            # 2. Rachunek wyników TTM
            ebit_series = cls._find_row(q_is, ["operating income", "ebit"])
            pretax_series = cls._find_row(q_is, ["pretax income", "income before tax"])
            tax_series = cls._find_row(q_is, ["tax provision", "income tax expense"])

            if ebit_series is not None and len(ebit_series) >= 4:
                ebit_ttm = float(ebit_series.iloc[:4].fillna(0).sum()) * fx_rate
                pretax_ttm = (float(pretax_series.iloc[:4].fillna(0).sum()) if pretax_series is not None else 0.0) * fx_rate
                tax_ttm = (float(tax_series.iloc[:4].fillna(0).sum()) if tax_series is not None else 0.0) * fx_rate
            else:
                ebit_ttm = float(info.get("operatingIncome") or info.get("ebitda") or 0.0) * fx_rate
                pretax_ttm = ebit_ttm
                tax_ttm = ebit_ttm * 0.21

            effective_tax_rate = float(np.clip(tax_ttm / pretax_ttm, 0.15, 0.25)) if pretax_ttm > 0 and tax_ttm > 0 else 0.21

            # 3. Bilans
            total_debt = float(info.get("totalDebt") or 0.0) * fx_rate
            total_equity = float(info.get("totalStockholderEquity") or 0.0) * fx_rate
            cash = float(info.get("totalCash") or 0.0) * fx_rate
            total_assets = float(info.get("totalAssets") or 0.0) * fx_rate

            shares_change_pct = 0.0
            if q_bs is not None and not q_bs.empty:
                shares_row = cls._find_row(q_bs, ["ordinary shares number", "share issued", "common stock shares outstanding"])
                if shares_row is not None and len(shares_row) >= 4:
                    try:
                        s_now = float(shares_row.iloc[0])
                        s_prev_year = float(shares_row.iloc[3])
                        if s_prev_year > 0:
                            shares_change_pct = (s_now - s_prev_year) / s_prev_year
                    except Exception:
                        pass

            earnings_info = cls.get_upcoming_earnings_date(ticker)

            return {
                "ticker": ticker,
                "info": info,
                "hist": hist,
                "real_fcf_ttm": real_fcf_ttm,
                "reported_fcf_ttm": reported_fcf_ttm,
                "sbc_ttm": sbc_ttm,
                "ebit_ttm": ebit_ttm,
                "tax_rate": effective_tax_rate,
                "total_debt": total_debt,
                "total_equity": total_equity,
                "cash": cash,
                "total_assets": total_assets,
                "shares_change_pct": shares_change_pct,
                "capex_qoq_change_pct": capex_qoq_change_pct,
                "latest_capex_millions": round(latest_capex / 1e6, 1),
                "earnings_info": earnings_info,
            }
        except Exception as e:
            logging.error(f"Krytyczny błąd pobierania danych dla {ticker}: {e}")
            return None