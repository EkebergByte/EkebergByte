import logging
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd


class ValuationEngine:
    def __init__(self, config: Dict[str, Any], rf_rate: float):
        self.cfg = config
        self.weights = config["weights"]
        self.rf_rate = rf_rate

    @staticmethod
    def calculate_wilder_rsi(prices: pd.Series, period: int = 14) -> float:
        """
        Oryginalny wskaźnik RSI J. Wellesa Wildera oparty na wykładniczym
        wygładzaniu (Wilder's RMA / ewm), identyczny z TradingView i Bloomberg.
        """
        if len(prices) < period + 2:
            return 50.0

        delta = prices.diff()
        gain = delta.clip(lower=0.0)
        loss = -delta.clip(upper=0.0)

        avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

        last_gain = float(avg_gain.iloc[-1])
        last_loss = float(avg_loss.iloc[-1])

        if last_loss == 0.0:
            return 100.0 if last_gain > 0 else 50.0

        rs = last_gain / last_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return float(np.clip(rsi, 0.0, 100.0))

    @staticmethod
    def calculate_real_roic(
        ebit: float,
        tax_rate: float,
        total_debt: float,
        total_equity: float,
        cash: float,
        total_assets: float
    ) -> float:
        """
        Kalkulacja rzeczywistego ROIC = NOPAT / Invested Capital.
        Dostosowana do różnych struktur bilansowych:
         - Standard: Dług + Kapitał Własny - Gotówka
         - Spółki z ujemnym kapitałem własnym przez buybacki (np. Apple):
           zabezpieczenie oparte na aktywach operacyjnych netto.
        """
        nopat = ebit * (1.0 - tax_rate)
        if ebit <= 0:
            return 0.0

        invested_capital = total_debt + total_equity - cash

        # Obsługa ujemnego kapitału przez agresywny skup akcji
        if invested_capital <= 0:
            invested_capital = max(total_assets - cash, 0.1 * total_assets)

        if invested_capital <= 0:
            return 0.35  # Domyślna wysoka rentowność dla monopolu działającego na ujemnym kapitale

        roic = nopat / invested_capital
        return float(np.clip(roic, -0.20, 1.0))

    def solve_reverse_dcf(self, fcf: float, market_cap: float) -> Optional[float]:
        """
        Odwrócone DCF:
        Zwraca wymaganą roczną stopę wzrostu FCF w % lub None, jeśli FCF <= 0.
        """
        if fcf <= 0 or market_cap <= 0:
            return None  # Ujemny FCF = brak możliwości wyliczenia implied growth

        r = max(self.cfg["thresholds"]["discount_rate_dcf"], self.rf_rate + 0.045)
        tg = self.cfg["thresholds"]["terminal_growth_dcf"]

        low, high = -0.30, 0.70
        implied_g = 0.0

        for _ in range(25):
            g = (low + high) / 2
            pv = 0.0
            cf = fcf
            for t in range(1, 11):
                cf *= (1.0 + g)
                pv += cf / ((1.0 + r) ** t)
            tv = (cf * (1.0 + tg)) / (r - tg)
            pv_tv = tv / ((1.0 + r) ** 10)
            model_val = pv + pv_tv

            if model_val < market_cap:
                low = g
            else:
                high = g
            implied_g = g

        return round(implied_g * 100.0, 2)

    def evaluate_stock(self, data: Dict[str, Any]) -> Dict[str, Any]:
        info = data["info"]
        hist = data["hist"]
        real_fcf = data["real_fcf_ttm"]
        shares_change = data["shares_change_pct"]
        earnings_info = data.get("earnings_info")
        capex_qoq = data.get("capex_qoq_change_pct", 0.0)

        price = float(hist["Close"].iloc[-1])
        market_cap = float(info.get("marketCap") or (price * info.get("sharesOutstanding", 1)))
        sma200 = float(hist["Close"].rolling(200).mean().iloc[-1])
        sma200_dev = ((price - sma200) / sma200) * 100.0

        # 1. Prawdziwe RSI Wildera
        rsi = self.calculate_wilder_rsi(hist["Close"], period=14)

        # 2. Prawdziwy ROIC
        roic = self.calculate_real_roic(
            ebit=data["ebit_ttm"],
            tax_rate=data["tax_rate"],
            total_debt=data["total_debt"],
            total_equity=data["total_equity"],
            cash=data["cash"],
            total_assets=data["total_assets"],
        )

        # 3. Real FCF Yield TTM
        real_fcf_yield = (real_fcf / market_cap) if market_cap > 0 else 0.0

        # 4. Wycena i PEG
        forward_pe = float(info.get("forwardPE") or info.get("trailingPE") or 25.0)
        peg_ratio = float(info.get("pegRatio") or 1.5)
        if peg_ratio <= 0:
            peg_ratio = 2.5  # Kara punktowa za ujemny/zaburzony wzrost

        # 5. Equity Risk Premium Spread
        earnings_yield = (1.0 / forward_pe) if forward_pe > 0 else 0.0
        erp_spread = (earnings_yield - self.rf_rate) * 100.0

        # 6. Reverse DCF
        implied_growth = self.solve_reverse_dcf(real_fcf, market_cap)

        # Consensus analityków
        target_price = float(info.get("targetMeanPrice") or price)
        analyst_upside = ((target_price - price) / price) * 100.0

        # === SYSTEM PUNKTACJI ALPHASCORE (0 - 100) ===
        # Filar A: Real FCF Yield
        if real_fcf <= 0:
            s_fcf = 0.0
        else:
            s_fcf = np.clip((real_fcf_yield - 0.02) / 0.04 * 100.0, 0.0, 100.0)

        # Filar B: ROIC (Fosa kapitałowa)
        s_roic = np.clip((roic - 0.10) / 0.15 * 100.0, 0.0, 100.0)

        # Filar C: PEG
        s_peg = np.clip((2.2 - peg_ratio) / 1.2 * 100.0, 0.0, 100.0)

        # Filar C: ERP Spread
        s_erp = np.clip((erp_spread - (-0.5)) / 2.5 * 100.0, 0.0, 100.0)

        # Filar C: Reverse DCF
        if implied_growth is None:
            s_dcf = 0.0
        else:
            s_dcf = np.clip((18.0 - implied_growth) / 12.0 * 100.0, 0.0, 100.0)

        # Filar D: Timing Techniczny
        s_rsi = np.clip((50.0 - rsi) / 20.0 * 100.0, 0.0, 100.0)
        s_sma = np.clip((-sma200_dev) / 15.0 * 100.0, 0.0, 100.0)
        s_tech = (s_rsi * 0.6) + (s_sma * 0.4)

        w = self.weights
        alpha_score = (
            s_fcf * w["real_fcf_yield"]
            + s_roic * w["roic_quality"]
            + s_peg * w["peg_valuation"]
            + s_erp * w["erp_spread"]
            + s_dcf * w["reverse_dcf"]
            + s_tech * w["technical_timing"]
        )

        # === DETEKCJA SYSTEMOWYCH SYGNAŁÓW OSTRZEGAWCZYCH ===
        danger_signals = []
        if earnings_info and 0 <= earnings_info["days_left"] <= 7:
            danger_signals.append(f"⚠️ Wyniki za {earnings_info['days_left']} dni (Zakaz otwierania pozycji!)")
        if capex_qoq < -8.0:
            danger_signals.append(f"🚨 Cięcie wydatków CapEx o {capex_qoq:.1f}% kw/kw (Schłodzenie inwestycji AI)")
        if real_fcf <= 0:
            danger_signals.append("⚠️ Ujemny Real FCF (Spółka przepala gotówkę)")

        return {
            "ticker": data["ticker"],
            "price": round(price, 2),
            "alpha_score": round(alpha_score, 1),
            "real_fcf_yield_pct": round(real_fcf_yield * 100.0, 2),
            "sbc_millions": round(data["sbc_ttm"] / 1e6, 1),
            "peg_ratio": round(peg_ratio, 2),
            "forward_pe": round(forward_pe, 1),
            "erp_spread_pct": round(erp_spread, 2),
            "implied_growth_pct": implied_growth,
            "roic_pct": round(roic * 100.0, 1),
            "rsi_14": round(rsi, 1),
            "sma200_dev_pct": round(sma200_dev, 1),
            "analyst_upside_pct": round(analyst_upside, 1),
            "net_buyback": True if shares_change < -0.01 else False,
            "earnings_info": earnings_info,
            "capex_qoq_pct": round(capex_qoq, 1),
            "danger_signals": danger_signals,
        }