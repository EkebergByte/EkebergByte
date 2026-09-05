import html
import os
from typing import Any, Dict, List
import requests


class Notifier:
    def __init__(self):
        self.bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID")

    def send_telegram(self, message_html: str):
        if not self.bot_token or not self.chat_id:
            print("[INFO] Brak TELEGRAM_BOT_TOKEN lub TELEGRAM_CHAT_ID. Pomijam wysyłkę.")
            return

        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        try:
            res = requests.post(
                url,
                json={"chat_id": self.chat_id, "text": message_html, "parse_mode": "HTML"},
                timeout=10,
            )
            res.raise_for_status()
            print("✓ Pomyślnie wysłano raport na Telegram.")
        except Exception as e:
            print(f"✗ Błąd podczas wysyłki Telegram: {e}")

    @staticmethod
    def format_report(
        opportunities: List[Dict[str, Any]],
        all_metrics: List[Dict[str, Any]],
        all_calendar: List[Dict[str, Any]],
        capex_history: List[Dict[str, Any]],
        rf_rate: float,
    ) -> str:
        msg = "🎯 <b>ALPHAMOAT RADAR: RAPORT RYNKOWY</b>\n"
        msg += f"<i>Rentowność 10Y US Treasury: {rf_rate * 100:.2f}%</i>\n\n"

        # === 1. CZERWONE FLAGI I OSTRZEŻENIA ===
        threats = []
        for m in all_metrics:
            for s in m.get("danger_signals", []):
                threats.append(f"• <b>${m['ticker']}</b>: {s}")

        if threats:
            msg += "🚨 <b>SYSTEMOWE SYGNAŁY OSTRZEGAWCZE:</b>\n"
            msg += "\n".join(threats[:8]) + "\n\n"

        # === 2. WYKRYTE ASYMETRYCZNE OKAZJE ===
        if not opportunities:
            msg += "☕ <i>Brak spółek spełniających kryteria AlphaScore. Cierpliwość to zysk.</i>\n\n"
        else:
            msg += "🔥 <b>WYKRYTE ASYMETRYCZNE OKAZJE:</b>\n"
            for op in opportunities:
                growth_str = (
                    f"<b>{op['implied_growth_pct']}% r/r</b>"
                    if op["implied_growth_pct"] is not None
                    else "⚠️ <i>FCF &lt; 0</i>"
                )
                msg += f"💎 <b>${op['ticker']}</b> | AlphaScore: <b>{op['alpha_score']}/100</b>\n"
                msg += f" • Kurs: <code>${op['price']}</code> | FCF Yield: <b>{op['real_fcf_yield_pct']}%</b>\n"
                msg += f" • ROIC: <b>{op['roic_pct']}%</b> | Reverse DCF: {growth_str}\n"
                msg += f" • RSI Wildera: <b>{op['rsi_14']}</b> | 200 SMA: <b>{op['sma200_dev_pct']}%</b>\n"
                if op.get("net_buyback"):
                    msg += " • 💎 <i>Zarząd agresywnie skupuje akcje (Net Buyback)</i>\n"
                msg += "───────────────────\n"

        # === 3. PEŁNY KALENDARZ RAPORTÓW KWARTALNYCH (POSORTOWANY) ===
        msg += "📅 <b>KALENDARZ WYNIKÓW (WG TERMINU):</b>\n"
        for item in all_calendar:
            days = item["days_left"]
            ticker = item["ticker"]
            date_str = item["date"]

            if days is not None:
                if days <= 7:
                    ico = "🔴"
                    desc = f"za <b>{days} dni! (NIE KUPOWAĆ)</b>"
                elif days <= 21:
                    ico = "🟡"
                    desc = f"za {days} dni"
                else:
                    ico = "🟢"
                    desc = f"za {days} dni"
            else:
                ico = "⚪"
                desc = "TBD"

            msg += f"{ico} <b>${ticker:<5}</b>: <code>{date_str}</code> ({desc})\n"
        msg += "\n"

        # === 4. AUDYT CAPEX HYPERSCALERÓW AI ===
        if capex_history:
            msg += "🏗️ <b>AUDYT WYDATKÓW CAPEX AI (Czy tną inwestycje?):</b>\n"
            for h in capex_history:
                sign = "+" if h["capex_qoq_pct"] > 0 else ""
                msg += f"• <b>${h['ticker']}</b>: CapEx <code>${h['last_capex_billions']}B</code> ({sign}{h['capex_qoq_pct']}% kw/kw) | {h['eps_verdict']}\n"

        return msg