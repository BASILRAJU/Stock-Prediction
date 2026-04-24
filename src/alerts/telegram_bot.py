"""
src/alerts/telegram_bot.py
───────────────────────────
Sends trading alerts to Telegram using direct HTTP.
Simpler and more reliable than python-telegram-bot library.
"""

from __future__ import annotations

import html
from datetime import datetime

import requests

from config.settings import settings
from src.utils.logger import log


class TelegramAlerter:
    """
    Sends formatted alerts to Telegram via HTTP API.

    Uses only the `requests` library — no heavy dependencies
    that conflict with torch/transformers.
    """

    def __init__(
        self,
        bot_token: str = None,
        chat_id:   str = None,
    ):
        self.bot_token = bot_token or settings.telegram_bot_token
        self.chat_id   = chat_id   or settings.telegram_chat_id

        if not self.bot_token or not self.chat_id:
            raise ValueError(
                "Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID "
                "in your .env file"
            )

        self.api_url = f"https://api.telegram.org/bot{self.bot_token}"

    def _send(self, message: str) -> bool:
        """Send a Telegram message via HTTP."""
        try:
            resp = requests.post(
                f"{self.api_url}/sendMessage",
                json={
                    "chat_id":    self.chat_id,
                    "text":       message,
                    "parse_mode": "HTML",
                    "disable_web_page_preview": True,
                },
                timeout=10,
            )
            if resp.status_code == 200:
                return True
            log.error(
                f"Telegram error {resp.status_code}: {resp.text}"
            )
            return False
        except Exception as e:
            log.error(f"Telegram send failed: {e}")
            return False

    def send_signal_alert(self, signal, recommendation=None) -> bool:
        """Send alert for a new trading signal."""
        if signal.signal == "NEUTRAL":
            return False

        emoji = "🟢" if signal.signal == "BULLISH" else "🔴"

        lines = [
            f"{emoji} <b>{signal.ticker}</b>  {signal.signal}",
            "",
            f"<b>Confidence:</b> {signal.confidence:.1%}",
            f"<b>Models:</b> xgb={signal.xgb_prob:.2f} "
            f"lstm={signal.lstm_prob:.2f} "
            f"cnn={signal.cnn_prob:.2f}",
            f"<b>Ensemble:</b> {signal.ensemble_prob:.3f}",
        ]

        if recommendation and recommendation.direction != "NO TRADE":
            if recommendation.shares < 1:
                shares_str = f"{recommendation.shares:.4f}"
            elif recommendation.shares != int(recommendation.shares):
                shares_str = f"{recommendation.shares:.2f}"
            else:
                shares_str = str(int(recommendation.shares))

            lines.extend([
                "",
                "<b>TRADE SETUP</b>",
                f"Entry:  ${recommendation.entry_price:.2f}",
                f"Stop:   ${recommendation.stop_loss:.2f}",
                f"Target: ${recommendation.target_1:.2f} (1:2 R:R)",
                f"Target2: ${recommendation.target_2:.2f} (1:3 R:R)",
                "",
                f"Shares: {shares_str}",
                f"Value:  ${recommendation.position_value:.2f}",
                f"Risk:   ${recommendation.max_risk_dollars:.2f} "
                f"({recommendation.risk_pct_capital:.1%})",
            ])

        lines.append("")
        lines.append(
            f"<i>{datetime.now().strftime('%Y-%m-%d %H:%M')}</i>"
        )

        return self._send("\n".join(lines))

    def send_news_alert(
        self,
        ticker:          str,
        headline:        str,
        sentiment_label: str,
        confidence:      float,
        source:          str = "",
    ) -> bool:
        """Send alert for a breaking news headline."""
        emoji = {
            "positive": "🟢",
            "negative": "🔴",
            "neutral":  "⚪",
        }.get(sentiment_label, "⚪")

        safe_headline = html.escape(headline[:200])
        safe_source   = html.escape(source[:50])

        lines = [
            f"{emoji} <b>{ticker}</b> NEWS — {sentiment_label.upper()}",
            "",
            f"<b>{safe_headline}</b>",
            "",
            f"Confidence: {confidence:.1%}",
        ]
        if source:
            lines.append(f"Source: {safe_source}")

        lines.append(
            f"<i>{datetime.now().strftime('%Y-%m-%d %H:%M')}</i>"
        )
        return self._send("\n".join(lines))

    def send_text(self, text: str) -> bool:
        """Send arbitrary text message."""
        return self._send(text)

    def test_connection(self) -> bool:
        """Send a test message to verify bot works."""
        return self.send_text(
            "✅ <b>Stock Signal Bot is online!</b>\n\n"
            "You'll receive alerts here when signals appear.\n\n"
            f"<i>Connected at "
            f"{datetime.now().strftime('%Y-%m-%d %H:%M')}</i>"
        )