"""
config/settings.py
──────────────────
Central configuration for the entire Stock Prediction system.
Reads all values from your .env file automatically.
"""

from __future__ import annotations
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT_DIR / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── API Keys ──────────────────────────────────────────────
    alpha_vantage_api_key: str = "demo"
    news_api_key: str = ""
    telegram_bot_token: str = ""
    telegram_chat_id:   str = ""

    # ── Storage Paths ─────────────────────────────────────────
    data_raw_path: Path = ROOT_DIR / "data" / "raw"
    data_processed_path: Path = ROOT_DIR / "data" / "processed"
    data_cache_path: Path = ROOT_DIR / "data" / "cache"
    log_file: Path = ROOT_DIR / "logs" / "pipeline.log"
    log_level: str = "INFO"

    # ── Your Tickers ──────────────────────────────────────────
    # US Stocks (22 tickers)
    us_tickers: list[str] = [
        # Tier 1 — Mega-Cap Tech (7)
        "AAPL",   # Apple
        "MSFT",   # Microsoft
        "NVDA",   # Nvidia
        "GOOGL",  # Alphabet
        "META",   # Meta
        "AMZN",   # Amazon
        "TSLA",   # Tesla

        # Tier 2 — Semi Infrastructure (4)
        "AVGO",   # Broadcom
        "AMD",    # AMD
        "ASML",   # ASML
        "AMAT",   # Applied Materials

        # Tier 3 — Financials & Payments (3)
        "JPM",    # JPMorgan
        "GS",     # Goldman Sachs
        "V",      # Visa

        # Tier 4 — US Energy (2)
        "XOM",    # ExxonMobil
        "CVX",    # Chevron

        # Tier 5 — Defensive / Hedges (3)
        "GLD",    # Gold ETF
        "ETR",    # Entergy (utility)
        "SPY",    # S&P 500 ETF

        # Tier 6 — Speculation (2)
        "PLTR",   # Palantir
        "BTC-USD", # Bitcoin
    ]

    # Canadian Stocks - Toronto Stock Exchange (5 tickers)
    ca_tickers: list[str] = [
        "CSU.TO",  # Constellation Software
        "BN.TO",   # Brookfield Corp
        "ATD.TO",  # Couche-Tard
        "ENB.TO",  # Enbridge
        "CNQ.TO",  # Canadian Natural Resources
        "VFV.TO",  # Vanguard S&P 500 ETF (CAD)
    ]

    @property
    def all_tickers(self) -> list[str]:
        return self.us_tickers + self.ca_tickers

    # ── yfinance Settings ─────────────────────────────────────
    yf_daily_period: str = "10y"
    yf_intraday_interval: str = "1h"
    yf_intraday_period: str = "60d"

    # ── Alpha Vantage Settings ────────────────────────────────
    av_request_delay_s: float = 12.5  # free tier: 5 req/min
    av_max_retries: int = 3

    # ── Technical Indicator Settings ──────────────────────────
    sma_windows: list[int] = [10, 20, 50, 200]
    ema_windows: list[int] = [9, 21, 55]
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_period: int = 20
    bb_std: float = 2.0
    atr_period: int = 14
    adx_period: int = 14
    stoch_k: int = 14
    stoch_d: int = 3
    cci_period: int = 20
    williams_r_period: int = 14

    # ── Macro Indicators to fetch from Alpha Vantage ──────────
    macro_indicators: list[str] = [
        "REAL_GDP",
        "CPI",
        "FEDERAL_FUNDS_RATE",
        "UNEMPLOYMENT",
        "TREASURY_YIELD",
    ]

    def create_directories(self) -> None:
        """Create all data directories if they don't exist."""
        for path in [
            self.data_raw_path,
            self.data_processed_path,
            self.data_cache_path,
            self.log_file.parent,
        ]:
            path.mkdir(parents=True, exist_ok=True)


# Single instance used everywhere in the project
settings = Settings()
settings.create_directories()