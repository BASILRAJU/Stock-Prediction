"""
src/utils/models.py
────────────────────
Data models that validate everything flowing through the pipeline.
If bad data enters, these models catch it immediately.
"""

from __future__ import annotations
from datetime import date, datetime, timezone
from enum import Enum
from pydantic import BaseModel, Field, field_validator, model_validator


class Interval(str, Enum):
    MIN_1   = "1m"
    MIN_5   = "5m"
    MIN_15  = "15m"
    MIN_30  = "30m"
    HOUR_1  = "1h"
    DAY_1   = "1d"
    WEEK_1  = "1wk"
    MONTH_1 = "1mo"


class Regime(str, Enum):
    BULL     = "bull"
    BEAR     = "bear"
    SIDEWAYS = "sideways"
    UNKNOWN  = "unknown"


class DataRequest(BaseModel):
    """Validates what data we want to fetch before any API call."""
    tickers:              list[str]
    period:               str       = "2y"
    include_intraday:     bool      = False
    include_macro:        bool      = True
    include_fundamentals: bool      = True

    @field_validator("tickers", mode="before")
    @classmethod
    def normalise_tickers(cls, v):
        if isinstance(v, str):
            v = [t.strip() for t in v.split(",")]
        return [t.upper().strip() for t in v if t.strip()]

    @field_validator("period")
    @classmethod
    def valid_period(cls, v: str) -> str:
        valid = {
            "1d","5d","1mo","3mo","6mo",
            "1y","2y","5y","10y","ytd","max"
        }
        if v not in valid:
            raise ValueError(f"period must be one of {valid}")
        return v


class OHLCVBar(BaseModel):
    """A single price bar — rejects bad data immediately."""
    timestamp: datetime
    open:      float = Field(gt=0)
    high:      float = Field(gt=0)
    low:       float = Field(gt=0)
    close:     float = Field(gt=0)
    volume:    float = Field(ge=0)
    ticker:    str

    @model_validator(mode="after")
    def check_high_low(self) -> "OHLCVBar":
        if self.high < self.low:
            raise ValueError(
                f"high ({self.high}) cannot be less than low ({self.low})"
            )
        return self


class FundamentalSnapshot(BaseModel):
    """Key company metrics from Alpha Vantage."""
    ticker:               str
    fetched_at:           datetime
    market_cap:           float | None = None
    pe_ratio:             float | None = None
    pb_ratio:             float | None = None
    eps:                  float | None = None
    profit_margin:        float | None = None
    dividend_yield:       float | None = None
    beta:                 float | None = None
    week_52_high:         float | None = None
    week_52_low:          float | None = None
    analyst_target_price: float | None = None
    sector:               str          = ""
    industry:             str          = ""


class FeatureSet(BaseModel):
    """Summary of a completed feature matrix for one ticker."""
    ticker:       str
    interval:     Interval
    rows:         int
    columns:      int
    column_names: list[str]
    regime:       Regime   = Regime.UNKNOWN
    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


class PipelineResult(BaseModel):
    """Final result returned after running the full Phase 1 pipeline."""
    tickers_success: list[str]            = []
    tickers_failed:  list[str]            = []
    feature_sets:    dict[str, FeatureSet] = {}
    errors:          dict[str, str]        = {}
    duration_s:      float                 = 0.0
    completed_at:    datetime              = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    @property
    def success_rate(self) -> float:
        total = len(self.tickers_success) + len(self.tickers_failed)
        return len(self.tickers_success) / total if total else 0.0