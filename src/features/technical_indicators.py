"""
src/features/technical_indicators.py
──────────────────────────────────────
Builds 60+ technical indicator features using pandas-ta.

Feature groups:
  - Trend:      SMA, EMA, MACD, ADX, Parabolic SAR
  - Momentum:   RSI, Stochastic, Williams %R, CCI, ROC, MFI
  - Volatility: Bollinger Bands, ATR, Keltner, Donchian, Realised Vol
  - Volume:     OBV, VWAP, CMF, Volume ratios, Force Index
  - Returns:    Multi-horizon returns, log returns, overnight gap
  - Patterns:   Doji, Hammer, Engulfing, body/wick sizes
  - Regime:     Trend slope, drawdown, above/below key MAs

Research basis:
  XGBoost study (2025): price features >60% importance
  RSI/BB contribute ~14-15% but help with risk filtering
  MPDTransformer: converts these features to 2D images for CNN
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd

try:
    import pandas_ta as ta
    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False
    warnings.warn("pandas-ta not installed. Run: pip install pandas-ta")

from config.settings import settings
from src.utils.logger import log

# Return horizons in trading days
RETURN_HORIZONS = [1, 3, 5, 10, 20]

# Realised volatility windows
VOL_WINDOWS = [5, 10, 20]


class TechnicalFeatureEngine:
    """
    Transforms raw OHLCV data into a rich feature matrix.

    Usage:
        engine = TechnicalFeatureEngine()
        features = engine.build(df, ticker="AAPL")
    """

    def build(
        self,
        df: pd.DataFrame,
        ticker: str = "",
        drop_nulls: bool = True,
    ) -> pd.DataFrame:
        """
        Build all features for one ticker.

        Args:
            df:          OHLCV DataFrame from YFinanceFetcher
                         Must have: open, high, low, close, volume columns
            ticker:      Ticker name for logging
            drop_nulls:  Drop warm-up rows with NaN values (recommended)

        Returns:
            Wide DataFrame with all features + original OHLCV
        """
        if not TA_AVAILABLE:
            raise ImportError("pandas-ta is required")

        df = df.copy()
        df.columns = [c.lower() for c in df.columns]
        log.info(f"[{ticker}] Building features from {len(df)} bars")

        # Build each feature group
        groups = [
            ("Trend",      self._trend(df)),
            ("Momentum",   self._momentum(df)),
            ("Volatility", self._volatility(df)),
            ("Volume",     self._volume(df)),
            ("Returns",    self._returns(df)),
            ("Patterns",   self._patterns(df)),
            ("Regime",     self._regime(df)),
        ]

        # Start with OHLCV columns
        result = df[["open", "high", "low", "close", "volume"]].copy()

        # Join each feature group
        for name, feat_df in groups:
            if feat_df is None or feat_df.empty:
                log.warning(f"[{ticker}] {name}: empty — skipping")
                continue
            # Only add columns not already present
            new_cols = [
                c for c in feat_df.columns
                if c not in result.columns
            ]
            result = result.join(feat_df[new_cols], how="left")
            log.debug(
                f"[{ticker}] {name}: +{len(new_cols)} features"
            )

        # Replace infinite values with NaN
        result = result.replace([np.inf, -np.inf], np.nan)

        # Drop warm-up rows that have NaN from indicator calculation
        if drop_nulls:
            before = len(result)
            result = result.dropna()
            dropped = before - len(result)
            if dropped > 0:
                log.debug(
                    f"[{ticker}] Dropped {dropped} warm-up rows "
                    f"({len(result)} rows remaining)"
                )

        log.info(
            f"[{ticker}] Feature matrix ready: "
            f"{result.shape[0]} rows x {result.shape[1]} columns"
        )
        return result

    # ── Trend Features ────────────────────────────────────────

    def _trend(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]
        high  = df["high"]
        low   = df["low"]
        f     = pd.DataFrame(index=df.index)

        # Simple Moving Averages
        for w in settings.sma_windows:
            f[f"sma_{w}"] = ta.sma(close, length=w)
            f[f"price_to_sma_{w}"] = close / f[f"sma_{w}"] - 1

        # Exponential Moving Averages
        for w in settings.ema_windows:
            f[f"ema_{w}"] = ta.ema(close, length=w)
            f[f"price_to_ema_{w}"] = close / f[f"ema_{w}"] - 1

        # EMA crossover signals
        if "ema_9" in f and "ema_21" in f:
            f["ema_9_21_cross"] = np.where(
                f["ema_9"] > f["ema_21"], 1, -1
            )
        if "ema_21" in f and "ema_55" in f:
            f["ema_21_55_cross"] = np.where(
                f["ema_21"] > f["ema_55"], 1, -1
            )

        # Golden/Death cross (SMA 50 vs 200)
        if "sma_50" in f and "sma_200" in f:
            f["golden_cross"] = np.where(
                f["sma_50"] > f["sma_200"], 1, -1
            )

        # MACD
        macd_df = ta.macd(
            close,
            fast=settings.macd_fast,
            slow=settings.macd_slow,
            signal=settings.macd_signal,
        )
        if macd_df is not None and not macd_df.empty:
            cols = macd_df.columns.tolist()
            f["macd"]        = macd_df[cols[0]]
            f["macd_hist"]   = macd_df[cols[1]]
            f["macd_signal"] = macd_df[cols[2]]
            f["macd_cross"]  = np.where(
                f["macd"] > f["macd_signal"], 1, -1
            )

        # ADX - trend strength (not direction)
        adx_df = ta.adx(high, low, close, length=settings.adx_period)
        if adx_df is not None and not adx_df.empty:
            cols = adx_df.columns.tolist()
            f["adx"]     = adx_df[cols[0]]
            f["adx_dmp"] = adx_df[cols[1]]
            f["adx_dmn"] = adx_df[cols[2]]
            f["di_diff"] = f["adx_dmp"] - f["adx_dmn"]

        return f

    # ── Momentum Features ─────────────────────────────────────

    def _momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        close  = df["close"]
        high   = df["high"]
        low    = df["low"]
        volume = df["volume"]
        f      = pd.DataFrame(index=df.index)

        # RSI
        f["rsi"] = ta.rsi(close, length=settings.rsi_period)
        f["rsi_overbought"] = (f["rsi"] > 70).astype(int)
        f["rsi_oversold"]   = (f["rsi"] < 30).astype(int)

        # Stochastic
        stoch = ta.stoch(
            high, low, close,
            k=settings.stoch_k,
            d=settings.stoch_d,
        )
        if stoch is not None and not stoch.empty:
            cols = stoch.columns.tolist()
            f["stoch_k"] = stoch[cols[0]]
            f["stoch_d"] = stoch[cols[1]]
            f["stoch_cross"] = np.where(
                f["stoch_k"] > f["stoch_d"], 1, -1
            )

        # Williams %R
        f["williams_r"] = ta.willr(
            high, low, close,
            length=settings.williams_r_period,
        )

        # CCI
        f["cci"] = ta.cci(
            high, low, close,
            length=settings.cci_period,
        )

        # Rate of Change
        for period in [3, 5, 10, 20]:
            f[f"roc_{period}"] = ta.roc(close, length=period)

        # Money Flow Index (volume-weighted RSI)
        f["mfi"] = ta.mfi(high, low, close, volume, length=14)

        return f

    # ── Volatility Features ───────────────────────────────────

    def _volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]
        high  = df["high"]
        low   = df["low"]
        f     = pd.DataFrame(index=df.index)

        # Bollinger Bands
        bb = ta.bbands(
            close,
            length=settings.bb_period,
            std=settings.bb_std,
        )
        if bb is not None and not bb.empty:
            cols = bb.columns.tolist()
            f["bb_lower"] = bb[cols[0]]
            f["bb_mid"]   = bb[cols[1]]
            f["bb_upper"] = bb[cols[2]]
            f["bb_width"] = (
                (f["bb_upper"] - f["bb_lower"]) / f["bb_mid"]
            )
            f["bb_pct"] = (
                (close - f["bb_lower"]) /
                (f["bb_upper"] - f["bb_lower"])
            )
            f["bb_squeeze"] = (
                f["bb_width"] < f["bb_width"].rolling(20).mean()
            ).astype(int)

        # ATR - key for stop loss sizing in Phase 5
        f["atr"]     = ta.atr(high, low, close, length=settings.atr_period)
        f["atr_pct"] = f["atr"] / close

        # Keltner Channels
        kc = ta.kc(high, low, close, length=20)
        if kc is not None and not kc.empty:
            cols = kc.columns.tolist()
            f["kc_upper"] = kc[cols[0]]
            f["kc_lower"] = kc[cols[2]]
            denom = f["kc_upper"] - f["kc_lower"]
            f["kc_pct"] = np.where(
                denom != 0,
                (close - f["kc_lower"]) / denom,
                np.nan,
            )

        # Donchian Channels - support/resistance proxy
        dc = ta.donchian(high, low, length=20)
        if dc is not None and not dc.empty:
            cols = dc.columns.tolist()
            f["dc_upper"] = dc[cols[0]]
            f["dc_lower"] = dc[cols[2]]
            f["dc_width"] = f["dc_upper"] - f["dc_lower"]

        # Realised volatility (annualised)
        log_ret = np.log(close / close.shift(1))
        for w in VOL_WINDOWS:
            f[f"realised_vol_{w}d"] = (
                log_ret.rolling(w).std() * np.sqrt(252)
            )

        # High volatility regime flag
        if "realised_vol_20d" in f:
            f["high_vol_regime"] = (
                f["realised_vol_20d"] >
                f["realised_vol_20d"].rolling(60).mean()
            ).astype(int)

        return f

    # ── Volume Features ───────────────────────────────────────

    def _volume(self, df: pd.DataFrame) -> pd.DataFrame:
        close  = df["close"]
        high   = df["high"]
        low    = df["low"]
        volume = df["volume"]
        f      = pd.DataFrame(index=df.index)

        # OBV - On Balance Volume
        f["obv"]        = ta.obv(close, volume)
        f["obv_ema"]    = ta.ema(f["obv"], length=20)
        f["obv_signal"] = np.where(f["obv"] > f["obv_ema"], 1, -1)

        # VWAP approximation (daily anchor)
        typical_price = (high + low + close) / 3
        f["vwap"] = (
            (typical_price * volume).rolling(20).sum() /
            volume.rolling(20).sum()
        )
        f["price_to_vwap"] = close / f["vwap"] - 1

        # Chaikin Money Flow
        f["cmf"] = ta.cmf(high, low, close, volume, length=20)

        # Volume ratios
        vol_sma_20      = volume.rolling(20).mean()
        f["vol_sma_20"] = vol_sma_20
        f["vol_ratio_20"] = volume / vol_sma_20
        f["vol_ratio_5"]  = volume / volume.rolling(5).mean()

        # Force Index
        f["force_index"]     = (close - close.shift(1)) * volume
        f["force_index_ema"] = ta.ema(f["force_index"], length=13)

        return f

    # ── Return Features ───────────────────────────────────────

    def _returns(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]
        open_ = df["open"]
        f     = pd.DataFrame(index=df.index)

        # Multi-horizon returns (these become model targets too)
        for h in RETURN_HORIZONS:
            f[f"return_{h}d"] = close.pct_change(h)

        # Log returns
        f["log_return_1d"] = np.log(close / close.shift(1))
        f["log_return_5d"] = np.log(close / close.shift(5))

        # Overnight gap
        f["overnight_gap"] = (open_ - close.shift(1)) / close.shift(1)
        f["gap_up"]        = (f["overnight_gap"] >  0.01).astype(int)
        f["gap_down"]      = (f["overnight_gap"] < -0.01).astype(int)

        # Intraday range
        f["intraday_range"]  = (df["high"] - df["low"]) / df["open"]
        f["close_position"]  = (
            (close - df["low"]) /
            (df["high"] - df["low"])
        )

        # Distance from 52-week high/low
        f["pct_from_52w_high"] = close / close.rolling(252).max() - 1
        f["pct_from_52w_low"]  = close / close.rolling(252).min() - 1

        # Lagged returns as features
        for lag in [1, 2, 3, 5]:
            f[f"lag_return_{lag}d"] = f["return_1d"].shift(lag)

        return f

    # ── Pattern Features ──────────────────────────────────────

    def _patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        open_ = df["open"]
        high  = df["high"]
        low   = df["low"]
        close = df["close"]
        f     = pd.DataFrame(index=df.index)

        try:
            f["cdl_doji"]          = ta.cdl_doji(open_, high, low, close)
            f["cdl_hammer"]        = ta.cdl_hammer(open_, high, low, close)
            f["cdl_engulfing"]     = ta.cdl_engulfing(open_, high, low, close)
            f["cdl_morning_star"]  = ta.cdl_morning_star(open_, high, low, close)
            f["cdl_evening_star"]  = ta.cdl_evening_star(open_, high, low, close)
        except Exception as e:
            log.warning(f"Candlestick patterns failed: {e}")

        # Body and wick sizes
        f["body_size"]   = abs(close - open_) / open_
        f["upper_wick"]  = (
            high - pd.concat([open_, close], axis=1).max(axis=1)
        ) / open_
        f["lower_wick"]  = (
            pd.concat([open_, close], axis=1).min(axis=1) - low
        ) / open_

        return f

    # ── Regime Features ───────────────────────────────────────

    def _regime(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]
        f     = pd.DataFrame(index=df.index)

        # Price above/below key MAs
        sma_50  = ta.sma(close, length=50)
        sma_200 = ta.sma(close, length=200)
        f["above_sma50"]  = (close > sma_50).astype(int)
        f["above_sma200"] = (close > sma_200).astype(int)

        # Trend slope over 20 days
        f["trend_slope_20"] = close.rolling(20).apply(
            lambda x: (
                np.polyfit(range(len(x)), x, 1)[0] / x.mean()
                if x.mean() != 0 else 0
            ),
            raw=True,
        )

        # Drawdown from rolling peak
        rolling_max   = close.cummax()
        f["drawdown"] = (close - rolling_max) / rolling_max

        return f


# ── Convenience function ──────────────────────────────────────

def build_features(
    df: pd.DataFrame,
    ticker: str = "",
    drop_nulls: bool = True,
) -> pd.DataFrame:
    """
    Quick way to build features without creating an engine object.

    Usage:
        from src.features.technical_indicators import build_features
        features = build_features(df, ticker="AAPL")
    """
    return TechnicalFeatureEngine().build(
        df, ticker=ticker, drop_nulls=drop_nulls
    )