"""
src/backtesting/backtester.py
──────────────────────────────
Adaptive trade simulation:

  TRENDING trades:
    - Liquidity-aware trailing stop (10-day swing low)
    - Only exits on extreme reversal (sweep + close)
    - Lets winners run

  MEAN-REVERTING trades:
    - Take-profit at resistance/POC
    - Immediate exit on bearish reversal patterns
    - Tighter exit logic

Classification at entry uses multi-timeframe alignment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from src.utils.logger import log


@dataclass
class Trade:
    ticker:        str
    entry_date:    pd.Timestamp
    exit_date:     Optional[pd.Timestamp]
    entry_price:   float
    exit_price:    Optional[float]
    direction:     str           # LONG or SHORT
    shares:        float
    pnl:           float = 0.0
    pnl_pct:       float = 0.0
    status:        str = "open"
    smart_stop:    float = 0.0
    trade_type:    str = "mean_reverting"  # or "trending"
    highest_close: float = 0.0   # tracking peak for trailing stop
    lowest_close:  float = 0.0   # tracking trough for SHORT trailing


class Backtester:
    """Adaptive backtester with trending vs mean-reverting trade modes."""

    def __init__(
        self,
        initial_capital:    float = 100_000.0,
        risk_per_trade:     float = 0.02,
        target_days:        int   = 5,
        max_positions:      int   = 3,
        stop_loss_atr:      float = 2.0,
        bullish_threshold:  float = 0.55,
        bearish_threshold:  float = 0.45,
    ):
        self.initial_capital   = initial_capital
        self.risk_per_trade    = risk_per_trade
        self.target_days       = target_days
        self.max_positions     = max_positions
        self.stop_loss_atr     = stop_loss_atr
        self.bullish_threshold = bullish_threshold
        self.bearish_threshold = bearish_threshold

    def run(
        self,
        ticker: str,
        model_probs: pd.Series,
        price_df: pd.DataFrame,
        per_model: pd.DataFrame = None,
    ) -> dict:
        log.info(f"[{ticker}] Running backtest...")

        common_idx = model_probs.index.intersection(price_df.index)
        if len(common_idx) < 20:
            log.warning(f"[{ticker}] Not enough aligned data")
            return self._empty_result(ticker)

        probs  = model_probs.loc[common_idx]
        prices = price_df.loc[common_idx].copy()

        if per_model is not None:
            per_model = per_model.loc[common_idx]

        prices["atr"] = self._calculate_atr(prices)

        capital      = self.initial_capital
        equity       = [capital]
        dates        = [prices.index[0]]
        trades       = []
        open_trades  = []

        for i in range(1, len(prices) - self.target_days):
            date  = prices.index[i]
            prob  = probs.iloc[i]
            row   = prices.iloc[i]
            close = row["close"]
            atr   = row.get("atr", close * 0.02)

            # ── Manage open positions ─────────────────────────
            still_open = []
            for trade in open_trades:
                exit_action = self._check_exit(
                    trade, prices, i, row, close, atr, date
                )

                if exit_action:
                    exit_status, exit_price = exit_action
                    trade.exit_date  = date
                    trade.exit_price = exit_price
                    trade.status     = exit_status
                    trade.pnl = self._calc_pnl(trade, exit_price)
                    trade.pnl_pct = trade.pnl / (
                        trade.entry_price * trade.shares
                    )
                    capital += trade.pnl
                    trades.append(trade)
                else:
                    # Update trailing stop for trending trades
                    if trade.trade_type == "trending":
                        self._update_trailing_stop(
                            trade, prices, i, close
                        )
                    still_open.append(trade)
            open_trades = still_open

            # ── Generate signal ───────────────────────────────
            if len(open_trades) >= self.max_positions:
                equity.append(capital)
                dates.append(date)
                continue

            signal = None
            if prob > self.bullish_threshold:
                signal = "LONG"
            elif prob < self.bearish_threshold:
                signal = "SHORT"

            # Trend filter
            if signal:
                close_now = row["close"]
                sma200 = row.get("sma_200", np.nan)
                if not np.isnan(sma200):
                    in_uptrend   = close_now > sma200 * 1.02
                    in_downtrend = close_now < sma200 * 0.98
                    if signal == "LONG" and in_downtrend:
                        signal = None
                    if signal == "SHORT" and in_uptrend:
                        signal = None

            # Agreement filter
            if signal and per_model is not None:
                row_models = per_model.iloc[i]
                xgb_p  = row_models.get("xgb",  0.5)
                lstm_p = row_models.get("lstm", 0.5)
                cnn_p  = row_models.get("cnn",  0.5)

                bull_count = sum([
                    xgb_p > 0.5, lstm_p > 0.5, cnn_p > 0.5
                ])
                bear_count = sum([
                    xgb_p < 0.5, lstm_p < 0.5, cnn_p < 0.5
                ])

                if signal == "LONG":
                    if bull_count < 2 or bear_count >= 2:
                        signal = None
                else:
                    if bear_count < 2 or bull_count >= 2:
                        signal = None

            # ── Enter new position ────────────────────────────
            if signal and i + 1 < len(prices):
                next_row    = prices.iloc[i + 1]
                entry_price = next_row.get("open", close)

                lookback = prices.iloc[max(0, i - 20):i]

                if signal == "LONG":
                    swing_low = lookback["low"].min()
                    smart_stop = swing_low * 0.995
                    stop_dist = entry_price - smart_stop

                    max_dist = entry_price * 0.04
                    if stop_dist > max_dist or stop_dist <= 0:
                        stop_dist = self.stop_loss_atr * atr
                        smart_stop = entry_price - stop_dist
                else:
                    swing_high = lookback["high"].max()
                    smart_stop = swing_high * 1.005
                    stop_dist = smart_stop - entry_price

                    max_dist = entry_price * 0.04
                    if stop_dist > max_dist or stop_dist <= 0:
                        stop_dist = self.stop_loss_atr * atr
                        smart_stop = entry_price + stop_dist

                if stop_dist > 0:
                    risk_amount = capital * self.risk_per_trade
                    shares = risk_amount / stop_dist
                    shares = min(
                        shares,
                        (capital * 0.3) / entry_price
                    )
                else:
                    shares = (capital * 0.1) / entry_price

                # Classify trade type using MTF alignment
                trade_type = self._classify_trade(row)

                if shares > 0 and entry_price > 0:
                    trade = Trade(
                        ticker=ticker,
                        entry_date=prices.index[i + 1],
                        exit_date=None,
                        entry_price=entry_price,
                        exit_price=None,
                        direction=signal,
                        shares=round(shares, 4),
                        smart_stop=smart_stop,
                        trade_type=trade_type,
                        highest_close=entry_price,
                        lowest_close=entry_price,
                    )
                    open_trades.append(trade)

            equity.append(capital)
            dates.append(date)

        # Close remaining trades
        last_price = prices["close"].iloc[-1]
        last_date  = prices.index[-1]
        for trade in open_trades:
            trade.exit_date  = last_date
            trade.exit_price = last_price
            trade.status     = "closed"
            trade.pnl = self._calc_pnl(trade, last_price)
            trade.pnl_pct = trade.pnl / (
                trade.entry_price * trade.shares
            )
            capital += trade.pnl
            trades.append(trade)

        equity.append(capital)
        dates.append(last_date)

        equity_series = pd.Series(equity, index=dates)
        metrics = self._calculate_metrics(
            ticker, equity_series, trades, prices
        )

        return {
            "ticker":  ticker,
            "equity":  equity_series,
            "trades":  trades,
            "metrics": metrics,
        }

    def _classify_trade(self, row: pd.Series) -> str:
        """
        Decide trade type based on multi-timeframe alignment.

        TRENDING: MTF alignment >=2 AND strength > 0.3
        MEAN_REVERTING: otherwise
        """
        mtf_align    = row.get("mtf_alignment", 0)
        mtf_strength = row.get("mtf_strength", 0)

        if pd.isna(mtf_align) or pd.isna(mtf_strength):
            return "mean_reverting"

        if mtf_align >= 2 and mtf_strength > 0.3:
            return "trending"
        return "mean_reverting"

    def _check_exit(
        self,
        trade: Trade,
        prices: pd.DataFrame,
        i: int,
        row: pd.Series,
        close: float,
        atr: float,
        date: pd.Timestamp,
    ) -> Optional[tuple]:
        """
        Check all exit conditions, return (status, exit_price) or None.

        Trending trades:  trailing stop, ignore most reversal signals
        Mean-reverting:   TP at resistance, exit on bearish reversal
        """
        days_held = (date - trade.entry_date).days
        resistance, support = self._get_levels(prices, i, lookback=20)

        if trade.direction == "LONG":
            cur_pnl_pct = (
                close - trade.entry_price
            ) / trade.entry_price
        else:
            cur_pnl_pct = (
                trade.entry_price - close
            ) / trade.entry_price

        # ── Always honor smart stop ────────────────────────
        smart_stop = trade.smart_stop
        if trade.direction == "LONG" and close < smart_stop:
            return ("stopped", close)
        if trade.direction == "SHORT" and close > smart_stop:
            return ("stopped", close)

        # ── Mean-reverting trade exits ─────────────────────
        if trade.trade_type == "mean_reverting":
            # TP at resistance
            if trade.direction == "LONG" and resistance:
                if (close >= resistance * 0.995 and
                        cur_pnl_pct > 0.015):
                    return ("tp_resistance", close)
                if (row["high"] >= resistance and
                        close < row["open"] and
                        cur_pnl_pct > 0.005):
                    return ("tp_resistance", close)
            elif trade.direction == "SHORT" and support:
                if (close <= support * 1.005 and
                        cur_pnl_pct > 0.015):
                    return ("tp_resistance", close)
                if (row["low"] <= support and
                        close > row["open"] and
                        cur_pnl_pct > 0.005):
                    return ("tp_resistance", close)

            # Bearish reversal exit (LONG only)
            if trade.direction == "LONG" and cur_pnl_pct > 0.005:
                bearish_engulfing = row.get("cdl_engulfing_bear", 0)
                shooting_star     = row.get("cdl_shooting_star", 0)
                bearish_pin       = row.get("lq_bearish_ob", 0)

                if (bearish_engulfing or shooting_star or
                        bearish_pin):
                    return ("reversal_exit", close)

            # Bullish reversal exit (SHORT only)
            if trade.direction == "SHORT" and cur_pnl_pct > 0.005:
                bullish_engulfing = row.get("cdl_engulfing_bull", 0)
                hammer            = row.get("cdl_hammer", 0)
                bullish_ob        = row.get("lq_bullish_ob", 0)

                if (bullish_engulfing or hammer or bullish_ob):
                    return ("reversal_exit", close)

        # ── Trending trade exits (only on extreme signals) ──
        else:  # trending
            # Liquidity sweep + reversal = strong exit signal
            sweep_reversal = row.get("lq_sweep_reversal", 0)
            if sweep_reversal and abs(cur_pnl_pct) > 0.02:
                return ("sweep_reversal_exit", close)

            # Strong reversal day (3%+ red bar in profitable LONG)
            if trade.direction == "LONG" and cur_pnl_pct > 0.03:
                day_pct = (close - row["open"]) / row["open"]
                if day_pct < -0.03:
                    return ("strong_reversal", close)
            elif trade.direction == "SHORT" and cur_pnl_pct > 0.03:
                day_pct = (close - row["open"]) / row["open"]
                if day_pct > 0.03:
                    return ("strong_reversal", close)

        # ── Time exit (held for full window) ───────────────
        if days_held >= self.target_days * 1.4:
            if cur_pnl_pct > 0.015:
                return ("target", close)
            elif cur_pnl_pct < -0.005:
                return ("time_loss", close)
            else:
                return ("time_neutral", close)

        return None

    def _update_trailing_stop(
        self,
        trade: Trade,
        prices: pd.DataFrame,
        i: int,
        close: float,
    ) -> None:
        """
        Update trailing stop for trending trade.

        LONG  → stop trails below 10-day swing low - 0.5%
        SHORT → stop trails above 10-day swing high + 0.5%

        Only ratchets in profitable direction.
        """
        if i < 10:
            return

        recent = prices.iloc[i - 10:i + 1]

        if trade.direction == "LONG":
            # Track highest close
            if close > trade.highest_close:
                trade.highest_close = close

            # Only trail once price has moved up at least 1.5% from entry
            if close > trade.entry_price * 1.015:
                new_stop = recent["low"].min() * 0.995
                if new_stop > trade.smart_stop:
                    trade.smart_stop = new_stop

        else:  # SHORT
            if close < trade.lowest_close or trade.lowest_close == 0:
                trade.lowest_close = close

            if close < trade.entry_price * 0.985:
                new_stop = recent["high"].max() * 1.005
                if new_stop < trade.smart_stop:
                    trade.smart_stop = new_stop

    def _get_levels(
        self,
        prices: pd.DataFrame,
        i: int,
        lookback: int = 20,
    ) -> tuple:
        if i < lookback:
            return None, None

        window = prices.iloc[i - lookback:i]
        resistance = float(window["high"].max())
        support    = float(window["low"].min())

        row = prices.iloc[i]
        vp_vah = row.get("vp_val_high")
        vp_val = row.get("vp_val_low")
        close_now = row["close"]

        if vp_vah and not pd.isna(vp_vah) and vp_vah > 0:
            if close_now < vp_vah < resistance:
                resistance = float(vp_vah)
        if vp_val and not pd.isna(vp_val) and vp_val > 0:
            if close_now > vp_val > support:
                support = float(vp_val)

        return resistance, support

    def _calc_pnl(self, trade: Trade, exit_price: float) -> float:
        if trade.direction == "LONG":
            return (exit_price - trade.entry_price) * trade.shares
        return (trade.entry_price - exit_price) * trade.shares

    def _calculate_atr(
        self, df: pd.DataFrame, period: int = 14
    ) -> pd.Series:
        high  = df["high"]
        low   = df["low"]
        close = df["close"]

        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low  - close.shift(1)).abs(),
        ], axis=1).max(axis=1)

        return tr.rolling(period).mean().fillna(close * 0.02)

    def _calculate_metrics(
        self,
        ticker: str,
        equity: pd.Series,
        trades: list,
        prices: pd.DataFrame,
    ) -> dict:
        if len(equity) < 2:
            return self._empty_metrics(ticker)

        total_return = (equity.iloc[-1] / equity.iloc[0] - 1)
        bah_return   = (
            prices["close"].iloc[-1] /
            prices["close"].iloc[0] - 1
        )

        daily_returns = equity.pct_change().dropna()
        if daily_returns.std() > 0:
            sharpe = (
                daily_returns.mean() /
                daily_returns.std() *
                np.sqrt(252)
            )
        else:
            sharpe = 0.0

        rolling_max = equity.cummax()
        drawdown    = (equity - rolling_max) / rolling_max
        max_dd      = drawdown.min()
        calmar      = (
            total_return / abs(max_dd) if max_dd != 0 else 0.0
        )

        closed = [t for t in trades if t.status != "open"]
        n_trades = len(closed)

        if n_trades > 0:
            winners = [t for t in closed if t.pnl > 0]
            win_rate = len(winners) / n_trades
            avg_win = (
                np.mean([t.pnl_pct for t in winners])
                if winners else 0.0
            )
            losers = [t for t in closed if t.pnl <= 0]
            avg_loss = (
                np.mean([t.pnl_pct for t in losers])
                if losers else 0.0
            )

            n_target = len([
                t for t in closed if t.status == "target"
            ])
            n_tp_resistance = len([
                t for t in closed if t.status == "tp_resistance"
            ])
            n_reversal = len([
                t for t in closed if t.status == "reversal_exit"
            ])
            n_sweep_rev = len([
                t for t in closed
                if t.status == "sweep_reversal_exit"
            ])
            n_strong_rev = len([
                t for t in closed
                if t.status == "strong_reversal"
            ])
            n_stopped = len([
                t for t in closed if t.status == "stopped"
            ])
            n_time_loss = len([
                t for t in closed if t.status == "time_loss"
            ])
            n_time_neutral = len([
                t for t in closed if t.status == "time_neutral"
            ])
            n_trending = len([
                t for t in closed if t.trade_type == "trending"
            ])
            n_mean_rev = len([
                t for t in closed if t.trade_type == "mean_reverting"
            ])
        else:
            win_rate = avg_win = avg_loss = 0.0
            n_target = n_tp_resistance = n_reversal = 0
            n_sweep_rev = n_strong_rev = 0
            n_stopped = n_time_loss = n_time_neutral = 0
            n_trending = n_mean_rev = 0

        metrics = {
            "ticker":          ticker,
            "total_return":    round(total_return, 4),
            "bah_return":      round(bah_return, 4),
            "alpha":           round(total_return - bah_return, 4),
            "sharpe":          round(float(sharpe), 3),
            "max_drawdown":    round(float(max_dd), 4),
            "calmar":          round(calmar, 3),
            "n_trades":        n_trades,
            "win_rate":        round(win_rate, 4),
            "avg_win_pct":     round(avg_win, 4),
            "avg_loss_pct":    round(avg_loss, 4),
            "n_target":        n_target,
            "n_tp_resistance": n_tp_resistance,
            "n_reversal":      n_reversal,
            "n_sweep_rev":     n_sweep_rev,
            "n_strong_rev":    n_strong_rev,
            "n_stopped":       n_stopped,
            "n_time_loss":     n_time_loss,
            "n_time_neutral":  n_time_neutral,
            "n_trending":      n_trending,
            "n_mean_rev":      n_mean_rev,
            "final_capital":   round(float(equity.iloc[-1]), 2),
        }

        log.info(
            f"[{ticker}] Backtest: "
            f"return={total_return:.1%} | "
            f"vs BaH={bah_return:.1%} | "
            f"sharpe={sharpe:.2f} | "
            f"trades={n_trades} "
            f"(trend={n_trending} mean={n_mean_rev})"
        )
        return metrics

    def _empty_result(self, ticker: str) -> dict:
        return {
            "ticker":  ticker,
            "equity":  pd.Series(dtype=float),
            "trades":  [],
            "metrics": self._empty_metrics(ticker),
        }

    def _empty_metrics(self, ticker: str) -> dict:
        return {
            "ticker": ticker,
            "total_return": 0.0, "bah_return": 0.0,
            "alpha": 0.0, "sharpe": 0.0,
            "max_drawdown": 0.0, "calmar": 0.0,
            "n_trades": 0, "win_rate": 0.0,
            "avg_win_pct": 0.0, "avg_loss_pct": 0.0,
            "n_target": 0, "n_tp_resistance": 0,
            "n_reversal": 0, "n_sweep_rev": 0, "n_strong_rev": 0,
            "n_stopped": 0, "n_time_loss": 0, "n_time_neutral": 0,
            "n_trending": 0, "n_mean_rev": 0,
            "final_capital": self.initial_capital,
        }