"""
src/backtesting/backtester.py
──────────────────────────────
Simulates trading based on historical model signals.

How it works:
  1. Walk forward through historical data day by day
  2. At each day, generate what the signal WOULD have been
     using only data available up to that point
  3. Simulate entering and exiting positions
  4. Track portfolio value over time

Key design principles (anti-lookahead-bias):
  - Signal generated using only past data
  - Position entered at NEXT day's open (realistic)
  - Position held for target_days then closed
  - No future data ever used in signal generation

Position sizing:
  - Fixed fractional: risk 2% of portfolio per trade
  - ATR-based stop loss (volatility-adjusted)
  - Max 3 concurrent positions

Performance metrics:
  - Total return vs buy-and-hold
  - Sharpe ratio (risk-adjusted return)
  - Max drawdown (worst peak-to-trough loss)
  - Win rate (% of profitable trades)
  - Calmar ratio (return / max drawdown)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from config.settings import settings
from src.utils.logger import log


# ─── Trade record ─────────────────────────────────────────────

@dataclass
class Trade:
    ticker:       str
    entry_date:   pd.Timestamp
    exit_date:    Optional[pd.Timestamp]
    entry_price:  float
    exit_price:   Optional[float]
    direction:    str          # LONG or SHORT
    shares:       float
    pnl:          float = 0.0
    pnl_pct:      float = 0.0
    status:       str = "open"   # open / closed / stopped


# ─── Backtester ───────────────────────────────────────────────

class Backtester:
    """
    Walk-forward backtester for ensemble signals.

    Simulates realistic trading:
      - Signals generated day by day
      - Positions entered next day at open
      - ATR-based stop losses
      - Fixed fractional position sizing
    """

    def __init__(
        self,
        initial_capital: float = 100_000.0,
        risk_per_trade:  float = 0.02,      # 2% risk per trade
        target_days:     int   = 5,          # hold period
        max_positions:   int   = 3,          # max concurrent
        stop_loss_atr:   float = 2.0,        # stop = 2x ATR
        bullish_threshold: float = 0.55,
        bearish_threshold: float = 0.45,
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
    ) -> dict:
        """
        Run backtest for one ticker.

        Args:
            ticker:      Ticker symbol
            model_probs: Series of daily ensemble probabilities
                         Index: DatetimeIndex matching price_df
            price_df:    OHLCV DataFrame with DatetimeIndex

        Returns:
            Dict with trades, equity curve, and metrics
        """
        log.info(f"[{ticker}] Running backtest...")

        # Align data
        common_idx = model_probs.index.intersection(price_df.index)
        if len(common_idx) < 20:
            log.warning(f"[{ticker}] Not enough aligned data")
            return self._empty_result(ticker)

        probs  = model_probs.loc[common_idx]
        prices = price_df.loc[common_idx].copy()

        # ATR for stop losses
        prices["atr"] = self._calculate_atr(prices)

        capital    = self.initial_capital
        equity     = [capital]
        dates      = [prices.index[0]]
        trades:    list[Trade] = []
        open_trades: list[Trade] = []

        for i in range(1, len(prices) - self.target_days):
            date  = prices.index[i]
            prob  = probs.iloc[i]
            row   = prices.iloc[i]
            close = row["close"]
            atr   = row.get("atr", close * 0.02)

            # ── Close positions that have reached target_days ─
            still_open = []
            for trade in open_trades:
                days_held = (date - trade.entry_date).days
                if days_held >= self.target_days * 1.4:
                    # Close at today's close
                    trade.exit_date  = date
                    trade.exit_price = close
                    trade.status     = "closed"

                    if trade.direction == "LONG":
                        trade.pnl = (
                            (close - trade.entry_price)
                            * trade.shares
                        )
                    else:
                        trade.pnl = (
                            (trade.entry_price - close)
                            * trade.shares
                        )

                    trade.pnl_pct = trade.pnl / (
                        trade.entry_price * trade.shares
                    )
                    capital += trade.pnl
                    trades.append(trade)
                else:
                    # Check stop loss
                    if trade.direction == "LONG":
                        stop = trade.entry_price - (
                            self.stop_loss_atr * atr
                        )
                        if close < stop:
                            trade.exit_date  = date
                            trade.exit_price = close
                            trade.status     = "stopped"
                            trade.pnl = (
                                (close - trade.entry_price)
                                * trade.shares
                            )
                            trade.pnl_pct = trade.pnl / (
                                trade.entry_price * trade.shares
                            )
                            capital += trade.pnl
                            trades.append(trade)
                            continue
                    still_open.append(trade)

            open_trades = still_open

            # ── Generate signal for today ──────────────────────
            if len(open_trades) >= self.max_positions:
                equity.append(capital)
                dates.append(date)
                continue

            signal = None
            if prob > self.bullish_threshold:
                signal = "LONG"
            elif prob < self.bearish_threshold:
                signal = "SHORT"

            # ── Enter new position ────────────────────────────
            if signal and i + 1 < len(prices):
                next_row   = prices.iloc[i + 1]
                entry_price = next_row.get("open", close)

                # Position sizing: risk 2% of capital
                stop_dist  = self.stop_loss_atr * atr
                if stop_dist > 0:
                    risk_amount = capital * self.risk_per_trade
                    shares = risk_amount / stop_dist
                    shares = min(
                        shares,
                        (capital * 0.3) / entry_price
                    )
                else:
                    shares = (capital * 0.1) / entry_price

                if shares > 0 and entry_price > 0:
                    trade = Trade(
                        ticker=ticker,
                        entry_date=prices.index[i + 1],
                        exit_date=None,
                        entry_price=entry_price,
                        exit_price=None,
                        direction=signal,
                        shares=round(shares, 4),
                    )
                    open_trades.append(trade)

            equity.append(capital)
            dates.append(date)

        # Close any remaining open trades at last price
        last_price = prices["close"].iloc[-1]
        last_date  = prices.index[-1]
        for trade in open_trades:
            trade.exit_date  = last_date
            trade.exit_price = last_price
            trade.status     = "closed"
            if trade.direction == "LONG":
                trade.pnl = (
                    (last_price - trade.entry_price) * trade.shares
                )
            else:
                trade.pnl = (
                    (trade.entry_price - last_price) * trade.shares
                )
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

    def _calculate_atr(
        self,
        df: pd.DataFrame,
        period: int = 14,
    ) -> pd.Series:
        """Average True Range — measures volatility."""
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
        trades: list[Trade],
        prices: pd.DataFrame,
    ) -> dict:
        """Calculate performance metrics."""
        if len(equity) < 2:
            return self._empty_metrics(ticker)

        # Total return
        total_return = (
            equity.iloc[-1] / equity.iloc[0] - 1
        )

        # Buy and hold return
        bah_return = (
            prices["close"].iloc[-1] /
            prices["close"].iloc[0] - 1
        )

        # Daily returns
        daily_returns = equity.pct_change().dropna()

        # Sharpe ratio (annualised, assume 252 trading days)
        if daily_returns.std() > 0:
            sharpe = (
                daily_returns.mean() /
                daily_returns.std() *
                np.sqrt(252)
            )
        else:
            sharpe = 0.0

        # Max drawdown
        rolling_max = equity.cummax()
        drawdown    = (equity - rolling_max) / rolling_max
        max_dd      = drawdown.min()

        # Calmar ratio
        calmar = (
            total_return / abs(max_dd)
            if max_dd != 0 else 0.0
        )

        # Trade stats
        closed = [t for t in trades if t.status != "open"]
        n_trades  = len(closed)
        if n_trades > 0:
            winners  = [t for t in closed if t.pnl > 0]
            win_rate = len(winners) / n_trades
            avg_win  = (
                np.mean([t.pnl_pct for t in winners])
                if winners else 0.0
            )
            losers   = [t for t in closed if t.pnl <= 0]
            avg_loss = (
                np.mean([t.pnl_pct for t in losers])
                if losers else 0.0
            )
        else:
            win_rate = avg_win = avg_loss = 0.0

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
            "final_capital":   round(float(equity.iloc[-1]), 2),
        }

        log.info(
            f"[{ticker}] Backtest: "
            f"return={total_return:.1%} | "
            f"vs BaH={bah_return:.1%} | "
            f"sharpe={sharpe:.2f} | "
            f"max_dd={max_dd:.1%} | "
            f"trades={n_trades}"
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
            "final_capital": self.initial_capital,
        }