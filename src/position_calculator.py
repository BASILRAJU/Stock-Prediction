"""
src/position_calculator.py
───────────────────────────
Position size calculator for retail traders.

Given a trading signal and your capital, tells you exactly:
  - How many shares to buy/sell (fractional supported)
  - Where to put your stop loss
  - What your profit target is
  - Maximum risk in dollars

Based on professional risk management principles:
  - Never risk more than 2% of capital per trade
  - ATR-based stop loss (volatility-adjusted)
  - Risk:Reward minimum 1:2
  - Maximum 3 concurrent positions
  - Never deploy more than 30% capital in one position
  - Supports fractional shares for small accounts

Research basis:
  Van Tharp "Trade Your Way to Financial Freedom":
  position sizing is the most important factor in
  long-term trading success — more than entry/exit timing.

  Kelly Criterion: mathematically optimal bet size
  = Edge / Odds. We use half-Kelly for safety.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from config.settings import settings
from src.utils.logger import log


@dataclass
class PositionRecommendation:
    """Complete trade recommendation for one ticker."""
    ticker:          str
    signal:          str          # BULLISH / BEARISH / NEUTRAL
    direction:       str          # LONG / SHORT / NO TRADE
    confidence:      float

    # Price levels
    entry_price:     float        # suggested entry
    stop_loss:       float        # where to cut loss
    target_1:        float        # first profit target (1:2 R:R)
    target_2:        float        # second target (1:3 R:R)

    # Position sizing
    shares:          float        # number of shares (fractional ok)
    position_value:  float        # total $ deployed
    max_risk_dollars: float       # max $ you can lose
    risk_pct_capital: float       # % of your capital at risk

    # Context
    atr:             float        # current volatility measure
    capital:         float        # your total capital
    notes:           str          # human-readable explanation

    def __str__(self) -> str:
        if self.direction == "NO TRADE":
            return (
                f"⚪ {self.ticker:8s} NO TRADE — "
                f"{self.notes}"
            )

        emoji = "🟢" if self.direction == "LONG" else "🔴"

        # Format share count differently for fractional vs whole
        if self.shares < 1:
            shares_str = f"{self.shares:.4f} shares"
        elif self.shares != int(self.shares):
            shares_str = f"{self.shares:.2f} shares"
        else:
            shares_str = f"{int(self.shares)} shares"

        stop_pct = (
            abs(self.entry_price - self.stop_loss) /
            self.entry_price
        )

        return f"""
{emoji} {self.ticker} — {self.direction} ({self.signal})
──────────────────────────────────────────────────
  Entry:      ${self.entry_price:.2f}
  Stop Loss:  ${self.stop_loss:.2f}  ({stop_pct:.1%} from entry)
  Target 1:   ${self.target_1:.2f}  (1:2 risk/reward)
  Target 2:   ${self.target_2:.2f}  (1:3 risk/reward)

  Shares:     {shares_str}
  Position:   ${self.position_value:,.2f}  ({self.position_value/self.capital:.1%} of capital)
  Max Risk:   ${self.max_risk_dollars:.2f}  ({self.risk_pct_capital:.1%} of capital)
  ATR:        ${self.atr:.2f}

  {self.notes}"""


class PositionCalculator:
    """
    Calculates optimal position sizes based on:
    - Your available capital
    - Signal confidence
    - Current stock volatility (ATR)
    - Risk management rules
    - Fractional shares for small accounts

    Usage:
        calc = PositionCalculator(capital=500)
        rec = calc.calculate('AAPL', 'BULLISH', 0.25, features_df)
        print(rec)
    """

    def __init__(
        self,
        capital:              float = 5000.0,
        max_risk_pct:         float = 0.02,    # 2% max risk per trade
        max_position_pct:     float = 0.30,    # 30% max per position
        max_positions:        int   = 3,        # max concurrent trades
        atr_stop_mult:        float = 2.0,      # stop = 2x ATR from entry
        min_rr_ratio:         float = 2.0,      # minimum risk:reward
        allow_fractional:     bool  = True,     # allow fractional shares
        min_trade_value:      float = 10.0,     # minimum $10 per trade
    ):
        self.capital          = capital
        self.max_risk_pct     = max_risk_pct
        self.max_position_pct = max_position_pct
        self.max_positions    = max_positions
        self.atr_stop_mult    = atr_stop_mult
        self.min_rr_ratio     = min_rr_ratio
        self.allow_fractional = allow_fractional
        self.min_trade_value  = min_trade_value

    def calculate(
        self,
        ticker: str,
        signal_str: str,
        confidence: float,
        features_df: pd.DataFrame,
        open_positions: int = 0,
    ) -> PositionRecommendation:
        """
        Calculate position size for one trade.

        Args:
            ticker:          Stock ticker
            signal_str:      'BULLISH', 'BEARISH', or 'NEUTRAL'
            confidence:      Signal confidence 0.0-1.0
            features_df:     Feature DataFrame with OHLCV + ATR
            open_positions:  Number of currently open trades

        Returns:
            PositionRecommendation with all trade details
        """
        # ── No trade conditions ───────────────────────────────
        if signal_str == "NEUTRAL":
            return self._no_trade(
                ticker, signal_str, confidence,
                "Signal is NEUTRAL — no edge detected"
            )

        if open_positions >= self.max_positions:
            return self._no_trade(
                ticker, signal_str, confidence,
                f"Already at max positions ({self.max_positions})"
            )

        if confidence < 0.05:
            return self._no_trade(
                ticker, signal_str, confidence,
                f"Confidence too low ({confidence:.1%}) — minimum 5%"
            )

        # ── Get current price and ATR ─────────────────────────
        latest = features_df.dropna(subset=["close", "atr"]).iloc[-1]
        current_price = float(latest["close"])
        atr           = float(latest["atr"])

        if current_price <= 0 or atr <= 0:
            return self._no_trade(
                ticker, signal_str, confidence,
                "Invalid price or ATR data"
            )

        direction = "LONG" if signal_str == "BULLISH" else "SHORT"

        # ── Calculate stop loss and targets ───────────────────
        stop_distance = self.atr_stop_mult * atr

        if direction == "LONG":
            entry_price = current_price
            stop_loss   = entry_price - stop_distance
            target_1    = entry_price + (stop_distance * self.min_rr_ratio)
            target_2    = entry_price + (stop_distance * 3.0)
        else:
            entry_price = current_price
            stop_loss   = entry_price + stop_distance
            target_1    = entry_price - (stop_distance * self.min_rr_ratio)
            target_2    = entry_price - (stop_distance * 3.0)

        # ── Calculate position size ───────────────────────────
        # Base risk: 2% of capital
        base_risk = self.capital * self.max_risk_pct

        # Scale risk by confidence (higher confidence → slightly larger)
        # Max scaling: 1.5x at 100% confidence
        confidence_scale = 1.0 + (confidence * 0.5)
        adjusted_risk    = base_risk * confidence_scale

        # Shares = Risk amount / Stop distance per share
        shares_by_risk = adjusted_risk / stop_distance

        # Cap at max position size (30% of capital)
        max_shares_by_capital = (
            self.capital * self.max_position_pct / entry_price
        )

        # ── Choose fractional or whole shares ─────────────────
        # Use fractional when account is small relative to stock price
        use_fractional = (
            self.allow_fractional and
            self.capital < entry_price * 5
        )

        if use_fractional:
            shares = min(shares_by_risk, max_shares_by_capital)
            shares = round(shares, 4)  # 4 decimals = 0.0001 precision

            # Minimum trade value check
            if shares * entry_price < self.min_trade_value:
                return self._no_trade(
                    ticker, signal_str, confidence,
                    f"Position too small: "
                    f"${shares * entry_price:.2f} "
                    f"(min ${self.min_trade_value:.2f})"
                )
        else:
            shares = int(min(shares_by_risk, max_shares_by_capital))
            shares = max(shares, 1)

        # ── Affordability check ───────────────────────────────
        position_value = shares * entry_price

        if position_value > self.capital:
            if use_fractional:
                shares = round(self.capital / entry_price, 4)
            else:
                shares = int(self.capital / entry_price)
            position_value = shares * entry_price

        if shares <= 0 or position_value < self.min_trade_value:
            return self._no_trade(
                ticker, signal_str, confidence,
                f"Insufficient capital. "
                f"Need min ${self.min_trade_value:.2f}, "
                f"can only allocate ${position_value:.2f}"
            )

        # ── Final risk calculation ────────────────────────────
        actual_risk = shares * stop_distance
        risk_pct    = actual_risk / self.capital

        # ── Build human-readable notes ────────────────────────
        potential_profit = shares * stop_distance * self.min_rr_ratio
        frac_note = " (fractional shares)" if use_fractional else ""

        notes = (
            f"Risk ${actual_risk:.2f} to make "
            f"${potential_profit:.2f} "
            f"(1:{self.min_rr_ratio:.0f} R:R){frac_note} | "
            f"Confidence: {confidence:.1%}"
        )

        return PositionRecommendation(
            ticker=ticker,
            signal=signal_str,
            direction=direction,
            confidence=confidence,
            entry_price=round(entry_price, 2),
            stop_loss=round(stop_loss, 2),
            target_1=round(target_1, 2),
            target_2=round(target_2, 2),
            shares=shares,
            position_value=round(position_value, 2),
            max_risk_dollars=round(actual_risk, 2),
            risk_pct_capital=round(risk_pct, 4),
            atr=round(atr, 2),
            capital=self.capital,
            notes=notes,
        )

    def calculate_portfolio(
        self,
        signals: dict,
        features_dict: dict[str, pd.DataFrame],
    ) -> dict[str, PositionRecommendation]:
        """
        Calculate position sizes for all signals at once.
        Respects max concurrent positions AND total capital.

        Priority: highest confidence trades get capital first.
        """
        recommendations = {}
        open_positions  = 0
        capital_used    = 0.0

        # Sort by confidence — highest first
        sorted_tickers = sorted(
            signals.keys(),
            key=lambda t: signals[t].confidence,
            reverse=True,
        )

        for ticker in sorted_tickers:
            sig = signals[ticker]
            df  = features_dict.get(ticker)

            if df is None:
                recommendations[ticker] = self._no_trade(
                    ticker, sig.signal, sig.confidence,
                    "No feature data available"
                )
                continue

            # Calculate remaining capital after higher-priority trades
            remaining_capital = self.capital - capital_used

            if remaining_capital < self.min_trade_value:
                recommendations[ticker] = self._no_trade(
                    ticker, sig.signal, sig.confidence,
                    "Capital fully deployed in higher-confidence trades"
                )
                continue

            # Temporarily override capital for this trade's sizing
            original_capital = self.capital
            self.capital     = remaining_capital

            rec = self.calculate(
                ticker=ticker,
                signal_str=sig.signal,
                confidence=sig.confidence,
                features_df=df,
                open_positions=open_positions,
            )

            # Restore full capital for reporting
            self.capital = original_capital
            rec.capital  = original_capital

            if rec.direction != "NO TRADE":
                # Recalculate risk % against full capital
                rec.risk_pct_capital = round(
                    rec.max_risk_dollars / original_capital, 4
                )
                capital_used   += rec.position_value
                open_positions += 1

            recommendations[ticker] = rec

        return recommendations

    def _no_trade(
        self,
        ticker: str,
        signal: str,
        confidence: float,
        reason: str,
    ) -> PositionRecommendation:
        return PositionRecommendation(
            ticker=ticker, signal=signal,
            direction="NO TRADE", confidence=confidence,
            entry_price=0.0, stop_loss=0.0,
            target_1=0.0, target_2=0.0,
            shares=0.0, position_value=0.0,
            max_risk_dollars=0.0, risk_pct_capital=0.0,
            atr=0.0, capital=self.capital,
            notes=reason,
        )


def run_daily_recommendations(
    capital: float = 500.0,
    allow_fractional: bool = True,
) -> None:
    """
    Full daily workflow:
    1. Load signals from ensemble
    2. Load feature data
    3. Calculate position sizes
    4. Print actionable recommendations
    """
    from src.ensemble.signal_engine import SignalEngine
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    console = Console()

    console.print(Panel(
        f"[bold]Daily Trading Recommendations[/bold]\n"
        f"Capital: ${capital:,.2f} | "
        f"Max risk per trade: 2% = ${capital * 0.02:.2f}\n"
        f"Max positions: 3 | Max per position: 30% | "
        f"Fractional: {allow_fractional}",
        style="blue"
    ))

    # Load models and generate signals
    console.print("\n[dim]Loading models...[/dim]")
    engine = SignalEngine()
    engine.load_models(settings.all_tickers)

    console.print("[dim]Generating signals...[/dim]")
    signals = engine.generate_signals(settings.all_tickers)

    # Load feature data for ATR
    features_dict = {}
    for ticker in settings.all_tickers:
        path = (
            settings.data_processed_path /
            f"{ticker}_features_with_sentiment.parquet"
        )
        if path.exists():
            features_dict[ticker] = pd.read_parquet(path)

    # Calculate positions
    calc = PositionCalculator(
        capital=capital,
        allow_fractional=allow_fractional,
    )
    recs = calc.calculate_portfolio(signals, features_dict)

    # ── Summary table ──────────────────────────────────────────
    table = Table(
        title="Today's Actionable Trades",
        show_header=True,
        header_style="bold",
    )
    table.add_column("Ticker",    style="cyan", width=10)
    table.add_column("Signal",    width=10)
    table.add_column("Entry",     width=9)
    table.add_column("Stop",      width=9)
    table.add_column("Target",    width=9)
    table.add_column("Shares",    width=10)
    table.add_column("Value",     width=10)
    table.add_column("Risk $",    width=9)
    table.add_column("Conf%",     width=8)

    active_trades = 0

    for ticker in settings.all_tickers:
        rec = recs.get(ticker)
        sig = signals.get(ticker)
        if not rec or not sig:
            continue

        if rec.direction == "NO TRADE":
            continue

        active_trades += 1
        col = "green" if rec.direction == "LONG" else "red"

        # Format shares display
        if rec.shares < 1:
            shares_disp = f"{rec.shares:.4f}"
        elif rec.shares != int(rec.shares):
            shares_disp = f"{rec.shares:.2f}"
        else:
            shares_disp = str(int(rec.shares))

        table.add_row(
            ticker,
            f"[{col}]{rec.signal}[/{col}]",
            f"${rec.entry_price:.2f}",
            f"${rec.stop_loss:.2f}",
            f"${rec.target_1:.2f}",
            shares_disp,
            f"${rec.position_value:,.2f}",
            f"${rec.max_risk_dollars:.2f}",
            f"{rec.confidence:.0%}",
        )

    if active_trades > 0:
        console.print(f"\n")
        console.print(table)
    else:
        console.print(
            "\n[yellow]No actionable trades today — "
            "all signals are NEUTRAL or confidence too low.[/yellow]"
        )

    # ── Detailed recommendations ──────────────────────────────
    if active_trades > 0:
        console.print("\n[bold]Detailed Recommendations:[/bold]")
        for ticker in settings.all_tickers:
            rec = recs.get(ticker)
            if rec and rec.direction != "NO TRADE":
                console.print(str(rec))

    # ── Tickers we skipped and why ────────────────────────────
    skipped = [
        (t, r) for t, r in recs.items()
        if r.direction == "NO TRADE"
    ]
    if skipped:
        console.print(
            f"\n[dim]Skipped {len(skipped)} tickers (NEUTRAL or "
            f"insufficient capital):[/dim]"
        )
        for ticker, rec in skipped:
            console.print(
                f"  [dim]{ticker:8s} — {rec.notes}[/dim]"
            )

    # ── Capital summary ───────────────────────────────────────
    total_deployed = sum(
        r.position_value
        for r in recs.values()
        if r.direction != "NO TRADE"
    )
    total_risk = sum(
        r.max_risk_dollars
        for r in recs.values()
        if r.direction != "NO TRADE"
    )

    console.print(
        f"\n[bold]Capital Summary:[/bold]\n"
        f"  Total capital:   ${capital:,.2f}\n"
        f"  Deployed:        ${total_deployed:,.2f} "
        f"({total_deployed/capital:.1%})\n"
        f"  Cash reserved:   ${capital - total_deployed:,.2f} "
        f"({(capital - total_deployed)/capital:.1%})\n"
        f"  Total at risk:   ${total_risk:.2f} "
        f"({total_risk/capital:.1%} of capital)\n"
    )


if __name__ == "__main__":
    # Default: $500 with fractional shares (small account)
    run_daily_recommendations(capital=500.0, allow_fractional=True)