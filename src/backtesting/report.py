"""
src/backtesting/report.py
──────────────────────────
Runs backtests for all tickers and prints a summary report.

Generates model probabilities by running models on historical
data in a walk-forward manner, then passes them to the
Backtester to simulate trading.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

from config.settings import settings
from src.models.xgboost_model import XGBoostModel
from src.models.lstm_model import LSTMModel
from src.models.base_model import MODELS_DIR
from src.backtesting.backtester import Backtester
from src.utils.logger import log

console = Console()


def generate_historical_probs(
    ticker: str,
    df: pd.DataFrame,
    target_days: int = 5,
) -> pd.Series:
    """
    Generate historical ensemble probabilities for backtesting.

    Trains models on first 60% of data, then generates
    predictions for the remaining 40% walk-forward.
    This simulates what signals would have been generated
    in real-time without lookahead bias.
    """
    exclude = {
        "open", "high", "low", "close",
        "volume", "ticker", "target",
    }
    feat_cols = [
        c for c in df.columns
        if c not in exclude
        and df[c].dtype in [
            np.float64, np.float32, np.int64, np.int32
        ]
    ]

    df = df.dropna(subset=feat_cols).copy()

    # Build target
    future_return = (
        df["close"].shift(-target_days) / df["close"] - 1
    )
    df["target"] = (future_return > 0).astype(int)
    df = df.dropna(subset=["target"])

    n = len(df)
    train_end = int(n * 0.6)

    if train_end < 100:
        log.warning(f"[{ticker}] Not enough data for backtest")
        return pd.Series(dtype=float)

    # Train on first 60%
    train_df = df.iloc[:train_end]
    test_df  = df.iloc[train_end:]

    X_train = train_df[feat_cols].values
    y_train = train_df["target"].values
    X_test  = test_df[feat_cols].values

    # XGBoost
    from sklearn.preprocessing import StandardScaler
    scaler  = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    xgb = XGBoostModel(ticker, target_days=target_days)
    xgb.feature_cols = feat_cols
    xgb.scaler = scaler
    xgb.train(X_train_s, y_train)
    xgb_probs = xgb.predict_proba(X_test_s)

    # LSTM
    sequence_len = 60
    all_scaled   = scaler.transform(df[feat_cols].values)
    targets      = df["target"].values

    lstm_probs_dict = {}
    lstm = LSTMModel(ticker, target_days=target_days)
    lstm.feature_cols = feat_cols
    lstm.scaler = scaler

    # Build sequences for test period only
    X_seq, y_seq, idx_seq = [], [], []
    for i in range(sequence_len, len(all_scaled)):
        if i >= train_end:
            X_seq.append(
                all_scaled[i - sequence_len:i]
            )
            y_seq.append(targets[i])
            idx_seq.append(df.index[i])

    if len(X_seq) > 0:
        X_seq_arr = np.array(X_seq, dtype=np.float32)
        y_seq_arr = np.array(y_seq, dtype=np.float32)

        # Train LSTM on training sequences
        X_train_seq, y_train_seq = [], []
        for i in range(sequence_len, train_end):
            X_train_seq.append(
                all_scaled[i - sequence_len:i]
            )
            y_train_seq.append(targets[i])

        if len(X_train_seq) > 0:
            lstm.train(
                np.array(X_train_seq, dtype=np.float32),
                np.array(y_train_seq, dtype=np.float32),
            )
            lstm_prob_vals = lstm.predict_proba(X_seq_arr)
            for idx, prob in zip(idx_seq, lstm_prob_vals):
                lstm_probs_dict[idx] = float(prob)

    # Combine XGBoost + LSTM into ensemble
    test_index = test_df.index
    ensemble_probs = []

    for i, idx in enumerate(test_index):
        xgb_p  = float(xgb_probs[i]) if i < len(xgb_probs) else 0.5
        lstm_p = lstm_probs_dict.get(idx, xgb_p)
        # Equal weight ensemble for backtest
        ens_p  = 0.5 * xgb_p + 0.5 * lstm_p
        ensemble_probs.append(ens_p)

    return pd.Series(ensemble_probs, index=test_index)


def run_backtest_report(
    tickers: list[str] = None,
    initial_capital: float = 100_000.0,
    target_days: int = 5,
) -> dict:
    """
    Run backtests for all tickers and print summary report.
    """
    tickers = tickers or settings.all_tickers
    backtester = Backtester(
        initial_capital=initial_capital,
        target_days=target_days,
    )

    all_results = {}

    console.print(
        f"\n[bold cyan]Phase 5 — Backtest Report[/bold cyan]\n"
        f"Tickers: {len(tickers)} | "
        f"Capital: ${initial_capital:,.0f} | "
        f"Horizon: {target_days} days\n"
    )

    for ticker in tickers:
        console.print(f"[dim]Backtesting {ticker}...[/dim]")

        data_path = (
            settings.data_processed_path /
            f"{ticker}_features_with_sentiment.parquet"
        )
        if not data_path.exists():
            console.print(
                f"  [red]✗[/red] {ticker}: no data file"
            )
            continue

        df = pd.read_parquet(data_path)

        try:
            # Generate historical probabilities
            probs = generate_historical_probs(
                ticker, df, target_days
            )

            if probs.empty:
                continue

            # Run backtest
            result = backtester.run(ticker, probs, df)
            all_results[ticker] = result

        except Exception as e:
            console.print(
                f"  [red]✗[/red] {ticker}: {e}"
            )
            log.error(f"[{ticker}] Backtest failed: {e}")

    # ── Summary table ─────────────────────────────────────────
    table = Table(
        title="Backtest Results (last 40% of data)",
        show_header=True,
        header_style="bold",
    )
    table.add_column("Ticker",   style="cyan", width=10)
    table.add_column("Return",   width=10)
    table.add_column("BaH",      width=10)
    table.add_column("Alpha",    width=10)
    table.add_column("Sharpe",   width=8)
    table.add_column("Max DD",   width=10)
    table.add_column("Trades",   width=8)
    table.add_column("Win%",     width=8)

    total_alpha = 0.0
    count = 0

    for ticker, result in all_results.items():
        m = result["metrics"]
        ret   = m["total_return"]
        bah   = m["bah_return"]
        alpha = m["alpha"]
        total_alpha += alpha
        count += 1

        ret_col   = "green" if ret   > 0 else "red"
        alpha_col = "green" if alpha > 0 else "red"

        table.add_row(
            ticker,
            f"[{ret_col}]{ret:.1%}[/{ret_col}]",
            f"{bah:.1%}",
            f"[{alpha_col}]{alpha:+.1%}[/{alpha_col}]",
            f"{m['sharpe']:.2f}",
            f"{m['max_drawdown']:.1%}",
            str(m["n_trades"]),
            f"{m['win_rate']:.0%}",
        )

    console.print(f"\n")
    console.print(table)

    avg_alpha = total_alpha / count if count > 0 else 0
    console.print(
        f"\n[bold]Average Alpha vs Buy-and-Hold: "
        f"{'[green]' if avg_alpha > 0 else '[red]'}"
        f"{avg_alpha:+.1%}"
        f"{'[/green]' if avg_alpha > 0 else '[/red]'}[/bold]"
    )
    console.print(
        "[dim]Alpha = strategy return minus buy-and-hold return[/dim]"
    )

    return all_results


if __name__ == "__main__":
    run_backtest_report()