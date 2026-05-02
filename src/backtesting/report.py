"""
src/backtesting/report.py
──────────────────────────
Runs walk-forward backtests for all tickers using
3-model ensemble (XGBoost + LSTM + CNN) with smart stops
and resistance-based take-profit.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

from config.settings import settings
from src.models.xgboost_model import XGBoostModel
from src.models.lstm_model import LSTMModel
from src.models.cnn_model import CNNModel
from src.models.chart_generator import ChartGenerator
from src.models.base_model import MODELS_DIR
from src.ingestion.yfinance_fetcher import YFinanceFetcher
from src.backtesting.backtester import Backtester
from src.utils.logger import log

console = Console()

RESULTS_FILE = MODELS_DIR / "ensemble_results.json"


def _load_model_weights(ticker: str) -> dict:
    if not RESULTS_FILE.exists():
        return {"xgboost": 0.34, "lstm": 0.33, "cnn": 0.33}

    try:
        with open(RESULTS_FILE) as f:
            results = json.load(f)
    except Exception:
        return {"xgboost": 0.34, "lstm": 0.33, "cnn": 0.33}

    if ticker not in results:
        return {"xgboost": 0.34, "lstm": 0.33, "cnn": 0.33}

    r = results[ticker]
    xgb_auc  = r.get("xgboost", {}).get("auc_roc", 0.5)
    lstm_auc = r.get("lstm",    {}).get("auc_roc", 0.5)
    cnn_auc  = r.get("cnn",     {}).get("auc_roc", 0.5)

    total = xgb_auc + lstm_auc + cnn_auc
    if total > 0:
        return {
            "xgboost": xgb_auc  / total,
            "lstm":    lstm_auc / total,
            "cnn":     cnn_auc  / total,
        }
    return {"xgboost": 0.34, "lstm": 0.33, "cnn": 0.33}


def generate_historical_probs(
    ticker: str,
    df: pd.DataFrame,
    target_days: int = 5,
) -> tuple[pd.Series, pd.DataFrame]:
    """Generate per-day probabilities from XGBoost + LSTM + CNN."""
    exclude = {
        # Price columns
        "open", "high", "low", "close",
        "volume", "ticker", "target",
        # Raw moving averages
        "sma_10", "sma_20", "sma_50", "sma_200",
        "ema_9", "ema_21", "ema_55",
        # Raw volume profile levels
        "vp_poc", "vp_val_high", "vp_val_low",
        # Raw aggregates
        "obv_ema", "force_index_ema", "vol_sma_20",
    }
    feat_cols = [
        c for c in df.columns
        if c not in exclude
        and df[c].dtype in [
            np.float64, np.float32, np.int64, np.int32
        ]
    ]

    df = df.dropna(subset=feat_cols).copy()

    future_return = (
        df["close"].shift(-target_days) / df["close"] - 1
    )
    df["target"] = (future_return > 0).astype(int)
    df = df.dropna(subset=["target"])

    n = len(df)
    train_end = int(n * 0.6)

    if train_end < 100:
        log.warning(f"[{ticker}] Not enough data for backtest")
        return pd.Series(dtype=float), pd.DataFrame()

    train_df = df.iloc[:train_end]
    test_df  = df.iloc[train_end:]

    X_train = train_df[feat_cols].values
    y_train = train_df["target"].values
    X_test  = test_df[feat_cols].values

    # XGBoost
    from sklearn.preprocessing import StandardScaler
    scaler    = StandardScaler()
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

    lstm = LSTMModel(ticker, target_days=target_days)
    lstm.feature_cols = feat_cols
    lstm.scaler = scaler

    X_train_seq, y_train_seq = [], []
    for i in range(sequence_len, train_end):
        X_train_seq.append(all_scaled[i - sequence_len:i])
        y_train_seq.append(targets[i])

    lstm_probs_dict = {}
    if len(X_train_seq) > 0:
        lstm.train(
            np.array(X_train_seq, dtype=np.float32),
            np.array(y_train_seq, dtype=np.float32),
        )

        X_seq, idx_seq = [], []
        for i in range(sequence_len, len(all_scaled)):
            if i >= train_end:
                X_seq.append(all_scaled[i - sequence_len:i])
                idx_seq.append(df.index[i])

        if len(X_seq) > 0:
            lstm_prob_vals = lstm.predict_proba(
                np.array(X_seq, dtype=np.float32)
            )
            for idx, prob in zip(idx_seq, lstm_prob_vals):
                lstm_probs_dict[idx] = float(prob)

    # CNN
    cnn_probs_dict = {}
    try:
        fetcher = YFinanceFetcher()
        raw_df  = fetcher.fetch_daily(ticker, use_cache=True)

        gen = ChartGenerator(window_days=30)
        images, labels, dates, patterns = gen.generate_dataset(
            ticker, raw_df, target_days=target_days,
            use_cache=True,
        )

        if len(images) > 0:
            n_charts = len(images)
            chart_split = int(n_charts * 0.6)

            cnn = CNNModel(
                ticker, target_days=target_days,
                max_epochs=15, patience=5,
            )
            cnn.train_on_images(
                images[:chart_split],
                labels[:chart_split],
                patterns[:chart_split],
            )

            for j in range(chart_split, n_charts):
                prob = cnn.predict_proba_image(images[j])
                cnn_probs_dict[dates[j]] = float(prob)

    except Exception as e:
        log.warning(f"[{ticker}] CNN backtest skipped: {e}")

    # Ensemble
    weights = _load_model_weights(ticker)

    test_index = test_df.index
    rows = []
    ensemble_probs = []

    for i, idx in enumerate(test_index):
        xgb_p  = float(xgb_probs[i]) if i < len(xgb_probs) else 0.5
        lstm_p = lstm_probs_dict.get(idx, xgb_p)
        cnn_p  = cnn_probs_dict.get(idx, 0.5)

        has_xgb  = True
        has_lstm = idx in lstm_probs_dict
        has_cnn  = idx in cnn_probs_dict

        w_xgb  = weights["xgboost"] if has_xgb  else 0.0
        w_lstm = weights["lstm"]    if has_lstm else 0.0
        w_cnn  = weights["cnn"]     if has_cnn  else 0.0

        total_w = w_xgb + w_lstm + w_cnn
        if total_w > 0:
            w_xgb  /= total_w
            w_lstm /= total_w
            w_cnn  /= total_w

        ens_p = w_xgb * xgb_p + w_lstm * lstm_p + w_cnn * cnn_p
        ensemble_probs.append(ens_p)

        rows.append({
            "xgb":  xgb_p,
            "lstm": lstm_p,
            "cnn":  cnn_p,
        })

    ensemble = pd.Series(ensemble_probs, index=test_index)
    per_model = pd.DataFrame(rows, index=test_index)

    return ensemble, per_model


def run_backtest_report(
    tickers: list[str] = None,
    initial_capital: float = 100_000.0,
    target_days: int = 5,
    bullish_threshold: float = 0.55,
    bearish_threshold: float = 0.45,
    require_agreement: bool = True,
) -> dict:
    tickers = tickers or settings.all_tickers
    backtester = Backtester(
        initial_capital=initial_capital,
        target_days=target_days,
        bullish_threshold=bullish_threshold,
        bearish_threshold=bearish_threshold,
    )

    all_results = {}

    console.print(
        f"\n[bold cyan]Phase 5 — 3-Model Backtest "
        f"(Smart Stops + Take-Profit at Resistance)[/bold cyan]\n"
        f"Tickers: {len(tickers)} | "
        f"Capital: ${initial_capital:,.0f} | "
        f"Horizon: {target_days} days\n"
        f"Thresholds: bull>{bullish_threshold} "
        f"bear<{bearish_threshold}\n"
        f"Require agreement: {require_agreement}\n"
    )

    for ticker in tickers:
        console.print(f"[dim]Backtesting {ticker}...[/dim]")

        data_path = (
            settings.data_processed_path /
            f"{ticker}_features_with_sentiment.parquet"
        )
        if not data_path.exists():
            console.print(f"  [red]✗[/red] {ticker}: no data file")
            continue

        df = pd.read_parquet(data_path)

        try:
            probs, per_model = generate_historical_probs(
                ticker, df, target_days
            )

            if probs.empty:
                continue

            result = backtester.run(
                ticker, probs, df,
                per_model=per_model if require_agreement else None,
            )
            all_results[ticker] = result

        except Exception as e:
            console.print(f"  [red]✗[/red] {ticker}: {e}")
            log.error(f"[{ticker}] Backtest failed: {e}")

    # Summary table
    table = Table(
        title="Backtest Results — Smart Stops + TP at Resistance",
        show_header=True,
        header_style="bold",
    )
    table.add_column("Ticker",  style="cyan", width=10)
    table.add_column("Return",  width=8)
    table.add_column("Alpha",   width=8)
    table.add_column("Sharpe",  width=6)
    table.add_column("MaxDD",   width=7)
    table.add_column("Trades",  width=6)
    table.add_column("Win%",    width=5)
    table.add_column("Trnd",    width=4)
    table.add_column("Mean",    width=4)
    table.add_column("TPR",     width=4)
    table.add_column("Rev",     width=4)
    table.add_column("Trail",   width=5)
    table.add_column("Stop",    width=4)
    table.add_column("TimeL",   width=5)

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

        n_trnd    = m.get("n_trending", 0)
        n_mean    = m.get("n_mean_rev", 0)
        n_tp_res  = m.get("n_tp_resistance", 0)
        n_rev     = m.get("n_reversal", 0)
        n_trail   = (
            m.get("n_sweep_rev", 0) +
            m.get("n_strong_rev", 0)
        )
        n_stopped = m.get("n_stopped", 0)
        n_time_l  = m.get("n_time_loss", 0)

        table.add_row(
            ticker,
            f"[{ret_col}]{ret:.1%}[/{ret_col}]",
            f"[{alpha_col}]{alpha:+.1%}[/{alpha_col}]",
            f"{m['sharpe']:.2f}",
            f"{m['max_drawdown']:.1%}",
            str(m["n_trades"]),
            f"{m['win_rate']:.0%}",
            f"[blue]{n_trnd}[/blue]",
            f"[magenta]{n_mean}[/magenta]",
            f"[cyan]{n_tp_res}[/cyan]",
            f"[yellow]{n_rev}[/yellow]",
            f"[green]{n_trail}[/green]",
            f"[red]{n_stopped}[/red]",
            f"[yellow]{n_time_l}[/yellow]",
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
        "\n[dim]"
        "Trnd = trending trades | Mean = mean-reverting trades | "
        "TPR = take-profit at resistance | "
        "Rev = bearish/bullish reversal exit | "
        "Trail = trailing stop hit (sweep/strong reversal) | "
        "Stop = initial stop hit | "
        "TimeL = time-out at loss"
        "[/dim]"
    )

    return all_results


if __name__ == "__main__":
    run_backtest_report()