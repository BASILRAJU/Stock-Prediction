"""
src/ensemble/ensemble_trainer.py
──────────────────────────────────
Trains XGBoost + LSTM + CNN for all tickers and saves
their results to disk for the signal engine to use.

Design:
  Each ticker gets its own trio of models trained
  independently on that ticker's feature history.
  This is ticker-specific training — AAPL's models
  learn AAPL patterns, not general market patterns.

Three model types:
  XGBoost — reads 117 numeric features (tabular patterns)
  LSTM    — reads 60-day sequences (temporal patterns)
  CNN     — reads 30-day chart images (visual patterns)

Output per ticker:
  data/models/xgboost_{ticker}.joblib
  data/models/lstm_{ticker}.pt
  data/models/cnn_{ticker}.pt
  data/models/ensemble_results.json   ← accuracy scores

Research basis:
  Dynamic ensemble weighting by Sharpe/AUC
  (TradingAgents arXiv:2412.20138, 2024).
  Models weighted by recent out-of-sample performance,
  not fixed equal weights.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

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
from src.utils.logger import log

console = Console()

RESULTS_FILE = MODELS_DIR / "ensemble_results.json"


def train_all_tickers(
    tickers: list[str] = None,
    target_days: int = 5,
    skip_existing: bool = False,
    train_cnn: bool = True,
) -> dict:
    """
    Train XGBoost + LSTM + CNN for every ticker.

    Args:
        tickers:       List of tickers (defaults to all)
        target_days:   Prediction horizon in trading days
        skip_existing: Skip tickers that already have saved models
        train_cnn:     Whether to train CNN (set False for fast retraining
                       without chart-based model)

    Returns:
        Dict of {ticker: {xgboost: metrics, lstm: metrics, cnn: metrics}}
    """
    tickers      = tickers or settings.all_tickers
    all_results  = {}

    # Load existing results so we don't overwrite (allows incremental updates)
    if RESULTS_FILE.exists():
        try:
            with open(RESULTS_FILE) as f:
                all_results = json.load(f)
        except Exception:
            all_results = {}

    console.print(
        f"\n[bold cyan]Phase 4 — Ensemble Trainer[/bold cyan]\n"
        f"Training XGBoost + LSTM"
        f"{' + CNN' if train_cnn else ''}"
        f" for {len(tickers)} tickers\n"
        f"Prediction horizon: {target_days} trading days\n"
        f"Models saved to: {MODELS_DIR}\n"
    )

    total_start = time.time()
    fetcher     = YFinanceFetcher() if train_cnn else None

    for i, ticker in enumerate(tickers, 1):
        console.print(
            f"\n[bold]({i}/{len(tickers)}) {ticker}[/bold]"
            f" ─────────────────────────────"
        )

        # Load feature data
        data_path = (
            settings.data_processed_path /
            f"{ticker}_features_with_sentiment.parquet"
        )

        if not data_path.exists():
            console.print(
                f"  [red]✗[/red] No data found: {data_path.name}"
            )
            console.print(
                f"  [dim]Run python run_pipeline.py first[/dim]"
            )
            continue

        df = pd.read_parquet(data_path)
        console.print(
            f"  [dim]Loaded {len(df)} rows × "
            f"{df.shape[1]} features[/dim]"
        )

        ticker_results = all_results.get(ticker, {})

        # ── XGBoost ───────────────────────────────────────────
        xgb_path = MODELS_DIR / f"xgboost_{ticker}.joblib"
        if skip_existing and xgb_path.exists():
            console.print(
                f"  [dim]XGBoost: skipping (model exists)[/dim]"
            )
        else:
            try:
                t0  = time.time()
                xgb = XGBoostModel(ticker, target_days=target_days)
                m   = xgb.fit_and_evaluate(df)
                xgb.save()
                duration = round(time.time() - t0, 1)

                ticker_results["xgboost"] = m
                status = (
                    "[green]✓[/green]" if m["accuracy"] > 0.52
                    else "[yellow]~[/yellow]"
                )
                console.print(
                    f"  {status} XGBoost: "
                    f"accuracy={m['accuracy']:.1%} | "
                    f"auc={m['auc_roc']:.3f} | "
                    f"{duration}s"
                )
            except Exception as e:
                console.print(
                    f"  [red]✗[/red] XGBoost failed: {e}"
                )
                log.error(f"[{ticker}] XGBoost training failed: {e}")

        # ── LSTM ──────────────────────────────────────────────
        lstm_path = MODELS_DIR / f"lstm_{ticker}.pt"
        if skip_existing and lstm_path.exists():
            console.print(
                f"  [dim]LSTM: skipping (model exists)[/dim]"
            )
        else:
            try:
                t0   = time.time()
                lstm = LSTMModel(ticker, target_days=target_days)
                m    = lstm.fit_and_evaluate(df)
                lstm.save()
                duration = round(time.time() - t0, 1)

                ticker_results["lstm"] = m
                status = (
                    "[green]✓[/green]" if m["accuracy"] > 0.52
                    else "[yellow]~[/yellow]"
                )
                console.print(
                    f"  {status} LSTM:    "
                    f"accuracy={m['accuracy']:.1%} | "
                    f"auc={m['auc_roc']:.3f} | "
                    f"{duration}s"
                )
            except Exception as e:
                console.print(
                    f"  [red]✗[/red] LSTM failed: {e}"
                )
                log.error(f"[{ticker}] LSTM training failed: {e}")

        # ── CNN ───────────────────────────────────────────────
        if train_cnn:
            cnn_path = MODELS_DIR / f"cnn_{ticker}.pt"
            if skip_existing and cnn_path.exists():
                console.print(
                    f"  [dim]CNN: skipping (model exists)[/dim]"
                )
            else:
                try:
                    t0 = time.time()

                    # CNN needs raw OHLCV, not feature DataFrame
                    raw_df = fetcher.fetch_daily(
                        ticker, use_cache=True
                    )

                    # Generate chart images + patterns
                    gen = ChartGenerator(window_days=30)
                    images, labels, dates, patterns = gen.generate_dataset(
                        ticker, raw_df, target_days=target_days,
                        use_cache=True,
                    )

                    if len(images) == 0:
                        raise ValueError("No chart images generated")

                    console.print(
                        f"  [dim]Generated {len(images)} chart "
                        f"images ({labels.sum()}/{len(labels)} "
                        f"bullish)[/dim]"
                    )

                    # Train CNN with pattern labels (multi-task)
                    cnn = CNNModel(
                        ticker, target_days=target_days,
                        max_epochs=30, patience=8,
                    )
                    cnn.train_on_images(images, labels, patterns)

                    # Evaluate + save
                    m = cnn.evaluate_on_images(
                        images, labels, dates
                    )
                    cnn.save()
                    duration = round(time.time() - t0, 1)

                    ticker_results["cnn"] = m
                    status = (
                        "[green]✓[/green]" if m["accuracy"] > 0.52
                        else "[yellow]~[/yellow]"
                    )
                    console.print(
                        f"  {status} CNN:     "
                        f"accuracy={m['accuracy']:.1%} | "
                        f"auc={m['auc_roc']:.3f} | "
                        f"{duration}s"
                    )

                except Exception as e:
                    console.print(
                        f"  [red]✗[/red] CNN failed: {e}"
                    )
                    log.error(
                        f"[{ticker}] CNN training failed: {e}"
                    )

        all_results[ticker] = ticker_results

        # Save incrementally after each ticker (resume-safe)
        with open(RESULTS_FILE, "w") as f:
            json.dump(all_results, f, indent=2)

    # ── Summary table ─────────────────────────────────────────
    total_duration = round(time.time() - total_start, 1)

    table = Table(
        title="Ensemble Training Results",
        show_header=True,
        header_style="bold",
    )
    table.add_column("Ticker",  style="cyan", width=10)
    table.add_column("XGBoost", width=16)
    table.add_column("LSTM",    width=16)
    table.add_column("CNN",     width=16)
    table.add_column("Best",    width=10)

    for ticker, results in all_results.items():
        if ticker not in tickers:
            continue   # only show tickers we just trained

        xgb_str  = "—"
        lstm_str = "—"
        cnn_str  = "—"
        best_acc = 0.0

        if "xgboost" in results:
            acc = results["xgboost"]["accuracy"]
            auc = results["xgboost"]["auc_roc"]
            col = "green" if acc > 0.52 else "yellow"
            xgb_str  = f"[{col}]{acc:.1%}[/{col}] / {auc:.3f}"
            best_acc = max(best_acc, acc)

        if "lstm" in results:
            acc = results["lstm"]["accuracy"]
            auc = results["lstm"]["auc_roc"]
            col = "green" if acc > 0.52 else "yellow"
            lstm_str = f"[{col}]{acc:.1%}[/{col}] / {auc:.3f}"
            best_acc = max(best_acc, acc)

        if "cnn" in results:
            acc = results["cnn"]["accuracy"]
            auc = results["cnn"]["auc_roc"]
            col = "green" if acc > 0.52 else "yellow"
            cnn_str  = f"[{col}]{acc:.1%}[/{col}] / {auc:.3f}"
            best_acc = max(best_acc, acc)

        best_str = (
            f"[green]{best_acc:.1%}[/green]"
            if best_acc > 0.52
            else f"[yellow]{best_acc:.1%}[/yellow]"
        )

        table.add_row(ticker, xgb_str, lstm_str, cnn_str, best_str)

    console.print(f"\n")
    console.print(table)
    console.print(
        f"\n[bold]Total training time: {total_duration}s[/bold] | "
        f"{total_duration/60:.1f} minutes"
    )
    console.print(
        f"[dim]Results saved to: {RESULTS_FILE}[/dim]"
    )

    return all_results


if __name__ == "__main__":
    train_all_tickers()