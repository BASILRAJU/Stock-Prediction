"""
rebuild_results.py
───────────────────
Re-evaluates all trained models and rebuilds ensemble_results.json
with metrics for XGBoost + LSTM + CNN per ticker.

Doesn't retrain anything — just runs predictions on test sets
and saves the metrics. Takes ~30 seconds total.
"""

import json
import pandas as pd

from config.settings import settings
from src.models.xgboost_model import XGBoostModel
from src.models.lstm_model import LSTMModel
from src.models.cnn_model import CNNModel
from src.models.chart_generator import ChartGenerator
from src.models.base_model import MODELS_DIR
from src.ingestion.yfinance_fetcher import YFinanceFetcher
from src.utils.logger import log


def main():
    results = {}
    fetcher = YFinanceFetcher()

    for ticker in settings.all_tickers:
        log.info(f"=== {ticker} ===")
        ticker_results = {}

        feature_path = (
            settings.data_processed_path /
            f"{ticker}_features_with_sentiment.parquet"
        )
        if not feature_path.exists():
            log.warning(f"[{ticker}] No feature file — skipping")
            continue

        df = pd.read_parquet(feature_path)

        # ── XGBoost ──
        xgb_path = MODELS_DIR / f"xgboost_{ticker}.joblib"
        if xgb_path.exists():
            try:
                xgb = XGBoostModel(ticker)
                xgb.load(xgb_path)
                X_train, X_test, y_train, y_test = xgb.prepare_data(df)
                metrics = xgb.evaluate(X_test, y_test)
                ticker_results["xgboost"] = metrics
                log.info(
                    f"[{ticker}] XGBoost: "
                    f"acc={metrics['accuracy']:.1%} "
                    f"auc={metrics['auc_roc']:.3f}"
                )
            except Exception as e:
                log.error(f"[{ticker}] XGBoost eval failed: {e}")

        # ── LSTM ──
        lstm_path = MODELS_DIR / f"lstm_{ticker}.pt"
        if lstm_path.exists():
            try:
                lstm = LSTMModel(ticker)
                lstm.load(lstm_path)
                X_train, X_test, y_train, y_test = lstm.prepare_data(df)
                metrics = lstm.evaluate(X_test, y_test)
                ticker_results["lstm"] = metrics
                log.info(
                    f"[{ticker}] LSTM: "
                    f"acc={metrics['accuracy']:.1%} "
                    f"auc={metrics['auc_roc']:.3f}"
                )
            except Exception as e:
                log.error(f"[{ticker}] LSTM eval failed: {e}")

        # ── CNN ──
        cnn_path = MODELS_DIR / f"cnn_{ticker}.pt"
        if cnn_path.exists():
            try:
                # CNN needs chart images
                raw_df = fetcher.fetch_daily(ticker, use_cache=True)
                gen = ChartGenerator(window_days=30)
                images, labels, dates = gen.generate_dataset(
                    ticker, raw_df, target_days=5, use_cache=True,
                )

                cnn = CNNModel(ticker)
                cnn.load(cnn_path)
                metrics = cnn.evaluate_on_images(images, labels, dates)
                ticker_results["cnn"] = metrics
                log.info(
                    f"[{ticker}] CNN: "
                    f"acc={metrics['accuracy']:.1%} "
                    f"auc={metrics['auc_roc']:.3f}"
                )
            except Exception as e:
                log.error(f"[{ticker}] CNN eval failed: {e}")

        results[ticker] = ticker_results

    # Save merged results
    results_file = MODELS_DIR / "ensemble_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    log.info(f"\n✓ Saved to {results_file}")

    # Summary
    print("\n=== SUMMARY ===")
    for ticker, r in results.items():
        models = list(r.keys())
        print(f"{ticker:10s} → {models}")


if __name__ == "__main__":
    main()