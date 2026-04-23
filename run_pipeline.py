"""
run_pipeline.py
────────────────
Phase 1 + Phase 2 combined pipeline runner.

Runs every day to fetch fresh data, build features,
score sentiment, and save enriched feature files.

Usage:
    python run_pipeline.py                    # full pipeline
    python run_pipeline.py --skip-sentiment   # Phase 1 only
    python run_pipeline.py --skip-macro       # skip Alpha Vantage
"""

import sys
import time

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from config.settings import settings
from src.ingestion.yfinance_fetcher import YFinanceFetcher
from src.features.technical_indicators import build_features
from src.sentiment.news_fetcher import NewsFetcher
from src.sentiment.finbert_scorer import FinBERTScorer
from src.sentiment.sentiment_merger import SentimentMerger
from src.utils.logger import log

console = Console()


def run_pipeline(
    tickers: list[str] = None,
    period: str = "2y",
    include_sentiment: bool = True,
    use_cache: bool = True,
):
    start_time = time.time()
    tickers = tickers or settings.all_tickers

    # Parse command line args
    if "--skip-sentiment" in sys.argv:
        include_sentiment = False

    console.print(Panel(
        f"[bold]AI Stock Prediction — Phase 1 + 2 Pipeline[/bold]\n"
        f"Tickers:   {', '.join(tickers)}\n"
        f"Period:    {period} | "
        f"Sentiment: {include_sentiment} | "
        f"Cache:     {use_cache}",
        style="blue"
    ))

    # ── Results tracking ──────────────────────────────────────
    success  = []
    failed   = []
    features = {}
    enriched = {}

    # ─────────────────────────────────────────────────────────
    # STEP 1: Download price data
    # ─────────────────────────────────────────────────────────
    console.print(
        "\n[bold cyan]Step 1/5: Downloading price data...[/bold cyan]"
    )
    yf = YFinanceFetcher()

    for ticker in tickers:
        try:
            df = yf.fetch_daily(
                ticker, period=period, use_cache=use_cache
            )
            features[ticker] = df
            console.print(
                f"  [green]✓[/green] {ticker}: {len(df)} bars"
            )
        except Exception as e:
            console.print(f"  [red]✗[/red] {ticker}: {e}")
            failed.append(ticker)

    # ─────────────────────────────────────────────────────────
    # STEP 2: Build technical features
    # ─────────────────────────────────────────────────────────
    console.print(
        "\n[bold cyan]Step 2/5: Building technical features...[/bold cyan]"
    )
    processed = {}

    for ticker, df in features.items():
        try:
            feat_df = build_features(df, ticker=ticker, drop_nulls=True)
            out_path = (
                settings.data_processed_path /
                f"{ticker}_features.parquet"
            )
            feat_df.to_parquet(out_path)
            processed[ticker] = feat_df
            success.append(ticker)
            console.print(
                f"  [green]✓[/green] {ticker}: "
                f"{feat_df.shape[0]} rows × {feat_df.shape[1]} features"
            )
        except Exception as e:
            console.print(f"  [red]✗[/red] {ticker}: {e}")
            failed.append(ticker)

    # ─────────────────────────────────────────────────────────
    # STEP 3: Fetch news headlines
    # ─────────────────────────────────────────────────────────
    all_articles: dict = {}

    if include_sentiment:
        console.print(
            "\n[bold cyan]Step 3/5: Fetching news headlines...[/bold cyan]"
        )
        try:
            news = NewsFetcher()
            for ticker in success:
                try:
                    articles = news.fetch_ticker(
                        ticker, days=7, use_cache=use_cache
                    )
                    all_articles[ticker] = articles
                    console.print(
                        f"  [green]✓[/green] {ticker}: "
                        f"{len(articles)} headlines"
                    )
                except Exception as e:
                    console.print(
                        f"  [yellow]![/yellow] {ticker}: {e}"
                    )
                    all_articles[ticker] = []
        except Exception as e:
            console.print(f"  [red]✗[/red] News fetcher error: {e}")
            include_sentiment = False
    else:
        console.print("\n[dim]Step 3/5: Sentiment skipped[/dim]")

    # ─────────────────────────────────────────────────────────
    # STEP 4: Score sentiment with FinBERT
    # ─────────────────────────────────────────────────────────
    all_scores: dict = {}

    if include_sentiment and all_articles:
        console.print(
            "\n[bold cyan]Step 4/5: Scoring sentiment with FinBERT...[/bold cyan]"
        )
        try:
            scorer = FinBERTScorer()
            for ticker in success:
                articles = all_articles.get(ticker, [])
                if not articles:
                    all_scores[ticker] = []
                    continue
                try:
                    scored = scorer.score_ticker(
                        ticker,
                        articles,
                        use_cache=use_cache,
                        filter_neutral=True,
                    )
                    all_scores[ticker] = scored
                    bullish = sum(
                        1 for s in scored if s["signal"] == 1
                    )
                    bearish = sum(
                        1 for s in scored if s["signal"] == -1
                    )
                    console.print(
                        f"  [green]✓[/green] {ticker}: "
                        f"{bullish} bullish, {bearish} bearish signals"
                    )
                except Exception as e:
                    console.print(
                        f"  [yellow]![/yellow] {ticker}: {e}"
                    )
                    all_scores[ticker] = []
        except Exception as e:
            console.print(f"  [red]✗[/red] FinBERT error: {e}")
    else:
        console.print("\n[dim]Step 4/5: Scoring skipped[/dim]")

    # ─────────────────────────────────────────────────────────
    # STEP 5: Merge sentiment onto features
    # ─────────────────────────────────────────────────────────
    if include_sentiment and all_scores:
        console.print(
            "\n[bold cyan]"
            "Step 5/5: Merging sentiment onto features..."
            "[/bold cyan]"
        )
        merger = SentimentMerger()
        for ticker in success:
            try:
                feat_df  = processed[ticker]
                scored   = all_scores.get(ticker, [])
                enriched_df = merger.merge(feat_df, scored, ticker)
                enriched[ticker] = enriched_df

                out_path = (
                    settings.data_processed_path /
                    f"{ticker}_features_with_sentiment.parquet"
                )
                enriched_df.to_parquet(out_path)
                console.print(
                    f"  [green]✓[/green] {ticker}: "
                    f"{enriched_df.shape[0]} rows × "
                    f"{enriched_df.shape[1]} columns → saved"
                )
            except Exception as e:
                console.print(
                    f"  [red]✗[/red] {ticker} merge failed: {e}"
                )
    else:
        console.print("\n[dim]Step 5/5: Merge skipped[/dim]")

    # ─────────────────────────────────────────────────────────
    # SUMMARY TABLE
    # ─────────────────────────────────────────────────────────
    duration = round(time.time() - start_time, 1)

    table = Table(
        title="Pipeline Results",
        show_header=True,
        header_style="bold",
    )
    table.add_column("Ticker",       style="cyan", width=10)
    table.add_column("Status",       width=12)
    table.add_column("Features",     width=10)
    table.add_column("Sentiment",    width=12)
    table.add_column("Latest Close", width=14)

    for ticker in success:
        feat_df  = processed.get(ticker)
        enr_df   = enriched.get(ticker, feat_df)
        scored   = all_scores.get(ticker, [])

        if feat_df is None:
            continue

        latest = (
            f"${feat_df['close'].iloc[-1]:.2f}"
            if "close" in feat_df.columns else "—"
        )
        bullish = sum(1 for s in scored if s.get("signal") == 1)
        bearish = sum(1 for s in scored if s.get("signal") == -1)
        sent_str = (
            f"[green]+{bullish}[/green]/[red]-{bearish}[/red]"
            if scored else "[dim]n/a[/dim]"
        )
        feat_count = (
            str(enr_df.shape[1]) if enr_df is not None
            else str(feat_df.shape[1])
        )

        table.add_row(
            ticker,
            "[green]✓ Success[/green]",
            feat_count,
            sent_str,
            latest,
        )

    for ticker in set(failed):
        table.add_row(
            ticker,
            "[red]✗ Failed[/red]",
            "—", "—", "—",
        )

    console.print(f"\n")
    console.print(table)
    console.print(
        f"\n[bold]Done in {duration}s[/bold] | "
        f"Success: {len(success)}/{len(tickers)} "
        f"({len(success)/len(tickers)*100:.0f}%)"
    )
    console.print(
        f"[dim]Enriched features saved to: "
        f"{settings.data_processed_path}[/dim]"
    )

    return enriched if enriched else processed


if __name__ == "__main__":
    run_pipeline(
        use_cache=True,
        include_sentiment=True,
    )