"""
run_pipeline.py
────────────────
Phase 1 main runner.

Run this every day to fetch fresh data and rebuild features.

Usage:
    python run_pipeline.py
"""

import time
from datetime import datetime

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from config.settings import settings
from src.ingestion.yfinance_fetcher import YFinanceFetcher
from src.ingestion.alpha_vantage_fetcher import AlphaVantageFetcher
from src.features.technical_indicators import build_features
from src.utils.logger import log

console = Console()


def run_pipeline(
    tickers: list[str] = None,
    period: str = "2y",
    fetch_macro: bool = True,
    fetch_fundamentals: bool = True,
    use_cache: bool = True,
):
    """
    Full Phase 1 pipeline:
    1. Download OHLCV for all tickers
    2. Build 98 technical features per ticker
    3. Fetch macro indicators from Alpha Vantage
    4. Fetch company fundamentals from Alpha Vantage
    5. Save everything to data/processed/
    6. Print summary report
    """
    start_time = time.time()
    tickers = tickers or settings.all_tickers

    console.print(Panel(
        f"[bold]Phase 1 Pipeline — AI Stock Prediction System[/bold]\n"
        f"Tickers: {', '.join(tickers)}\n"
        f"Period:  {period} | "
        f"Macro: {fetch_macro} | "
        f"Fundamentals: {fetch_fundamentals}",
        style="blue"
    ))

    # ── Results tracking ──────────────────────────────────────
    success = []
    failed  = []
    feature_data = {}

    # ─────────────────────────────────────────────────────────
    # STEP 1: Download price data
    # ─────────────────────────────────────────────────────────
    console.print("\n[bold cyan]Step 1/4: Downloading price data...[/bold cyan]")
    yf = YFinanceFetcher()

    for ticker in tickers:
        try:
            df = yf.fetch_daily(ticker, period=period, use_cache=use_cache)
            feature_data[ticker] = df
            console.print(f"  [green]✓[/green] {ticker}: {len(df)} daily bars")
        except Exception as e:
            console.print(f"  [red]✗[/red] {ticker}: {e}")
            failed.append(ticker)
            log.error(f"[{ticker}] Price download failed: {e}")

    # ─────────────────────────────────────────────────────────
    # STEP 2: Build technical features
    # ─────────────────────────────────────────────────────────
    console.print(
        "\n[bold cyan]Step 2/4: Building technical features...[/bold cyan]"
    )
    processed = {}

    for ticker, df in feature_data.items():
        try:
            feat_df = build_features(df, ticker=ticker, drop_nulls=True)

            # Save to disk as Parquet
            out_path = settings.data_processed_path / f"{ticker}_features.parquet"
            feat_df.to_parquet(out_path)

            processed[ticker] = feat_df
            success.append(ticker)
            console.print(
                f"  [green]✓[/green] {ticker}: "
                f"{feat_df.shape[0]} rows × {feat_df.shape[1]} features "
                f"→ saved"
            )
        except Exception as e:
            console.print(f"  [red]✗[/red] {ticker}: {e}")
            failed.append(ticker)
            log.error(f"[{ticker}] Feature build failed: {e}")

    # ─────────────────────────────────────────────────────────
    # STEP 3: Fetch macro indicators
    # ─────────────────────────────────────────────────────────
    macro_wide = None
    if fetch_macro:
        console.print(
            "\n[bold cyan]Step 3/4: Fetching macro indicators "
            "(Alpha Vantage — this takes ~60s on free tier)...[/bold cyan]"
        )
        av = AlphaVantageFetcher()
        try:
            macro_dict = av.fetch_all_macro(use_cache=use_cache)
            macro_wide = av.get_macro_wide(macro_dict)

            # Save macro data
            if not macro_wide.empty:
                macro_path = settings.data_processed_path / "macro_wide.parquet"
                macro_wide.to_parquet(macro_path)
                console.print(
                    f"  [green]✓[/green] Macro data: "
                    f"{macro_wide.shape[0]} periods × "
                    f"{macro_wide.shape[1]} indicators → saved"
                )
        except Exception as e:
            console.print(f"  [red]✗[/red] Macro fetch failed: {e}")
            log.error(f"Macro fetch failed: {e}")
    else:
        console.print("\n[dim]Step 3/4: Macro skipped[/dim]")

    # ─────────────────────────────────────────────────────────
    # STEP 4: Fetch fundamentals (US tickers only)
    # ─────────────────────────────────────────────────────────
    if fetch_fundamentals:
        console.print(
            "\n[bold cyan]Step 4/4: Fetching fundamentals "
            "(US tickers only)...[/bold cyan]"
        )
        av = AlphaVantageFetcher()
        us_tickers = [t for t in success if not t.endswith(".TO")]

        for ticker in us_tickers:
            try:
                snap = av.fetch_fundamentals(
                    ticker, use_cache=use_cache
                )
                console.print(
                    f"  [green]✓[/green] {ticker}: "
                    f"P/E={snap.pe_ratio}, "
                    f"Beta={snap.beta}, "
                    f"Sector={snap.sector}"
                )
            except Exception as e:
                console.print(
                    f"  [yellow]![/yellow] {ticker} fundamentals: {e}"
                )
    else:
        console.print("\n[dim]Step 4/4: Fundamentals skipped[/dim]")

    # ─────────────────────────────────────────────────────────
    # SUMMARY REPORT
    # ─────────────────────────────────────────────────────────
    duration = round(time.time() - start_time, 1)

    table = Table(
        title="Phase 1 Results",
        show_header=True,
        header_style="bold"
    )
    table.add_column("Ticker",   style="cyan", width=10)
    table.add_column("Status",   width=12)
    table.add_column("Rows",     width=8)
    table.add_column("Features", width=10)
    table.add_column("Latest Close", width=14)

    for ticker in success:
        feat_df = processed.get(ticker)
        if feat_df is not None:
            latest_close = (
                f"${feat_df['close'].iloc[-1]:.2f}"
                if "close" in feat_df.columns else "—"
            )
            table.add_row(
                ticker,
                "[green]✓ Success[/green]",
                str(feat_df.shape[0]),
                str(feat_df.shape[1]),
                latest_close,
            )

    for ticker in failed:
        table.add_row(
            ticker,
            "[red]✗ Failed[/red]",
            "—", "—", "—",
        )

    console.print(f"\n")
    console.print(table)

    success_rate = (
        len(success) / len(tickers) * 100
        if tickers else 0
    )
    console.print(
        f"\n[bold]Done in {duration}s[/bold] | "
        f"Success: {len(success)}/{len(tickers)} "
        f"({success_rate:.0f}%)"
    )
    console.print(
        f"[dim]Features saved to: "
        f"{settings.data_processed_path}[/dim]"
    )

    return processed


if __name__ == "__main__":
    # Run with all your tickers
    # Set fetch_macro=False to skip Alpha Vantage for a quick test
    run_pipeline(
        fetch_macro=False,
        fetch_fundamentals=False,
        use_cache=True,
    )