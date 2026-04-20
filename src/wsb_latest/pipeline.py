from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import re
import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore", message="Timestamp.utcnow is deprecated")

REDDIT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 WSBLatestAnalysis/1.0"
}
REDDIT_TIMEOUT = 30
SUBREDDIT = "wallstreetbets"
CASHTAG_RE = re.compile(r"\$([A-Za-z]{2,5})\b", re.IGNORECASE)
UPPER_TOKEN_RE = re.compile(r"\b[A-Z]{2,5}\b")
COMMON_FALSE_POSITIVES = {
    "APE",
    "APES",
    "ATH",
    "BAG",
    "BAGS",
    "BTFD",
    "CEO",
    "CFO",
    "CTO",
    "CUSIP",
    "DD",
    "EDIT",
    "ELI5",
    "EPS",
    "ETF",
    "EV",
    "FOMO",
    "GDP",
    "GPU",
    "HODL",
    "IMO",
    "IPO",
    "IRS",
    "ITM",
    "LOL",
    "LOW",
    "MBA",
    "MOON",
    "NAV",
    "NYSE",
    "OTM",
    "PE",
    "PDT",
    "PUMP",
    "PUTS",
    "ROI",
    "SEC",
    "SELL",
    "SHORT",
    "SPAC",
    "TLDR",
    "USA",
    "USD",
    "WTF",
    "WSB",
    "WACC",
    "YOLO",
    "FCF",
    "ESG",
}


@dataclass
class PipelineConfig:
    top_n: int = 10
    per_feed: int = 100
    price_period: str = "1y"
    output_dir: Path = Path("outputs") / "latest_wsb_analysis"


def ensure_nltk_resources() -> None:
    for resource in ("vader_lexicon",):
        try:
            nltk.data.find(f"sentiment/{resource}.zip")
        except LookupError:
            nltk.download(resource, quiet=True)


def fetch_reddit_feed(feed: str, limit: int, top_time: str = "week") -> list[dict]:
    url = f"https://www.reddit.com/r/{SUBREDDIT}/{feed}.json"
    params: dict[str, object] = {"limit": limit, "raw_json": 1}
    if feed == "top":
        params["t"] = top_time

    response = requests.get(url, headers=REDDIT_HEADERS, params=params, timeout=REDDIT_TIMEOUT)
    response.raise_for_status()
    payload = response.json()
    return payload.get("data", {}).get("children", [])


def load_recent_posts(per_feed: int) -> tuple[pd.DataFrame, str]:
    feeds = ("hot", "new", "rising", "top")
    collected: list[dict] = []
    failures: list[str] = []

    for feed in feeds:
        try:
            children = fetch_reddit_feed(feed, per_feed)
            for child in children:
                post = child.get("data", {})
                collected.append(
                    {
                        "id": post.get("id"),
                        "feed": feed,
                        "title": post.get("title") or "",
                        "body": post.get("selftext") or "",
                        "score": post.get("score", 0),
                        "num_comments": post.get("num_comments", 0),
                        "created_utc": post.get("created_utc"),
                        "permalink": f"https://www.reddit.com{post.get('permalink', '')}",
                        "url": post.get("url") or "",
                    }
                )
        except Exception as exc:  # pragma: no cover - network fallback path
            failures.append(f"{feed}: {exc}")

    if collected:
        posts_df = pd.DataFrame(collected).drop_duplicates(subset=["id"]).copy()
        posts_df["created_utc"] = pd.to_datetime(posts_df["created_utc"], unit="s", utc=True)
        posts_df["text"] = (posts_df["title"].fillna("") + " " + posts_df["body"].fillna("")).str.strip()
        return posts_df.sort_values("created_utc", ascending=False), "reddit_public_json"

    fallback_paths = [Path("wsb_reddit_api_data.csv"), Path("wsb_pushshift_data.csv")]
    for fallback in fallback_paths:
        if fallback.exists():
            df = pd.read_csv(str(fallback))
            title_col = "title" if "title" in df.columns else df.columns[0]
            body_col = "body" if "body" in df.columns else ("selftext" if "selftext" in df.columns else title_col)
            timestamp_col = "timestamp" if "timestamp" in df.columns else None
            fallback_df = pd.DataFrame(
                {
                    "id": df.index.astype(str),
                    "feed": "fallback_csv",
                    "title": df[title_col].fillna(""),
                    "body": df[body_col].fillna(""),
                    "score": df.get("score", pd.Series(0, index=df.index)).fillna(0),
                    "num_comments": df.get("num_comments", pd.Series(0, index=df.index)).fillna(0),
                    "created_utc": pd.to_datetime(df[timestamp_col]) if timestamp_col else pd.Timestamp.utcnow(),
                    "permalink": "",
                    "url": "",
                }
            )
            fallback_df["text"] = (fallback_df["title"] + " " + fallback_df["body"]).str.strip()
            return fallback_df.sort_values("created_utc", ascending=False), f"fallback:{fallback.name}"

    failure_text = "; ".join(failures) if failures else "No posts returned from Reddit"
    raise RuntimeError(f"Unable to fetch recent WSB posts. {failure_text}")


def extract_candidate_tickers(text: str) -> Counter:
    cash_tags = [match.upper() for match in CASHTAG_RE.findall(text or "")]
    upper_tokens = [
        match.upper()
        for match in UPPER_TOKEN_RE.findall(text or "")
        if len(match) >= 3
    ]

    combined = [token for token in cash_tags + upper_tokens if token not in COMMON_FALSE_POSITIVES]
    return Counter(combined)


def annotate_posts_with_tickers(posts_df: pd.DataFrame) -> pd.DataFrame:
    ensure_nltk_resources()
    analyzer = SentimentIntensityAnalyzer()

    annotated = posts_df.copy()
    annotated["ticker_counts"] = annotated["text"].apply(extract_candidate_tickers)
    annotated["candidate_mentions"] = annotated["ticker_counts"].apply(lambda counter: sum(counter.values()))
    annotated["sentiment"] = annotated["text"].apply(lambda text: analyzer.polarity_scores(text or "")["compound"])
    annotated["engagement"] = annotated.apply(
        lambda row: math.log1p(max(float(row["score"]), 0.0)) + math.log1p(max(float(row["num_comments"]), 0.0)),
        axis=1,
    )
    return annotated


def validate_tickers(candidates: Iterable[str]) -> set[str]:
    valid: set[str] = set()

    for symbol in candidates:
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                history = yf.Ticker(symbol).history(period="1mo", interval="1d", auto_adjust=False)
            if not history.empty and history["Close"].notna().sum() >= 3:
                valid.add(symbol)
        except Exception:
            continue

    return valid


def build_mentions_table(posts_df: pd.DataFrame, top_n: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    exploded_rows: list[dict] = []
    candidate_counter = Counter()

    for _, row in posts_df.iterrows():
        ticker_counts: Counter = row["ticker_counts"]
        for symbol, count in ticker_counts.items():
            candidate_counter[symbol] += count
            exploded_rows.append(
                {
                    "post_id": row["id"],
                    "ticker": symbol,
                    "mentions": count,
                    "sentiment": row["sentiment"],
                    "score": row["score"],
                    "num_comments": row["num_comments"],
                    "engagement": row["engagement"],
                    "created_utc": row["created_utc"],
                    "title": row["title"],
                    "permalink": row["permalink"],
                }
            )

    mentions_df = pd.DataFrame(exploded_rows)
    if mentions_df.empty:
        raise RuntimeError("No ticker candidates were extracted from the latest WSB posts.")

    candidates = [symbol for symbol, _ in candidate_counter.most_common(80)]
    valid_symbols = validate_tickers(candidates)
    filtered = mentions_df[mentions_df["ticker"].isin(valid_symbols)].copy()
    if filtered.empty:
        raise RuntimeError("Ticker extraction found candidates, but none validated against Yahoo Finance.")

    summary = (
        filtered.groupby("ticker")
        .agg(
            mention_count=("mentions", "sum"),
            post_count=("post_id", "nunique"),
            avg_sentiment=("sentiment", "mean"),
            total_score=("score", "sum"),
            total_comments=("num_comments", "sum"),
            engagement=("engagement", "sum"),
            latest_mention=("created_utc", "max"),
        )
        .reset_index()
    )
    summary["rank_score"] = summary["mention_count"] + (summary["post_count"] * 4)
    summary = summary.sort_values(
        ["rank_score", "mention_count", "post_count", "engagement"],
        ascending=[False, False, False, False],
    ).head(top_n)

    top_tickers = set(summary["ticker"])
    filtered = filtered[filtered["ticker"].isin(top_tickers)].copy()
    return summary, filtered.sort_values(["ticker", "created_utc"], ascending=[True, False])


def compute_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    avg_gain = gains.ewm(alpha=1 / window, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1 / window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_price_features(price_df: pd.DataFrame) -> pd.DataFrame:
    enriched = price_df.copy()
    enriched["Daily Return"] = enriched["Close"].pct_change()
    enriched["30-Day Rolling STD"] = enriched["Close"].rolling(window=30).std()
    enriched["30-Day Rolling SMA"] = enriched["Close"].rolling(window=30).mean()
    enriched["30-Day Rolling EWMA"] = enriched["Close"].ewm(halflife=30).mean()
    enriched["RSI"] = compute_rsi(enriched["Close"])
    ewm_12 = enriched["Close"].ewm(span=12, adjust=False).mean()
    ewm_26 = enriched["Close"].ewm(span=26, adjust=False).mean()
    enriched["MACD"] = ewm_12 - ewm_26
    enriched["5D Return"] = enriched["Close"].pct_change(periods=5)
    enriched["21D Return"] = enriched["Close"].pct_change(periods=21)
    return enriched


def evaluate_linear_regression(price_df: pd.DataFrame) -> tuple[float | None, pd.DataFrame]:
    feature_columns = [
        "Volume",
        "30-Day Rolling STD",
        "30-Day Rolling SMA",
        "30-Day Rolling EWMA",
        "RSI",
        "MACD",
        "5D Return",
        "21D Return",
    ]
    model_df = price_df[feature_columns + ["Close"]].dropna().copy()
    if len(model_df) < 40:
        return None, pd.DataFrame()

    X = model_df[feature_columns]
    y = model_df[["Close"]]
    split = max(int(len(model_df) * 0.8), 1)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    if X_test.empty:
        return None, pd.DataFrame()

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    predictions = model.predict(X_test_scaled)
    rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))

    results = y_test.copy()
    results["Predicted Close"] = predictions
    return rmse, results


def fetch_price_history(symbol: str, period: str) -> pd.DataFrame:
    history = yf.Ticker(symbol).history(period=period, interval="1d", auto_adjust=False)
    if history.empty:
        raise RuntimeError(f"No Yahoo Finance price history returned for {symbol}.")
    history.index = pd.to_datetime(history.index).tz_localize(None)
    return history


def plot_market_summary(summary_df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ordered = summary_df.sort_values("mention_count", ascending=True)

    axes[0].barh(ordered["ticker"], ordered["mention_count"], color="#4c78a8")
    axes[0].set_title("Latest WSB mentions")
    axes[0].set_xlabel("Mentions")

    colors = ["#2ca02c" if value >= 0 else "#d62728" for value in ordered["avg_sentiment"]]
    axes[1].barh(ordered["ticker"], ordered["avg_sentiment"], color=colors)
    axes[1].set_title("Average VADER sentiment")
    axes[1].set_xlabel("Compound sentiment")
    axes[1].axvline(0, color="black", linewidth=0.8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_relative_performance(price_histories: dict[str, pd.DataFrame], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    for symbol, history in price_histories.items():
        close = history["Close"].dropna()
        if close.empty:
            continue
        normalized = close / close.iloc[0] * 100
        ax.plot(normalized.index, normalized.values, label=symbol, linewidth=1.8)

    ax.set_title("Relative price performance (normalized to 100)")
    ax.set_ylabel("Normalized close")
    ax.legend(ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_symbol_dashboard(symbol: str, features_df: pd.DataFrame, regression_results: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    axes[0].plot(features_df.index, features_df["Close"], label="Close", color="#1f77b4")
    axes[0].plot(features_df.index, features_df["30-Day Rolling SMA"], label="30-Day SMA", color="#ff7f0e")
    axes[0].plot(features_df.index, features_df["30-Day Rolling EWMA"], label="30-Day EWMA", color="#2ca02c")
    axes[0].set_title(f"{symbol} technical overview")
    axes[0].legend(loc="upper left")
    axes[0].grid(alpha=0.3)

    axes[1].plot(features_df.index, features_df["RSI"], label="RSI", color="#9467bd")
    axes[1].axhline(70, linestyle="--", color="red", linewidth=0.8)
    axes[1].axhline(30, linestyle="--", color="green", linewidth=0.8)
    axes[1].set_ylabel("RSI")
    axes[1].grid(alpha=0.3)

    axes[2].plot(features_df.index, features_df["MACD"], label="MACD", color="#8c564b")
    axes[2].axhline(0, linestyle="--", color="black", linewidth=0.8)
    if not regression_results.empty:
        axes[2].plot(
            regression_results.index,
            regression_results["Predicted Close"],
            label="Predicted Close",
            color="#e377c2",
            linewidth=1.5,
        )
    axes[2].set_ylabel("MACD / Predicted")
    axes[2].grid(alpha=0.3)
    axes[2].legend(loc="upper left")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_summary_markdown(
    config: PipelineConfig,
    source_name: str,
    summary_df: pd.DataFrame,
    analysis_rows: list[dict],
    output_path: Path,
) -> None:
    lines = [
        "# Latest WallStreetBets Top 10 Analysis",
        "",
        f"- Data source: `{source_name}`",
        f"- Subreddit: `{SUBREDDIT}`",
        f"- Price lookback: `{config.price_period}`",
        f"- Generated at (UTC): `{pd.Timestamp.utcnow().isoformat()}`",
        "",
        "## Top tickers",
        "",
        "| Rank | Ticker | Mentions | Posts | Avg Sentiment |",
        "| --- | --- | ---: | ---: | ---: |",
    ]

    for rank, row in enumerate(summary_df.to_dict(orient="records"), start=1):
        lines.append(
            f"| {rank} | {row['ticker']} | {int(row['mention_count'])} | {int(row['post_count'])} | {row['avg_sentiment']:.3f} |"
        )

    lines.extend(["", "## Market snapshot", ""])
    for row in analysis_rows:
        lines.append(
            "- "
            f"**{row['ticker']}**: close `{row['last_close']:.2f}`, "
            f"1M return `{row['return_21d']:.2%}` if available, "
            f"RSI `{row['latest_rsi']:.2f}` and "
            f"regression RMSE `{row['rmse']:.3f}`"
            if row["rmse"] is not None
            else "- "
            f"**{row['ticker']}**: close `{row['last_close']:.2f}`, "
            f"1M return `{row['return_21d']:.2%}` if available, "
            f"RSI `{row['latest_rsi']:.2f}`"
        )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_pipeline(top_n: int = 10, per_feed: int = 100, price_period: str = "1y", output_dir: str | Path | None = None) -> dict:
    config = PipelineConfig(
        top_n=top_n,
        per_feed=per_feed,
        price_period=price_period,
        output_dir=Path(output_dir) if output_dir else Path("outputs") / "latest_wsb_analysis",
    )
    charts_dir = config.output_dir / "charts"
    prices_dir = config.output_dir / "prices"
    charts_dir.mkdir(parents=True, exist_ok=True)
    prices_dir.mkdir(parents=True, exist_ok=True)

    for file_name in (
        "latest_wsb_posts.csv",
        "latest_wsb_mentions.csv",
        "top10_wsb_stocks.csv",
        "top10_summary.json",
        "top10_summary.md",
    ):
        (config.output_dir / file_name).unlink(missing_ok=True)

    for file_name in (
        "top10_mentions_and_sentiment.png",
        "top10_relative_performance.png",
    ):
        (charts_dir / file_name).unlink(missing_ok=True)

    for pattern in ("*_dashboard.png",):
        for existing_file in charts_dir.glob(pattern):
            existing_file.unlink(missing_ok=True)

    for pattern in ("*_price_features.csv", "*_regression_results.csv"):
        for existing_file in prices_dir.glob(pattern):
            existing_file.unlink(missing_ok=True)

    posts_df, source_name = load_recent_posts(config.per_feed)
    annotated_posts = annotate_posts_with_tickers(posts_df)
    summary_df, mentions_df = build_mentions_table(annotated_posts, config.top_n)

    analysis_rows: list[dict] = []
    price_histories: dict[str, pd.DataFrame] = {}

    def latest_value(series: pd.Series) -> float:
        cleaned = series.dropna()
        return float(cleaned.iloc[-1]) if not cleaned.empty else float("nan")

    for symbol in summary_df["ticker"]:
        history = fetch_price_history(symbol, config.price_period)
        features_df = compute_price_features(history)
        rmse, regression_results = evaluate_linear_regression(features_df)

        features_df.to_csv(prices_dir / f"{symbol.lower()}_price_features.csv")
        regression_results.to_csv(prices_dir / f"{symbol.lower()}_regression_results.csv")
        plot_symbol_dashboard(symbol, features_df, regression_results, charts_dir / f"{symbol.lower()}_dashboard.png")
        price_histories[symbol] = features_df

        close = features_df["Close"].dropna()
        analysis_rows.append(
            {
                "ticker": symbol,
                "last_close": float(close.iloc[-1]),
                "latest_rsi": latest_value(features_df["RSI"]),
                "latest_macd": latest_value(features_df["MACD"]),
                "return_5d": latest_value(features_df["5D Return"]),
                "return_21d": latest_value(features_df["21D Return"]),
                "rmse": rmse,
            }
        )

    analysis_df = pd.DataFrame(analysis_rows).sort_values("ticker")
    merged_summary = summary_df.merge(analysis_df, on="ticker", how="left")

    annotated_posts.to_csv(config.output_dir / "latest_wsb_posts.csv", index=False)
    mentions_df.to_csv(config.output_dir / "latest_wsb_mentions.csv", index=False)
    merged_summary.to_csv(config.output_dir / "top10_wsb_stocks.csv", index=False)

    plot_market_summary(merged_summary, charts_dir / "top10_mentions_and_sentiment.png")
    plot_relative_performance(price_histories, charts_dir / "top10_relative_performance.png")

    summary_payload = {
        "data_source": source_name,
        "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
        "top_n": config.top_n,
        "symbols": merged_summary.to_dict(orient="records"),
    }
    (config.output_dir / "top10_summary.json").write_text(json.dumps(summary_payload, indent=2, default=str), encoding="utf-8")
    write_summary_markdown(config, source_name, merged_summary, analysis_rows, config.output_dir / "top10_summary.md")

    return {
        "data_source": source_name,
        "posts": int(len(annotated_posts)),
        "tickers": merged_summary["ticker"].tolist(),
        "output_dir": str(config.output_dir.resolve()),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Refresh the project with the latest top WallStreetBets stocks.")
    parser.add_argument("--top-n", type=int, default=10, help="Number of WSB tickers to analyze.")
    parser.add_argument("--per-feed", type=int, default=100, help="Posts to fetch from each Reddit listing feed.")
    parser.add_argument("--price-period", type=str, default="1y", help="Yahoo Finance lookback period, e.g. 6mo, 1y, 2y.")
    parser.add_argument("--output-dir", type=str, default=str(Path("outputs") / "latest_wsb_analysis"), help="Directory for CSV, JSON, markdown and PNG outputs.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run_pipeline(top_n=args.top_n, per_feed=args.per_feed, price_period=args.price_period, output_dir=args.output_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()




