from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

_PRICE_COLS = {"last_close", "return_21d", "rmse"}

START_MARKER = "<!-- AUTO_TOP10_START -->"
END_MARKER = "<!-- AUTO_TOP10_END -->"
ANCHOR_MARKER = "<!-- AUTO_TOP10_ANCHOR -->"


def _fmt_float(value: object, decimals: int = 3, percent: bool = False) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "n/a"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if percent:
        return f"{numeric * 100:.2f}%"
    return f"{numeric:.{decimals}f}"


def build_auto_section(project_root: Path) -> str:
    output_dir = project_root / "outputs" / "latest_wsb_analysis"
    charts_dir = output_dir / "charts"
    top10_csv = output_dir / "top10_wsb_stocks.csv"
    top10_json = output_dir / "top10_summary.json"
    top10_md = output_dir / "top10_summary.md"

    if not top10_csv.exists():
        raise FileNotFoundError(f"Missing expected file: {top10_csv}")

    top_df = pd.read_csv(top10_csv)
    if top_df.empty:
        raise RuntimeError("Top 10 summary CSV exists but has no rows.")

    # If price-analysis columns are absent from the CSV (written by an older run
    # or a run that crashed before the merge), backfill from the JSON payload,
    # which always receives the fully-merged symbol records.
    if top10_json.exists() and not _PRICE_COLS.issubset(top_df.columns):
        payload = json.loads(top10_json.read_text(encoding="utf-8"))
        json_df = pd.DataFrame(payload.get("symbols", []))
        if not json_df.empty and "ticker" in json_df.columns:
            fill_cols = [c for c in json_df.columns if c not in top_df.columns]
            if fill_cols:
                top_df = top_df.merge(json_df[["ticker"] + fill_cols], on="ticker", how="left")

    generated_at = "n/a"
    data_source = "n/a"
    if top10_json.exists():
        payload = json.loads(top10_json.read_text(encoding="utf-8"))
        generated_at = str(payload.get("generated_at_utc", "n/a"))
        data_source = str(payload.get("data_source", "n/a"))

    lines: list[str] = []
    lines.append("## Latest Top 10 Snapshot (Auto-updated)")
    lines.append("")
    lines.append(f"- Generated at (UTC): `{generated_at}`")
    lines.append(f"- Data source: `{data_source}`")
    lines.append("- Raw outputs:")
    lines.append("  - `outputs/latest_wsb_analysis/top10_wsb_stocks.csv`")
    lines.append("  - `outputs/latest_wsb_analysis/top10_summary.json`")
    lines.append("  - `outputs/latest_wsb_analysis/top10_summary.md`")
    lines.append("")

    if top10_md.exists():
        lines.append("### Summary")
        lines.append("")
        lines.append("See `outputs/latest_wsb_analysis/top10_summary.md` for the narrative market snapshot.")
        lines.append("")

    lines.append("### Top 10 Table")
    lines.append("")
    lines.append("| Rank | Ticker | Mentions | Posts | Avg Sentiment | Last Close | 21D Return | RMSE |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")

    ordered = top_df.sort_values("rank_score", ascending=False).reset_index(drop=True)
    for rank, row in ordered.iterrows():
        ticker = str(row.get("ticker", ""))
        lines.append(
            "| "
            f"{rank + 1} | {ticker} | {int(row.get('mention_count', 0))} | {int(row.get('post_count', 0))} | "
            f"{_fmt_float(row.get('avg_sentiment'))} | {_fmt_float(row.get('last_close'), decimals=2)} | "
            f"{_fmt_float(row.get('return_21d'), percent=True)} | {_fmt_float(row.get('rmse'))} |"
        )

    lines.append("")
    lines.append("### Aggregate Charts")
    lines.append("")
    lines.append("![Top 10 Mentions and Sentiment](outputs/latest_wsb_analysis/charts/top10_mentions_and_sentiment.png)")
    lines.append("")
    lines.append("![Top 10 Relative Performance](outputs/latest_wsb_analysis/charts/top10_relative_performance.png)")
    lines.append("")

    lines.append("### Per-Ticker Dashboards")
    lines.append("")
    for _, row in ordered.iterrows():
        ticker = str(row.get("ticker", "")).upper()
        chart_path = charts_dir / f"{ticker.lower()}_dashboard.png"
        if chart_path.exists():
            lines.append(f"#### {ticker}")
            lines.append("")
            lines.append(f"![{ticker} Dashboard](outputs/latest_wsb_analysis/charts/{ticker.lower()}_dashboard.png)")
            lines.append("")

    return "\n".join(lines).strip() + "\n"


def update_readme(project_root: Path) -> Path:
    readme_path = project_root / "README.md"
    if not readme_path.exists():
        raise FileNotFoundError(f"Missing README file: {readme_path}")

    text = readme_path.read_text(encoding="utf-8")
    section_text = build_auto_section(project_root)
    wrapped = f"{START_MARKER}\n{section_text}{END_MARKER}"

    marker_block = re.compile(
        rf"\n?{re.escape(START_MARKER)}.*?{re.escape(END_MARKER)}\n?",
        flags=re.DOTALL,
    )
    cleaned = marker_block.sub("\n", text)

    if ANCHOR_MARKER in cleaned:
        updated = cleaned.replace(ANCHOR_MARKER, wrapped, 1)
    else:
        updated = cleaned.rstrip() + "\n\n" + wrapped + "\n"

    readme_path.write_text(updated, encoding="utf-8")
    return readme_path


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    readme_path = update_readme(project_root)
    print(f"Updated README section in: {readme_path}")


if __name__ == "__main__":
    main()


