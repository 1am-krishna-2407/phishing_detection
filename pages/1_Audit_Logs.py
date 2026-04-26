from __future__ import annotations

from pathlib import Path
import warnings

import streamlit as st

from src.dashboard_service import delete_url_log_entry, read_url_logs
from src.ui_theme import inject_theme, render_sidebar

warnings.filterwarnings(
    "ignore",
    message=r"Accessing `__path__` from `\.models\..*`\. Returning `__path__` instead\.",
    category=FutureWarning,
)


st.set_page_config(
    page_title="Audit Logs",
    page_icon=":ledger:",
    layout="wide",
)

inject_theme()
render_sidebar(active="logs")


def _score_text(value: float | None) -> str:
    if value is None:
        return "Not provided"
    return f"{float(value) * 100:.2f}%"


def _table_for_display(rows: list[dict[str, object]]) -> object:
    try:
        import pandas as pd
    except ImportError:
        return rows
    return pd.DataFrame(rows)


st.markdown(
    """
    <section class="app-hero">
      <div class="eyebrow">Audit Logs</div>
      <div class="hero-title">Prediction history and trace review</div>
      <div class="hero-copy">
        Browse recent URL analyses, inspect branch-level evidence, and remove entries that should not stay in the record.
      </div>
    </section>
    """,
    unsafe_allow_html=True,
)

logs = read_url_logs(limit=250)
summary_a, summary_b, summary_c = st.columns(3)
summary_a.metric("Entries", str(len(logs)))
summary_b.metric(
    "Phishing verdicts",
    str(sum(1 for row in logs if row["prediction"] == "Phishing")) if logs else "0",
)
summary_c.metric("Storage", str(Path("logs/url_prediction_log.csv")))

st.markdown(
    """
    <div class="panel">
      <div class="section-kicker">Archive</div>
      <div class="section-title">Recorded analyses</div>
      <div class="section-copy">
        The newest entries appear first. Expand any row for branch probabilities, OCR excerpts, and timestamp details.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if not logs:
    st.info(f"No URL predictions logged yet. Entries will be saved to `{Path('logs/url_prediction_log.csv')}`.")
else:
    filter_cols = st.columns([1.1, 1.1, 2.4])
    prediction_filter = filter_cols[0].selectbox("Verdict", ["All", "Phishing", "Legitimate"])
    sort_filter = filter_cols[1].selectbox("Sort", ["Newest", "Highest risk"])
    query = filter_cols[2].text_input("Search URL", placeholder="Filter by URL or image name")

    filtered = list(logs)
    if prediction_filter != "All":
        filtered = [row for row in filtered if row["prediction"] == prediction_filter]
    if query.strip():
        needle = query.strip().lower()
        filtered = [
            row for row in filtered
            if needle in (row["url"] or "").lower() or needle in (row["image_name"] or "").lower()
        ]
    if sort_filter == "Highest risk":
        filtered = sorted(
            filtered,
            key=lambda row: row["phishing_probability"] if row["phishing_probability"] is not None else -1.0,
            reverse=True,
        )

    table = []
    for row in filtered:
        table.append(
            {
                "timestamp_utc": row["timestamp_utc"],
                "url": row["url"],
                "prediction": row["prediction"],
                "phishing_probability": _score_text(row["phishing_probability"]),
                "url_probability": _score_text(row["url_probability"]),
                "image_probability": _score_text(row["image_probability"]),
                "ocr_probability": _score_text(row["ocr_probability"]),
                "image_name": row["image_name"],
            }
        )
    st.dataframe(_table_for_display(table), width="stretch", hide_index=True)

    st.markdown("")
    for row in filtered:
        label = f'{row["url"]}  •  {row["prediction"]}  •  {_score_text(row["phishing_probability"])}'
        with st.expander(label):
            details = [
                {"Field": "Timestamp", "Value": row["timestamp_utc"]},
                {"Field": "URL Probability", "Value": _score_text(row["url_probability"])},
                {"Field": "Image Probability", "Value": _score_text(row["image_probability"])},
                {"Field": "OCR Probability", "Value": _score_text(row["ocr_probability"])},
                {"Field": "Image Name", "Value": row["image_name"] or "Not provided"},
                {"Field": "OCR Excerpt", "Value": row["ocr_text_excerpt"] or "Not provided"},
            ]
            st.dataframe(_table_for_display(details), width="stretch", hide_index=True)
            if st.button("Delete entry", key=f'delete-log-{row["row_id"]}', width="stretch"):
                delete_url_log_entry(int(row["row_id"]))
                st.rerun()
