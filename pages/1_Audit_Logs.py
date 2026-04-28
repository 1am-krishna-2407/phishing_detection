from __future__ import annotations

from pathlib import Path
import warnings

import pandas as pd
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

try:
    logs = read_url_logs(limit=250)
except Exception:
    logs = pd.DataFrame(columns=["row_id", "timestamp_utc", "url", "prediction"])
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

if logs.empty:
    st.info("No URL predictions logged yet. Entries will be saved to logs/url_prediction_log.csv.")
else:
    filter_cols = st.columns([1.1, 1.1, 2.4])
    prediction_filter = filter_cols[0].selectbox("Verdict", ["All", "Phishing", "Legitimate"])
    sort_filter = filter_cols[1].selectbox("Sort", ["Newest", "Highest risk"])
    query = filter_cols[2].text_input("Search URL", placeholder="Filter by URL or image name")

    filtered = logs.copy()
    if prediction_filter != "All":
        filtered = filtered[filtered["prediction"] == prediction_filter]
    if query.strip():
        needle = query.strip().lower()
        filtered = filtered[
            filtered["url"].fillna("").str.lower().str.contains(needle)
            | filtered["image_name"].fillna("").str.lower().str.contains(needle)
        ]
    if sort_filter == "Highest risk":
        filtered = filtered.sort_values("phishing_probability", ascending=False, na_position="last")

    table = filtered[
        [
            "timestamp_utc",
            "url",
            "prediction",
            "phishing_probability",
            "url_probability",
            "image_probability",
            "ocr_probability",
            "image_name",
        ]
    ].copy()
    if not table.empty:
        for column in ("phishing_probability", "url_probability", "image_probability", "ocr_probability"):
            table[column] = table[column].map(_score_text)
    st.dataframe(table, use_container_width=True, hide_index=True)

    st.markdown("")
    for row in filtered.itertuples(index=False):
        label = f"{row.url}  •  {row.prediction}  •  {_score_text(row.phishing_probability)}"
        with st.expander(label):
            details = pd.DataFrame(
                [
                    ("Timestamp", row.timestamp_utc),
                    ("URL Probability", _score_text(row.url_probability)),
                    ("Image Probability", _score_text(row.image_probability)),
                    ("OCR Probability", _score_text(row.ocr_probability)),
                    ("Image Name", row.image_name or "Not provided"),
                    ("OCR Excerpt", row.ocr_text_excerpt or "Not provided"),
                ],
                columns=["Field", "Value"],
            )
            st.dataframe(details, use_container_width=True, hide_index=True)
            if st.button("Delete entry", key=f"delete-log-{row.row_id}", use_container_width=True):
                delete_url_log_entry(int(row.row_id))
                st.rerun()
