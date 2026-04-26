from __future__ import annotations

from pathlib import Path
import warnings

import pandas as pd
import streamlit as st

from src.dashboard_service import (
    ServiceConfigurationError,
    append_url_log,
    delete_url_log_entry,
    get_runtime_diagnostics,
    get_runtime_profile,
    get_warmup_status,
    predict_phishing,
    read_url_logs,
    start_background_warmup,
)

warnings.filterwarnings(
    "ignore",
    message=r"Accessing `__path__` from `\.models\..*`\. Returning `__path__` instead\.",
    category=FutureWarning,
)


st.set_page_config(
    page_title="Phishing Detection Dashboard",
    page_icon=":shield:",
    layout="wide",
)


def _score_text(value: float | None) -> str:
    if value is None:
        return "Not provided"
    return f"{value * 100:.2f}%"


def _load_logs() -> pd.DataFrame:
    return read_url_logs(limit=25)


st.title("Phishing Detection Dashboard")
st.caption("Check pasted URLs and uploaded screenshots, then keep a history of URL prediction results.")

start_background_warmup()
warmup_status = get_warmup_status()
if warmup_status["state"] == "running":
    st.info("Preparing the URL model in the background. The app is usable while it warms up.")
elif warmup_status["state"] == "error":
    st.warning(
        "The hosted URL model could not be prepared yet. URL checks will fail until the model assets are ready."
    )

diagnostics = get_runtime_diagnostics()
runtime_profile = get_runtime_profile()
st.caption(
    f"Runtime profile: `{runtime_profile['name']}`  •  Active branches: "
    + ", ".join(runtime_profile["active_branches"])
)
if diagnostics:
    issue_keywords = ("Missing model checkpoint", "not found", "requires the optional")
    actionable_issues = [issue for issue in diagnostics if any(keyword in issue for keyword in issue_keywords)]
    informational_notes = [issue for issue in diagnostics if issue not in actionable_issues]

    if actionable_issues:
        st.warning(
            "This deployment is missing some runtime assets. Predictions that depend on those "
            "models will fail until the environment is completed."
        )
        for issue in actionable_issues:
            st.caption(f"- {issue}")

    for issue in informational_notes:
        st.info(issue)

with st.form("prediction_form", clear_on_submit=False):
    url = st.text_input("URL", placeholder="https://example.com/login")
    uploaded_image = st.file_uploader(
        "Upload a webpage screenshot or suspicious image",
        type=["png", "jpg", "jpeg", "webp"],
    )
    submitted = st.form_submit_button("Analyze")

if submitted:
    if not url.strip() and uploaded_image is None:
        st.warning("Enter a URL, upload an image, or provide both before running analysis.")
    else:
        image_bytes = uploaded_image.getvalue() if uploaded_image is not None else None
        try:
            with st.spinner("Running phishing analysis..."):
                result = predict_phishing(url=url.strip() or None, image_bytes=image_bytes)
                if url.strip():
                    append_url_log(
                        url=url.strip(),
                        result=result,
                        image_name=uploaded_image.name if uploaded_image is not None else None,
                    )
        except ServiceConfigurationError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(
                "The app hit an unexpected runtime error while processing this request. "
                "Check the Streamlit logs for details."
            )
            st.exception(exc)
        else:
            top_left, top_right = st.columns([1.2, 1.8])
            with top_left:
                st.subheader("Final Verdict")
                st.metric("Prediction", result.prediction)
                st.metric("Phishing Probability", _score_text(result.probability))
                if uploaded_image is not None:
                    st.image(image_bytes, caption=uploaded_image.name, width="stretch")

            with top_right:
                st.subheader("Branch Scores")
                score_cols = st.columns(3)
                score_cols[0].metric("URL Model", _score_text(result.url_probability))
                score_cols[1].metric("Image Model", _score_text(result.image_probability))
                score_cols[2].metric("OCR Model", _score_text(result.ocr_probability))

                if result.ocr_text:
                    st.subheader("Extracted OCR Text")
                    st.text_area(
                        "OCR output",
                        value=result.ocr_text,
                        height=180,
                        disabled=True,
                        label_visibility="collapsed",
                    )

st.divider()
st.subheader("Recent URL Prediction Log")
logs = _load_logs()
if logs.empty:
    st.info(f"No URL predictions logged yet. Entries will be saved to `{Path('logs/url_prediction_log.csv')}`.")
else:
    for row in logs.itertuples(index=False):
        info_col, action_col = st.columns([6, 1])
        with info_col:
            title = f"{row.url}  •  {row.prediction}  •  {_score_text(row.phishing_probability)}"
            with st.expander(title):
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
                st.dataframe(details, width="stretch", hide_index=True)
        with action_col:
            st.write("")
            st.write("")
            if st.button("Delete", key=f"delete-log-{row.row_id}", width="stretch"):
                delete_url_log_entry(int(row.row_id))
                st.rerun()
