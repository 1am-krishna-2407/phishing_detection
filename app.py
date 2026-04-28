from __future__ import annotations

import warnings

import pandas as pd
import streamlit as st

from src.dashboard_service import (
    ServiceConfigurationError,
    append_url_log,
    bootstrap_runtime,
    delete_url_log_entry,
    get_branch_availability,
    get_runtime_diagnostics,
    get_runtime_profile,
    predict_phishing,
    read_url_logs,
)
from src.ui_theme import inject_theme, render_sidebar

APP_BUILD_ID = "2026-04-28-streamlit-cloud-bootstrap"

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

inject_theme()
render_sidebar(active="analysis")


def _score_text(value: float | None) -> str:
    if value is None:
        return "Not provided"
    return f"{value * 100:.2f}%"


def _status_class(prediction: str) -> str:
    return "status-bad" if prediction.lower() == "phishing" else "status-good"


def _load_logs() -> pd.DataFrame:
    try:
        return read_url_logs(limit=6)
    except Exception:
        return pd.DataFrame(columns=["row_id", "timestamp_utc", "url", "prediction", "phishing_probability"])


@st.cache_resource(show_spinner=False)
def _bootstrap_runtime_once() -> dict[str, str]:
    return bootstrap_runtime()


try:
    with st.spinner("Preparing tokenizer and model bundles for this deployment..."):
        startup_status = _bootstrap_runtime_once()
except ServiceConfigurationError as exc:
    st.error(str(exc))
    st.caption(
        "Streamlit Cloud startup stopped before enabling scans. "
        "Verify the Hugging Face repo id, optional HF token secret, and runtime packages."
    )
    st.stop()

diagnostics = get_runtime_diagnostics()
runtime_profile = get_runtime_profile()
branch_availability = get_branch_availability()

st.markdown(
    f"""
    <section class="app-hero">
      <div class="eyebrow">Threat Analysis Engine</div>
      <div class="hero-title">Phishing Detection Dashboard</div>
      <div class="hero-copy">
        Review suspicious URLs and supporting screenshots in a calmer operator-style layout.
        The interface keeps the scan surface up front while preserving model telemetry and recent cases.
      </div>
      <div class="top-strip">
        <span class="pill">Build: {APP_BUILD_ID}</span>
        <span class="pill">Profile: {runtime_profile['name']}</span>
        <span class="pill">Branches: {", ".join(runtime_profile["active_branches"])}</span>
        <span class="pill">Startup: {startup_status["state"]}</span>
        <span class="pill">Downloads: {"enabled" if runtime_profile["downloads_enabled"] else "disabled"}</span>
      </div>
    </section>
    """,
    unsafe_allow_html=True,
)

issue_keywords = ("Missing model checkpoint", "not found", "requires the optional")
actionable_issues = [issue for issue in diagnostics if any(keyword in issue for keyword in issue_keywords)]
if actionable_issues:
    st.warning("Some required runtime assets are unavailable. Predictions for those branches will fail until fixed.")
    for issue in actionable_issues:
        st.caption(f"- {issue}")
else:
    st.caption(startup_status["message"])

left_col, right_col = st.columns([1.75, 1.0], gap="large")

with left_col:
    st.markdown(
        """
        <div class="panel">
          <div class="section-kicker">Ingestion Module</div>
          <div class="section-title">Submit a target for analysis</div>
          <div class="section-copy">
            Use the URL alone or pair it with a screenshot. The model output and branch scores
            will appear to the right after the scan completes.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    with st.form("prediction_form", clear_on_submit=False):
        url = st.text_input("Suspicious endpoint URL", placeholder="https://brand-verify.example/login")
        uploaded_image = st.file_uploader(
            "Visual asset upload",
            type=["png", "jpg", "jpeg", "webp"],
        )
        submitted = st.form_submit_button("Initiate scan", use_container_width=True)

with right_col:
    st.markdown(
        """
        <div class="result-shell">
          <div class="section-kicker">Runtime Surface</div>
          <div class="section-title">Environment snapshot</div>
          <div class="section-copy">
            Quick context for the current deployment profile and recent activity before you run a scan.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    stat_a, stat_b = st.columns(2)
    stat_a.metric("URL Branch", "Model" if "url model" in runtime_profile["active_branches"] else "Offline")
    stat_b.metric("Heavy Models", "On" if runtime_profile["heavy_models_enabled"] else "Off")
    st.markdown("Branch readiness")
    for branch in branch_availability:
        st.caption(f"- {branch.branch}: {branch.status} — {branch.detail}")
    preview_logs = _load_logs()
    st.markdown('<div class="preview-card">', unsafe_allow_html=True)
    st.markdown("Recent cases")
    if preview_logs.empty:
        st.caption("No URL predictions logged yet. Entries will be saved to logs/url_prediction_log.csv.")
    else:
        for row in preview_logs.head(3).itertuples(index=False):
            info_col, action_col = st.columns([5.0, 1.1], gap="small")
            with info_col:
                st.markdown(
                    f"""
                    <div class="log-row">
                      <div style="font-weight:700;">{row.url}</div>
                      <div style="color:var(--muted);font-size:0.88rem;">
                        {row.prediction} • {_score_text(row.phishing_probability)} • {row.timestamp_utc}
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with action_col:
                st.write("")
                if st.button(":material/delete:", key=f"preview-delete-{row.row_id}", use_container_width=True):
                    delete_url_log_entry(int(row.row_id))
                    st.rerun()
        st.caption("Open the Audit Logs page from Streamlit's Pages menu.")
    st.markdown("</div>", unsafe_allow_html=True)

result = None
image_bytes = None
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

st.markdown("")
result_left, result_right = st.columns([1.05, 1.35], gap="large")

with result_left:
    st.markdown(
        """
        <div class="result-shell">
          <div class="section-kicker">Decision Surface</div>
          <div class="section-title">Final verdict</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if result is None:
        st.info("Run a scan to populate the verdict panel.")
    else:
        st.markdown(
            f"""
            <div style="margin-bottom:0.7rem;font-size:2rem;font-weight:700;" class="{_status_class(result.prediction)}">
              {result.prediction}
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.metric("Phishing probability", _score_text(result.probability))
        if uploaded_image is not None and image_bytes is not None:
            st.image(image_bytes, caption=uploaded_image.name, use_container_width=True)

with result_right:
    st.markdown(
        """
        <div class="result-shell">
          <div class="section-kicker">Signal Decomposition</div>
          <div class="section-title">Branch scores</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if result is None:
        st.info("Branch metrics will appear here after a scan.")
    else:
        score_cols = st.columns(3)
        score_cols[0].metric("URL model", _score_text(result.url_probability))
        score_cols[1].metric("Image model", _score_text(result.image_probability))
        score_cols[2].metric("OCR model", _score_text(result.ocr_probability))

        if result.ocr_text:
            tabs = st.tabs(["Extracted text", "Recent log preview"])
            with tabs[0]:
                st.text_area(
                    "OCR output",
                    value=result.ocr_text,
                    height=220,
                    disabled=True,
                    label_visibility="collapsed",
                )
            with tabs[1]:
                st.dataframe(_load_logs(), use_container_width=True, hide_index=True)
        else:
            st.dataframe(_load_logs(), use_container_width=True, hide_index=True)
