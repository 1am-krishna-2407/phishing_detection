from __future__ import annotations

import streamlit as st


THEME_CSS = """
<style>
:root {
    --bg: #11100d;
    --panel: #1a1712;
    --panel-soft: #201c16;
    --panel-alt: #15120e;
    --border: #3b3125;
    --text: #f2e6d7;
    --muted: #b8a894;
    --beige: #e6cfb2;
    --tan: #c49a6c;
    --tan-strong: #a87645;
    --danger: #b65c4c;
    --ok: #9fba86;
}

.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stHeader"] {
    background: var(--bg);
}

[data-testid="stAppViewContainer"] > .main {
    background:
        radial-gradient(circle at top right, rgba(196, 154, 108, 0.14), transparent 30%),
        radial-gradient(circle at left top, rgba(230, 207, 178, 0.08), transparent 24%),
        var(--bg);
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #17140f 0%, #13110d 100%);
    border-right: 1px solid rgba(196, 154, 108, 0.18);
}

[data-testid="stSidebar"] * {
    color: var(--text);
}

.block-container {
    max-width: 1220px;
    padding-top: 2.2rem;
    padding-bottom: 3rem;
}

h1, h2, h3, h4, h5, h6, p, span, label, div {
    color: var(--text);
}

a {
    color: var(--beige) !important;
}

.app-hero {
    border: 1px solid rgba(196, 154, 108, 0.18);
    background: linear-gradient(145deg, rgba(32, 28, 22, 0.96), rgba(22, 18, 14, 0.96));
    border-radius: 8px;
    padding: 1.4rem 1.5rem;
    margin-bottom: 1rem;
}

.eyebrow {
    color: var(--tan);
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

.hero-title {
    font-size: 2.2rem;
    line-height: 1.05;
    font-weight: 700;
    margin: 0.35rem 0 0.55rem 0;
}

.hero-copy {
    max-width: 48rem;
    color: var(--muted);
    font-size: 0.98rem;
    line-height: 1.6;
}

.top-strip {
    display: flex;
    gap: 0.75rem;
    flex-wrap: wrap;
    margin: 1rem 0 1.3rem 0;
}

.pill {
    border: 1px solid rgba(196, 154, 108, 0.18);
    background: rgba(230, 207, 178, 0.05);
    color: var(--beige);
    border-radius: 999px;
    padding: 0.4rem 0.75rem;
    font-size: 0.8rem;
}

.panel {
    border: 1px solid rgba(196, 154, 108, 0.16);
    background: linear-gradient(180deg, rgba(26, 23, 18, 0.98), rgba(19, 17, 13, 0.98));
    border-radius: 8px;
    padding: 1rem 1rem 0.7rem 1rem;
}

.section-kicker {
    color: var(--tan);
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}

.section-title {
    font-size: 1.2rem;
    font-weight: 700;
    margin-bottom: 0.25rem;
}

.section-copy {
    color: var(--muted);
    font-size: 0.92rem;
    line-height: 1.55;
    margin-bottom: 0.9rem;
}

.result-shell {
    border: 1px solid rgba(196, 154, 108, 0.16);
    background: linear-gradient(180deg, rgba(20, 18, 14, 1), rgba(17, 15, 12, 1));
    border-radius: 8px;
    padding: 1rem;
    min-height: 100%;
}

.status-good {
    color: #d7ebc6;
}

.status-bad {
    color: #ffd3bf;
}

.preview-card {
    border: 1px solid rgba(196, 154, 108, 0.16);
    background: rgba(230, 207, 178, 0.04);
    border-radius: 8px;
    padding: 0.9rem 1rem;
}

.log-row {
    border-top: 1px solid rgba(196, 154, 108, 0.12);
    padding-top: 0.75rem;
    margin-top: 0.75rem;
}

[data-testid="stForm"] {
    border: 1px solid rgba(196, 154, 108, 0.16);
    background: linear-gradient(180deg, rgba(26, 23, 18, 0.98), rgba(18, 16, 12, 0.98));
    border-radius: 8px;
    padding: 1rem;
}

[data-testid="stMetric"] {
    border: 1px solid rgba(196, 154, 108, 0.12);
    background: rgba(230, 207, 178, 0.03);
    border-radius: 8px;
    padding: 0.85rem;
}

[data-testid="stMetricLabel"],
[data-testid="stMetricLabel"] * {
    color: var(--muted) !important;
}

[data-testid="stMetricValue"] {
    color: var(--text) !important;
}

.stTextInput input,
.stTextArea textarea,
.stFileUploader section,
.stFileUploader div[data-testid="stFileUploaderDropzone"] {
    background: #2a2219 !important;
    color: var(--text) !important;
    border-color: rgba(196, 154, 108, 0.22) !important;
    border-radius: 8px !important;
}

.stButton > button,
.stDownloadButton > button,
.stFormSubmitButton > button {
    background: linear-gradient(180deg, var(--tan), var(--tan-strong)) !important;
    color: #17120d !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
}

.stButton > button:hover,
.stDownloadButton > button:hover,
.stFormSubmitButton > button:hover {
    background: linear-gradient(180deg, #d7b089, #b9834e) !important;
}

.stAlert {
    border-radius: 8px !important;
    border-width: 1px !important;
}

[data-baseweb="tab-list"] {
    gap: 0.4rem;
}

button[role="tab"] {
    background: rgba(230, 207, 178, 0.04) !important;
    border: 1px solid rgba(196, 154, 108, 0.14) !important;
    border-radius: 8px !important;
    color: var(--muted) !important;
}

button[role="tab"][aria-selected="true"] {
    background: rgba(196, 154, 108, 0.18) !important;
    color: var(--text) !important;
}

.stDataFrame, [data-testid="stTable"] {
    border: 1px solid rgba(196, 154, 108, 0.16);
    border-radius: 8px;
    overflow: hidden;
}

[data-testid="stExpander"] {
    border: 1px solid rgba(196, 154, 108, 0.14) !important;
    border-radius: 8px !important;
    background: rgba(230, 207, 178, 0.03) !important;
}

.st-emotion-cache-16txtl3, .st-emotion-cache-1wmy9hl {
    color: var(--text);
}
</style>
"""


def inject_theme() -> None:
    st.markdown(THEME_CSS, unsafe_allow_html=True)


def render_sidebar(active: str) -> None:
    with st.sidebar:
        st.markdown(
            """
            <div style="padding:0.4rem 0 1.1rem 0;">
              <div class="eyebrow">Phishing Detection Suite</div>
              <div style="font-size:1.35rem;font-weight:700;margin-top:0.2rem;">Signal Desk</div>
              <div style="color:var(--muted);font-size:0.88rem;line-height:1.5;margin-top:0.45rem;">
                Warm-toned operator console for URL, OCR, and screenshot review.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        analysis_state = "Current view" if active == "analysis" else "Open from Pages menu"
        logs_state = "Current view" if active == "logs" else "Open from Pages menu"
        st.markdown(
            f"""
            <div class="preview-card" style="margin-bottom:0.7rem;">
              <div style="font-weight:700;">Analysis</div>
              <div style="color:var(--muted);font-size:0.84rem;">{analysis_state}</div>
            </div>
            <div class="preview-card" style="margin-bottom:0.7rem;">
              <div style="font-weight:700;">Audit Logs</div>
              <div style="color:var(--muted);font-size:0.84rem;">{logs_state}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("---")
        st.caption("Model-backed phishing analysis with persistent audit history.")
