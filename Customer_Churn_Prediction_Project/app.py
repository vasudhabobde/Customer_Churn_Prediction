"""
Customer Churn Prediction & Analytics Platform
----------------------------------------------
Software Intelligence Lab Project

Run:  streamlit run app.py
Required files in the same folder:
    churn_model_bundle.pkl   (exported by the notebook)
    cleaned_churn_data.csv   (exported by the notebook)
"""
import os

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import stats
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split

# =============================================================================
# CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="Churn Analytics Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Render Plotly charts with their own (white) styling instead of Streamlit's dark theme
def _plotly_light(fig, **kw):
    """Draw Plotly charts with their own white styling (not Streamlit's dark theme).
    Calls the DeltaGenerator method directly, so earlier reruns can never be wrapped twice."""
    kw["theme"] = None
    return st._main.plotly_chart(fig, **kw)


st.plotly_chart = _plotly_light

BUNDLE_PATH = "churn_model_bundle.pkl"
DATA_PATH = "cleaned_churn_data.csv"
RANDOM_STATE = 42

NAVY = "#12304f"
BLUE = "#2563eb"
TEAL = "#0d9488"
RED = "#dc2626"
AMBER = "#d97706"
GREEN = "#15803d"
GREY = "#64748b"
CHURN_COLORS = {"Yes": RED, "No": TEAL}
PLOT_TEMPLATE = "plotly_white"

st.markdown(
    f"""
    <style>
    html, body, [class*="css"] {{ font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif; }}
    .block-container {{ padding-top: 1.2rem; padding-bottom: 3rem; max-width: 1280px; }}
    #MainMenu, footer {{ visibility: hidden; }}

    /* Force white page background */
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {{ background-color: #ffffff; }}
    .stApp {{ color: #1e293b; }}

    /* Dark text for all main-area content, whichever Streamlit theme is active */
    [data-testid="stMain"] label, [data-testid="stMain"] label *,
    [data-testid="stMain"] [data-testid="stWidgetLabel"] *,
    [data-testid="stMain"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stMain"] [data-testid="stMarkdownContainer"] li,
    [data-testid="stMain"] [data-testid="stCaptionContainer"] *,
    [data-testid="stMain"] button[role="tab"] p,
    [data-testid="stMain"] [data-testid="stExpander"] summary * {{ color: #1e293b !important; }}
    [data-testid="stMain"] .result-box h3, [data-testid="stMain"] .result-box p {{ color: #ffffff !important; }}

    /* Light input boxes */
    [data-testid="stMain"] div[data-baseweb="select"] > div,
    [data-testid="stMain"] div[data-baseweb="input"],
    [data-testid="stMain"] div[data-baseweb="input"] input,
    [data-testid="stMain"] [data-testid="stNumberInput"] button {{
        background-color: #ffffff !important; color: #1e293b !important; border-color: #cbd5e1 !important;
    }}
    [data-testid="stMain"] div[data-baseweb="select"] * {{ color: #1e293b !important; }}
    [data-testid="stMain"] div[data-baseweb="select"] svg {{ fill: #1e293b !important; }}
    div[data-baseweb="popover"] li, div[data-baseweb="popover"] ul {{ background-color: #ffffff !important; color: #1e293b !important; }}
    [data-testid="stForm"] {{ background: #f8fafc; border-color: #e2e8f0; }}

    /* ---- Broad fallback rules (work across Streamlit versions) ---- */
    .stApp label, .stApp label *,
    .stApp [data-testid="stWidgetLabel"], .stApp [data-testid="stWidgetLabel"] *,
    .stApp [data-testid="stMarkdownContainer"] p, .stApp [data-testid="stMarkdownContainer"] li,
    .stApp [data-testid="stCaptionContainer"], .stApp [data-testid="stCaptionContainer"] *,
    .stApp button[role="tab"], .stApp button[role="tab"] *,
    .stApp [data-testid="stFileUploader"] small, .stApp [data-testid="stFileUploader"] span,
    .stApp [data-testid="stFileUploaderDropzoneInstructions"] * {{ color: #1e293b !important; }}

    /* Select boxes, inputs and dropdown menus */
    .stApp div[data-baseweb="select"] > div, .stApp div[data-baseweb="input"],
    .stApp div[data-baseweb="input"] input, .stApp div[data-baseweb="base-input"],
    .stApp div[data-baseweb="base-input"] input {{
        background-color: #ffffff !important; color: #1e293b !important; border-color: #cbd5e1 !important;
    }}
    .stApp div[data-baseweb="select"] *, .stApp div[data-baseweb="input"] * {{ color: #1e293b !important; }}
    .stApp div[data-baseweb="select"] svg {{ fill: #1e293b !important; }}
    div[data-baseweb="popover"], div[data-baseweb="popover"] * , div[data-baseweb="menu"], div[data-baseweb="menu"] * {{
        background-color: #ffffff !important; color: #1e293b !important;
    }}

    /* Buttons (download, upload, secondary) */
    .stApp [data-testid="stDownloadButton"] button, .stApp [data-testid="stBaseButton-secondary"],
    .stApp button[kind="secondary"], .stApp [data-testid="stFileUploader"] button {{
        background-color: #ffffff !important; border: 1px solid #94a3b8 !important;
    }}
    .stApp [data-testid="stDownloadButton"] button *, .stApp [data-testid="stBaseButton-secondary"] *,
    .stApp button[kind="secondary"] *, .stApp [data-testid="stFileUploader"] button * {{ color: #12304f !important; }}
    .stApp [data-testid="stBaseButton-primary"], .stApp button[kind="primary"],
    .stApp [data-testid="stFormSubmitButton"] button[kind="primary"] {{ background-color: {BLUE} !important; border: none !important; }}
    .stApp [data-testid="stBaseButton-primary"] *, .stApp button[kind="primary"] * {{ color: #ffffff !important; }}

    /* File upload area and inline code */
    .stApp [data-testid="stFileUploaderDropzone"] {{ background-color: #f8fafc !important; border: 1px dashed #94a3b8 !important; }}
    .stApp code {{ background-color: #e2e8f0 !important; color: #0f172a !important; }}

    /* Keep intentional light-on-dark text */
    .stApp .result-box h3, .stApp .result-box p {{ color: #ffffff !important; }}
    .stApp section[data-testid="stSidebar"], .stApp section[data-testid="stSidebar"] * {{ color: #e8eef6 !important; }}

    /* Sidebar */
    section[data-testid="stSidebar"] {{ background: {NAVY}; }}
    section[data-testid="stSidebar"] * {{ color: #e8eef6 !important; }}
    section[data-testid="stSidebar"] hr {{ border-color: rgba(255,255,255,.18); }}
    .brand {{ font-size: 1.15rem; font-weight: 700; letter-spacing: .3px; margin-bottom: .1rem; }}
    .brand-sub {{ font-size: .78rem; opacity: .75; margin-bottom: .6rem; }}

    /* Page header */
    .page-header {{
        border-left: 6px solid {BLUE}; padding: .35rem 0 .35rem 1rem; margin-bottom: 1.2rem;
    }}
    .page-header h1 {{ margin: 0; font-size: 1.85rem; color: {NAVY}; }}
    .page-header p {{ margin: .2rem 0 0 0; color: {GREY}; font-size: .98rem; }}

    /* KPI cards */
    .kpi {{
        background: #ffffff; border: 1px solid #e2e8f0; border-top: 4px solid {BLUE};
        border-radius: 10px; padding: 1rem 1.1rem; box-shadow: 0 1px 3px rgba(15,23,42,.06);
    }}
    .kpi .value {{ font-size: 1.75rem; font-weight: 700; color: {NAVY}; line-height: 1.15; }}
    .kpi .label {{ font-size: .8rem; color: {GREY}; text-transform: uppercase; letter-spacing: .6px; }}
    .kpi .note {{ font-size: .8rem; color: {GREY}; margin-top: .2rem; }}

    /* Section titles & callouts */
    .section-title {{
        font-size: 1.15rem; font-weight: 600; color: {NAVY};
        border-bottom: 2px solid #e2e8f0; padding-bottom: .3rem; margin: 1.4rem 0 .8rem 0;
    }}
    .callout {{
        background: #f1f5f9; border-left: 4px solid {TEAL}; border-radius: 6px;
        padding: .8rem 1rem; color: #1e293b; font-size: .93rem; margin: .6rem 0;
    }}
    .callout.warn {{ border-left-color: {AMBER}; background: #fffbeb; }}
    .result-box {{
        border-radius: 10px; padding: 1.1rem 1.3rem; color: #fff; margin-bottom: .8rem;
    }}
    .result-box h3 {{ margin: 0; font-size: 1.4rem; color: #fff; }}
    .result-box p {{ margin: .25rem 0 0 0; opacity: .95; font-size: .95rem; }}
    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# DATA & MODEL LOADING
# =============================================================================
@st.cache_resource(show_spinner="Loading trained model...")
def load_bundle(path):
    return joblib.load(path)


@st.cache_data(show_spinner="Loading dataset...")
def load_data(path):
    d = pd.read_csv(path)
    d["TotalCharges"] = pd.to_numeric(d["TotalCharges"], errors="coerce")
    d["SeniorCitizen"] = d["SeniorCitizen"].astype(str)
    d["ChurnFlag"] = (d["Churn"].astype(str).str.lower() == "yes").astype(int)
    return d


missing_files = [p for p in (BUNDLE_PATH, DATA_PATH) if not os.path.exists(p)]
if missing_files:
    st.error(
        "Required file(s) not found: " + ", ".join(f"`{m}`" for m in missing_files)
        + ". Run the notebook, then place the exported files in the same folder as `app.py`."
    )
    st.stop()

bundle = load_bundle(BUNDLE_PATH)
model = bundle["model"]
model_name = bundle.get("model_name", "Classifier")
feature_columns = bundle["feature_columns"]
data = load_data(DATA_PATH)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def page_header(title, subtitle):
    st.markdown(
        f'<div class="page-header"><h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


def section(title):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)


def kpi(col, value, label, note="", color=BLUE):
    col.markdown(
        f'<div class="kpi" style="border-top-color:{color}"><div class="label">{label}</div>'
        f'<div class="value">{value}</div><div class="note">{note}</div></div>',
        unsafe_allow_html=True,
    )


def callout(text, warn=False):
    cls = "callout warn" if warn else "callout"
    st.markdown(f'<div class="{cls}">{text}</div>', unsafe_allow_html=True)


def style_fig(fig, height=360):
    fig.update_layout(
        template=PLOT_TEMPLATE, height=height, margin=dict(l=20, r=20, t=50, b=20),
        title_font=dict(size=15, color=NAVY), font=dict(family="Segoe UI, Arial", color="#1e293b"),
        paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
    )
    return fig


def opts(col):
    return sorted(data[col].dropna().astype(str).unique().tolist())


def predict_proba(df):
    return model.predict_proba(df[feature_columns])[:, 1]


def risk_band(p, low, high):
    if p < low:
        return "Low", GREEN
    if p < high:
        return "Medium", AMBER
    return "High", RED


def get_importances():
    try:
        pre = model.named_steps["preprocessor"]
        est = model.named_steps["model"]
        names = [n.split("__", 1)[-1] for n in pre.get_feature_names_out()]
        if hasattr(est, "feature_importances_"):
            vals, signed = est.feature_importances_, None
        elif hasattr(est, "coef_"):
            signed = est.coef_.ravel()
            vals = np.abs(signed)
        else:
            return None
        out = pd.DataFrame({"Feature": names, "Importance": vals})
        if signed is not None:
            out["Effect"] = np.where(signed > 0, "Increases churn risk", "Decreases churn risk")
        return out.sort_values("Importance", ascending=False)
    except Exception:
        return None


@st.cache_data(show_spinner="Evaluating model on the held-out test set...")
def holdout_evaluation(_model, df):
    """Re-creates the notebook's 80/20 stratified split (random_state=42) to get the same test set."""
    X = df.drop(columns=[c for c in ("customerID", "Churn", "ChurnFlag") if c in df.columns])[feature_columns]
    y = df["ChurnFlag"]
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y)
    prob = _model.predict_proba(X_test)[:, 1]
    return y_test.to_numpy(), prob


# =============================================================================
# SIDEBAR NAVIGATION
# =============================================================================
PAGES = [
    "Overview",
    "Data Explorer",
    "Statistical Analysis",
    "Churn Prediction",
    "Batch Scoring",
    "Model Performance",
    "About",
]

with st.sidebar:
    st.markdown('<div class="brand">Churn Analytics Platform</div>'
                '<div class="brand-sub">Customer Retention Intelligence</div>', unsafe_allow_html=True)
    page = st.radio("Navigation", PAGES, label_visibility="collapsed")
    st.markdown("---")
    st.markdown("**Prediction settings**")
    threshold = st.slider("Decision threshold", 0.10, 0.90, 0.50, 0.05,
                          help="A customer with churn probability at or above this value is classified as 'Churn'.")
    low, high = st.slider("Risk band limits", 0.05, 0.95, (0.30, 0.60), 0.05,
                          help="Below the first value: Low. Between: Medium. Above the second: High.")
    st.markdown("---")
    st.caption(f"Active model: {model_name}")
    st.caption(f"Records loaded: {len(data):,}")


# =============================================================================
# PAGE 1: OVERVIEW
# =============================================================================
def page_overview():
    page_header("Executive Overview",
                "Key indicators of customer churn across the telecom customer base.")

    churn_rate = data["ChurnFlag"].mean()
    lost_revenue = data.loc[data["ChurnFlag"] == 1, "MonthlyCharges"].sum()

    c1, c2, c3, c4 = st.columns(4)
    kpi(c1, f"{len(data):,}", "Total customers", "Records analysed")
    kpi(c2, f"{churn_rate:.1%}", "Churn rate", f"{int(data['ChurnFlag'].sum()):,} customers lost", RED)
    kpi(c3, f"${data['MonthlyCharges'].mean():.2f}", "Avg monthly charge", "Per customer", TEAL)
    kpi(c4, f"${lost_revenue:,.0f}", "Monthly revenue lost", "From churned customers", AMBER)

    section("Where does churn occur?")
    a, b = st.columns(2)
    with a:
        r = data.groupby("Contract")["ChurnFlag"].mean().mul(100).reset_index()
        fig = px.bar(r, x="Contract", y="ChurnFlag", text_auto=".1f", color="Contract",
                     title="Churn rate by contract type (%)",
                     color_discrete_sequence=[RED, AMBER, TEAL])
        fig.update_layout(showlegend=False, yaxis_title="Churn rate (%)")
        st.plotly_chart(style_fig(fig), use_container_width=True)
    with b:
        r = data.groupby("InternetService")["ChurnFlag"].mean().mul(100).reset_index()
        fig = px.bar(r, x="InternetService", y="ChurnFlag", text_auto=".1f", color="InternetService",
                     title="Churn rate by internet service (%)",
                     color_discrete_sequence=[BLUE, RED, TEAL])
        fig.update_layout(showlegend=False, yaxis_title="Churn rate (%)")
        st.plotly_chart(style_fig(fig), use_container_width=True)

    a, b = st.columns(2)
    with a:
        fig = px.histogram(data, x="tenure", color="Churn", nbins=36, barmode="overlay", opacity=.7,
                           title="Customer tenure by churn status", color_discrete_map=CHURN_COLORS)
        fig.update_layout(xaxis_title="Tenure (months)", yaxis_title="Customers")
        st.plotly_chart(style_fig(fig), use_container_width=True)
    with b:
        fig = px.pie(data, names="Churn", hole=.55, title="Churn class distribution",
                     color="Churn", color_discrete_map=CHURN_COLORS)
        st.plotly_chart(style_fig(fig), use_container_width=True)

    section("Key findings")
    m2m = data.loc[data["Contract"] == "Month-to-month", "ChurnFlag"].mean()
    two = data.loc[data["Contract"] == "Two year", "ChurnFlag"].mean()
    avg_t_churn = data.loc[data["ChurnFlag"] == 1, "tenure"].mean()
    avg_t_stay = data.loc[data["ChurnFlag"] == 0, "tenure"].mean()
    callout(f"<b>Contract type:</b> month-to-month customers churn at {m2m:.1%}, compared with {two:.1%} "
            f"for two-year contracts.")
    callout(f"<b>Tenure:</b> customers who churned had an average tenure of {avg_t_churn:.1f} months, "
            f"versus {avg_t_stay:.1f} months for retained customers.")
    callout("These are observed associations in the dataset and do not by themselves prove cause and effect.",
            warn=True)


# =============================================================================
# PAGE 2: DATA EXPLORER
# =============================================================================
def page_explorer():
    page_header("Data Explorer", "Filter the dataset, inspect its structure and examine individual variables.")

    section("Filters")
    f1, f2, f3 = st.columns(3)
    contracts = f1.multiselect("Contract", opts("Contract"), default=opts("Contract"))
    internet = f2.multiselect("Internet service", opts("InternetService"), default=opts("InternetService"))
    t_min, t_max = f3.slider("Tenure range (months)", int(data["tenure"].min()), int(data["tenure"].max()),
                             (int(data["tenure"].min()), int(data["tenure"].max())))
    d = data[data["Contract"].isin(contracts) & data["InternetService"].isin(internet)
             & data["tenure"].between(t_min, t_max)]

    if d.empty:
        st.warning("No customers match the selected filters.")
        return

    c1, c2, c3, c4 = st.columns(4)
    kpi(c1, f"{len(d):,}", "Filtered records", f"of {len(data):,} total")
    kpi(c2, f"{d['ChurnFlag'].mean():.1%}", "Churn rate", "", RED)
    kpi(c3, f"${d['MonthlyCharges'].mean():.2f}", "Avg monthly charge", "", TEAL)
    kpi(c4, f"{d['tenure'].mean():.1f}", "Avg tenure (months)", "", AMBER)

    tab1, tab2, tab3 = st.tabs(["Data table", "Variable analysis", "Dataset structure"])

    with tab1:
        st.dataframe(d.drop(columns=["ChurnFlag"]), height=420)
        st.download_button("Download filtered data (CSV)",
                           d.drop(columns=["ChurnFlag"]).to_csv(index=False).encode(),
                           "filtered_customers.csv", "text/csv")

    with tab2:
        num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
        cat_cols = [c for c in d.columns if c not in num_cols + ["customerID", "Churn", "ChurnFlag"]]
        kind = st.radio("Variable type", ["Numeric", "Categorical"], horizontal=True)
        if kind == "Numeric":
            col = st.selectbox("Variable", num_cols)
            a, b = st.columns(2)
            with a:
                fig = px.histogram(d, x=col, color="Churn", nbins=40, barmode="overlay", opacity=.7,
                                   title=f"Distribution of {col}", color_discrete_map=CHURN_COLORS)
                st.plotly_chart(style_fig(fig), use_container_width=True)
            with b:
                fig = px.box(d, x="Churn", y=col, color="Churn", title=f"{col} by churn status",
                             color_discrete_map=CHURN_COLORS)
                fig.update_layout(showlegend=False)
                st.plotly_chart(style_fig(fig), use_container_width=True)
            st.dataframe(d.groupby("Churn")[col].describe().round(2))
        else:
            col = st.selectbox("Variable", cat_cols)
            r = d.groupby(col)["ChurnFlag"].agg(["mean", "count"]).reset_index()
            r["Churn rate (%)"] = (r["mean"] * 100).round(2)
            r = r.rename(columns={"count": "Customers"}).sort_values("Churn rate (%)")
            a, b = st.columns(2)
            with a:
                fig = px.bar(r, x="Churn rate (%)", y=col, orientation="h", text_auto=".1f",
                             title=f"Churn rate by {col}", color="Churn rate (%)",
                             color_continuous_scale="Reds")
                st.plotly_chart(style_fig(fig), use_container_width=True)
            with b:
                fig = px.bar(r, x="Customers", y=col, orientation="h", title=f"Customers by {col}",
                             color_discrete_sequence=[BLUE])
                st.plotly_chart(style_fig(fig), use_container_width=True)
            st.dataframe(r[[col, "Customers", "Churn rate (%)"]], hide_index=True)

    with tab3:
        info = pd.DataFrame({
            "Column": data.drop(columns=["ChurnFlag"]).columns,
            "Data type": data.drop(columns=["ChurnFlag"]).dtypes.astype(str).values,
            "Missing values": data.drop(columns=["ChurnFlag"]).isna().sum().values,
            "Unique values": data.drop(columns=["ChurnFlag"]).nunique().values,
        })
        st.dataframe(info, hide_index=True, height=420)
        callout("TotalCharges was stored as text in the source file and converted to numeric during cleaning; "
                "customers with a blank value (zero tenure) are treated as missing and imputed by the model pipeline.")


# =============================================================================
# PAGE 3: STATISTICAL ANALYSIS
# =============================================================================
def page_stats():
    page_header("Statistical Analysis",
                "Inferential statistics applied to the customer data: estimation, hypothesis testing, "
                "correlation and outlier detection.")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Sampling & confidence interval", "Hypothesis tests", "Correlation", "Outlier detection (IQR)"]
    )
    charges = data["MonthlyCharges"].dropna().to_numpy()

    # ---- Sampling & CI
    with tab1:
        a, b, c = st.columns(3)
        n = a.slider("Sample size (n)", 5, 300, 30, 5)
        k = b.slider("Number of samples", 100, 5000, 1000, 100)
        conf = c.slider("Confidence level", 0.80, 0.99, 0.95, 0.01)
        rng = np.random.default_rng(RANDOM_STATE)
        means = np.array([rng.choice(charges, n, replace=True).mean() for _ in range(k)])
        pop_mean = charges.mean()
        se = charges.std(ddof=1) / np.sqrt(n)
        lo, hi = stats.t.interval(conf, df=len(charges) - 1, loc=pop_mean, scale=stats.sem(charges))

        c1, c2, c3, c4 = st.columns(4)
        kpi(c1, f"${pop_mean:.2f}", "Population mean", "Monthly charges")
        kpi(c2, f"${means.mean():.2f}", "Mean of sample means", f"{k:,} samples", TEAL)
        kpi(c3, f"${se:.2f}", "Standard error", f"n = {n}", AMBER)
        kpi(c4, f"${lo:.2f} – ${hi:.2f}", f"{conf:.0%} confidence interval", "Mean monthly charge", RED)

        fig = px.histogram(x=means, nbins=40, title="Sampling distribution of the mean monthly charge",
                           color_discrete_sequence=[BLUE])
        fig.add_vline(x=pop_mean, line_dash="dash", line_color=RED, annotation_text="Population mean")
        fig.update_layout(xaxis_title="Sample mean ($)", yaxis_title="Frequency")
        st.plotly_chart(style_fig(fig), use_container_width=True)
        callout("As the sample size grows, the sampling distribution narrows around the population mean "
                "(Central Limit Theorem). Increase n to see the effect.")

    # ---- Hypothesis tests
    with tab2:
        alpha = st.select_slider("Significance level (α)", [0.01, 0.05, 0.10], value=0.05)

        section("1. Welch's t-test: monthly charges, churned vs retained")
        yes = data.loc[data["ChurnFlag"] == 1, "MonthlyCharges"].dropna()
        no = data.loc[data["ChurnFlag"] == 0, "MonthlyCharges"].dropna()
        t_stat, p_t = stats.ttest_ind(yes, no, equal_var=False)
        st.write("H0: the mean monthly charge is equal in both groups. H1: the means differ.")
        st.dataframe(pd.DataFrame({
            "Group": ["Churned", "Retained"], "n": [len(yes), len(no)],
            "Mean ($)": [round(yes.mean(), 2), round(no.mean(), 2)],
            "Std. dev. ($)": [round(yes.std(), 2), round(no.std(), 2)],
        }), hide_index=True)
        callout(f"t = {t_stat:.3f}, p-value = {p_t:.3g}. "
                + ("<b>Reject H0</b>: the difference is statistically significant."
                   if p_t < alpha else "<b>Fail to reject H0</b>: insufficient evidence of a difference."))

        section("2. One-sample Z-test: mean monthly charge against a reference value")
        ref = st.number_input("Reference mean ($) — illustrative benchmark, not an industry standard",
                              value=70.0, step=1.0)
        z = (charges.mean() - ref) / (charges.std(ddof=1) / np.sqrt(len(charges)))
        p_z = 2 * (1 - stats.norm.cdf(abs(z)))
        st.write(f"H0: μ = ${ref:.2f}. Sample mean = ${charges.mean():.2f}.")
        callout(f"Z = {z:.3f}, p-value = {p_z:.3g}. "
                + ("<b>Reject H0</b>." if p_z < alpha else "<b>Fail to reject H0</b>."))

        section("3. One-way ANOVA: monthly charges across contract types")
        groups = [g["MonthlyCharges"].dropna().to_numpy() for _, g in data.groupby("Contract")]
        f_stat, p_f = stats.f_oneway(*groups)
        callout(f"F = {f_stat:.3f}, p-value = {p_f:.3g}. "
                + ("<b>Reject H0</b>: at least one contract group has a different mean charge."
                   if p_f < alpha else "<b>Fail to reject H0</b>."))
        fig = px.box(data, x="Contract", y="MonthlyCharges", color="Contract",
                     title="Monthly charges by contract type")
        fig.update_layout(showlegend=False)
        st.plotly_chart(style_fig(fig, 340), use_container_width=True)
        callout("Significant results indicate association within this dataset, not causation.", warn=True)

    # ---- Correlation
    with tab3:
        method = st.radio("Method", ["pearson", "spearman"], horizontal=True)
        num = data[["tenure", "MonthlyCharges", "TotalCharges"]]
        a, b = st.columns(2)
        with a:
            fig = px.imshow(num.corr(method=method), text_auto=".2f", zmin=-1, zmax=1,
                            color_continuous_scale="RdBu_r", title=f"{method.title()} correlation matrix")
            st.plotly_chart(style_fig(fig), use_container_width=True)
        with b:
            samp = data.sample(min(1500, len(data)), random_state=RANDOM_STATE)
            fig = px.scatter(samp, x="tenure", y="TotalCharges", color="Churn", opacity=.6,
                             title="Tenure vs total charges", color_discrete_map=CHURN_COLORS)
            st.plotly_chart(style_fig(fig), use_container_width=True)
        with st.expander("Covariance matrix"):
            st.dataframe(num.cov().round(2))

    # ---- Outliers
    with tab4:
        col = st.selectbox("Variable", ["MonthlyCharges", "TotalCharges", "tenure"])
        v = data[col].dropna()
        q1, q3 = v.quantile([0.25, 0.75])
        iqr = q3 - q1
        lo_b, hi_b = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = int(((v < lo_b) | (v > hi_b)).sum())
        c1, c2, c3, c4 = st.columns(4)
        kpi(c1, f"{q1:.2f}", "Q1", "")
        kpi(c2, f"{q3:.2f}", "Q3", "", TEAL)
        kpi(c3, f"{lo_b:.2f} – {hi_b:.2f}", "IQR bounds", "1.5 × IQR rule", AMBER)
        kpi(c4, f"{n_out:,}", "Flagged records", "", RED)
        fig = px.box(data, x=col, points="outliers", title=f"Box plot: {col}", color_discrete_sequence=[BLUE])
        st.plotly_chart(style_fig(fig, 300), use_container_width=True)
        callout("Flagged values are statistical outliers, not necessarily errors. "
                "They are reviewed rather than deleted automatically.")


# =============================================================================
# PAGE 4: CHURN PREDICTION
# =============================================================================
def page_predict():
    page_header("Churn Prediction", "Enter a customer's details to estimate the probability of churn.")
    yn = ["No", "Yes"]

    with st.form("customer_form"):
        section("Customer demographics")
        c1, c2, c3, c4 = st.columns(4)
        gender = c1.selectbox("Gender", opts("gender"))
        senior = c2.selectbox("Senior citizen", yn)
        partner = c3.selectbox("Partner", opts("Partner"))
        dependents = c4.selectbox("Dependents", opts("Dependents"))

        section("Services subscribed")
        c1, c2, c3, c4 = st.columns(4)
        phone = c1.selectbox("Phone service", opts("PhoneService"))
        multi = c2.selectbox("Multiple lines", opts("MultipleLines"))
        internet = c3.selectbox("Internet service", opts("InternetService"))
        security = c4.selectbox("Online security", opts("OnlineSecurity"))
        c1, c2, c3, c4 = st.columns(4)
        backup = c1.selectbox("Online backup", opts("OnlineBackup"))
        device = c2.selectbox("Device protection", opts("DeviceProtection"))
        tech = c3.selectbox("Tech support", opts("TechSupport"))
        stv = c4.selectbox("Streaming TV", opts("StreamingTV"))
        smov = st.selectbox("Streaming movies", opts("StreamingMovies"))

        section("Account and billing")
        c1, c2, c3 = st.columns(3)
        contract = c1.selectbox("Contract", opts("Contract"))
        paperless = c2.selectbox("Paperless billing", opts("PaperlessBilling"))
        payment = c3.selectbox("Payment method", opts("PaymentMethod"))
        c1, c2, c3 = st.columns(3)
        tenure = c1.slider("Tenure (months)", 0, 72, 12)
        monthly = c2.number_input("Monthly charges ($)", 0.0, 200.0, 70.0, 0.5)
        total_in = c3.number_input("Total charges ($) — leave 0 to estimate", 0.0, 20000.0, 0.0, 10.0,
                                   help="If 0, total charges are estimated as tenure × monthly charges.")
        submitted = st.form_submit_button("Run prediction", type="primary")

    if submitted:
        total = total_in if total_in > 0 else float(tenure * monthly)
        st.session_state["last_row"] = pd.DataFrame([{
            "gender": gender, "SeniorCitizen": "1" if senior == "Yes" else "0", "Partner": partner,
            "Dependents": dependents, "tenure": tenure, "PhoneService": phone, "MultipleLines": multi,
            "InternetService": internet, "OnlineSecurity": security, "OnlineBackup": backup,
            "DeviceProtection": device, "TechSupport": tech, "StreamingTV": stv, "StreamingMovies": smov,
            "Contract": contract, "PaperlessBilling": paperless, "PaymentMethod": payment,
            "MonthlyCharges": monthly, "TotalCharges": total,
        }])

    if "last_row" not in st.session_state:
        callout("Complete the form above and select <b>Run prediction</b> to see the result.")
        return

    row = st.session_state["last_row"]
    prob = float(predict_proba(row)[0])
    band, color = risk_band(prob, low, high)
    verdict = "Predicted to churn" if prob >= threshold else "Predicted to stay"
    baseline = data["ChurnFlag"].mean()

    section("Prediction result")
    left, right = st.columns([1.1, 1])
    with left:
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=prob * 100, number={"suffix": "%", "font": {"size": 46, "color": NAVY}},
            title={"text": "Estimated churn probability"},
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": NAVY},
                   "steps": [{"range": [0, low * 100], "color": "#bbf7d0"},
                             {"range": [low * 100, high * 100], "color": "#fde68a"},
                             {"range": [high * 100, 100], "color": "#fecaca"}],
                   "threshold": {"line": {"color": "black", "width": 3}, "value": threshold * 100}},
        ))
        st.plotly_chart(style_fig(fig, 300), use_container_width=True)
    with right:
        st.markdown(f'<div class="result-box" style="background:{color}"><h3>{band} churn risk</h3>'
                    f'<p>{verdict} at a {threshold:.0%} decision threshold.</p></div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        kpi(c1, f"{prob:.1%}", "This customer", "", color)
        kpi(c2, f"{baseline:.1%}", "Dataset average", f"{(prob - baseline) * 100:+.1f} pts difference", GREY)

    section("Suggested retention actions")
    r = row.iloc[0]
    actions = []
    if r["Contract"] == "Month-to-month":
        actions.append("Offer an incentive to move to a one- or two-year contract.")
    if r["InternetService"] == "Fiber optic" and r["TechSupport"] == "No":
        actions.append("Bundle technical support with the fiber optic plan.")
    if r["OnlineSecurity"] == "No" and r["InternetService"] != "No":
        actions.append("Offer an online security trial period.")
    if r["PaymentMethod"] == "Electronic check":
        actions.append("Encourage a switch to an automatic payment method.")
    if r["tenure"] <= 12:
        actions.append("Schedule an early-tenure satisfaction check-in.")
    if actions:
        for a in actions:
            st.markdown(f"- {a}")
    else:
        st.write("No standard retention rule applies to this profile.")
    callout("These actions are rule-based suggestions for guidance. The probability is a model estimate, "
            "not a certainty, and risk bands are a display grouping rather than a calibrated business policy.",
            warn=True)

    section("Scenario analysis")
    st.caption("Change selected attributes to see how the estimated probability responds.")
    c1, c2, c3 = st.columns(3)
    contract_opts = opts("Contract")
    n_contract = c1.selectbox("Contract", contract_opts, index=contract_opts.index(r["Contract"]), key="wi_c")
    n_monthly = c2.slider("Monthly charges ($)", 0.0, 150.0, float(r["MonthlyCharges"]), 1.0, key="wi_m")
    tech_opts = opts("TechSupport")
    n_tech = c3.selectbox("Tech support", tech_opts, index=tech_opts.index(r["TechSupport"]), key="wi_t")
    scen = row.copy()
    scen["Contract"], scen["MonthlyCharges"], scen["TechSupport"] = n_contract, n_monthly, n_tech
    scen["TotalCharges"] = float(r["tenure"] * n_monthly)
    sp = float(predict_proba(scen)[0])
    c1, c2, c3 = st.columns(3)
    kpi(c1, f"{prob:.1%}", "Current probability", "")
    kpi(c2, f"{sp:.1%}", "Scenario probability", f"{(sp - prob) * 100:+.1f} pts change", TEAL)
    kpi(c3, risk_band(sp, low, high)[0], "Scenario risk band", "", risk_band(sp, low, high)[1])


# =============================================================================
# PAGE 5: BATCH SCORING
# =============================================================================
def page_batch():
    page_header("Batch Scoring", "Score many customers at once by uploading a CSV file.")
    callout(f"The file must contain these columns: {', '.join(feature_columns)}. "
            "<code>customerID</code> is optional and is kept in the output if present.")
    st.download_button("Download CSV template (sample rows)",
                       data.drop(columns=["ChurnFlag", "Churn"]).head(5).to_csv(index=False).encode(),
                       "template.csv", "text/csv")
    up = st.file_uploader("Upload customer file", type="csv")
    if up is None:
        return
    try:
        raw = pd.read_csv(up)
        miss = [c for c in feature_columns if c not in raw.columns]
        if miss:
            st.error(f"Missing required columns: {', '.join(miss)}")
            return
        w = raw.copy()
        w["TotalCharges"] = pd.to_numeric(w["TotalCharges"], errors="coerce")
        w["SeniorCitizen"] = w["SeniorCitizen"].astype(str)
        w["Churn_Probability"] = predict_proba(w).round(4)
        w["Prediction"] = np.where(w["Churn_Probability"] >= threshold, "Churn", "Stay")
        w["Risk_Band"] = w["Churn_Probability"].apply(lambda p: risk_band(p, low, high)[0])
    except Exception as ex:
        st.error(f"The file could not be scored: {ex}")
        return

    c1, c2, c3, c4 = st.columns(4)
    kpi(c1, f"{len(w):,}", "Customers scored", "")
    kpi(c2, f"{(w['Prediction'] == 'Churn').sum():,}", "Predicted to churn", "", RED)
    kpi(c3, f"{(w['Risk_Band'] == 'High').sum():,}", "High risk", "", AMBER)
    kpi(c4, f"{w['Churn_Probability'].mean():.1%}", "Mean probability", "", TEAL)

    fig = px.histogram(w, x="Churn_Probability", nbins=30, color="Risk_Band",
                       color_discrete_map={"Low": GREEN, "Medium": AMBER, "High": RED},
                       title="Distribution of churn probabilities")
    st.plotly_chart(style_fig(fig), use_container_width=True)
    st.dataframe(w.sort_values("Churn_Probability", ascending=False), height=380)
    st.download_button("Download scored results (CSV)", w.to_csv(index=False).encode(),
                       "scored_customers.csv", "text/csv", type="primary")


# =============================================================================
# PAGE 6: MODEL PERFORMANCE
# =============================================================================
def page_model():
    page_header("Model Performance", f"Held-out test evaluation of the selected model: {model_name}.")
    try:
        y_true, prob = holdout_evaluation(model, data)
    except Exception as ex:
        st.error(f"The test set could not be reconstructed: {ex}")
        return
    pred = (prob >= threshold).astype(int)

    callout("The evaluation re-creates the notebook's 80/20 stratified split (random_state = 42), so the "
            "figures below come from data the model did not train on. Metrics change with the decision "
            "threshold in the sidebar.")
    c1, c2, c3, c4, c5 = st.columns(5)
    kpi(c1, f"{accuracy_score(y_true, pred):.3f}", "Accuracy", "")
    kpi(c2, f"{precision_score(y_true, pred, zero_division=0):.3f}", "Precision", "", TEAL)
    kpi(c3, f"{recall_score(y_true, pred, zero_division=0):.3f}", "Recall", "", AMBER)
    kpi(c4, f"{f1_score(y_true, pred, zero_division=0):.3f}", "F1 score", "", RED)
    kpi(c5, f"{roc_auc_score(y_true, prob):.3f}", "ROC-AUC", "", NAVY)

    a, b = st.columns(2)
    with a:
        cm = confusion_matrix(y_true, pred)
        fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                        x=["Predicted stay", "Predicted churn"], y=["Actual stay", "Actual churn"],
                        title=f"Confusion matrix (threshold {threshold:.2f})")
        st.plotly_chart(style_fig(fig), use_container_width=True)
    with b:
        fpr, tpr, _ = roc_curve(y_true, prob)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="Model", line=dict(color=BLUE, width=3)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random guess",
                                 line=dict(color=GREY, dash="dash")))
        fig.update_layout(title="ROC curve", xaxis_title="False positive rate", yaxis_title="True positive rate")
        st.plotly_chart(style_fig(fig), use_container_width=True)

    section("Effect of the decision threshold")
    ths = np.arange(0.1, 0.91, 0.05)
    tbl = pd.DataFrame({
        "Threshold": ths.round(2),
        "Precision": [precision_score(y_true, prob >= t, zero_division=0) for t in ths],
        "Recall": [recall_score(y_true, prob >= t, zero_division=0) for t in ths],
        "F1": [f1_score(y_true, prob >= t, zero_division=0) for t in ths],
    })
    fig = px.line(tbl.melt("Threshold", var_name="Metric", value_name="Score"), x="Threshold", y="Score",
                  color="Metric", markers=True, title="Precision, recall and F1 across thresholds",
                  color_discrete_sequence=[TEAL, AMBER, RED])
    st.plotly_chart(style_fig(fig), use_container_width=True)
    callout("A lower threshold catches more churners (higher recall) but contacts more customers who would "
            "have stayed (lower precision). The right balance depends on the cost of each type of error.")

    section("Key drivers of churn")
    imp = get_importances()
    if imp is None:
        st.info("Feature importances are not available for this model type.")
    else:
        top = imp.head(15).iloc[::-1]
        fig = px.bar(top, x="Importance", y="Feature", orientation="h",
                     color="Effect" if "Effect" in top.columns else None,
                     color_discrete_map={"Increases churn risk": RED, "Decreases churn risk": TEAL},
                     title="Top 15 features by importance")
        st.plotly_chart(style_fig(fig, 480), use_container_width=True)
        st.caption("Importance describes what the model relies on; it indicates association, not causation.")


# =============================================================================
# PAGE 7: ABOUT
# =============================================================================
def page_about():
    page_header("About This Project", "Methodology, data and limitations.")
    section("Project summary")
    st.write(
        "This application is the front end of the Customer Churn Prediction & Analytics project. "
        "It presents the descriptive statistics, inferential tests and predictive model developed in the "
        "accompanying notebook, using the Telco Customer Churn dataset."
    )
    section("Methodology")
    st.markdown(
        "1. **Data cleaning:** conversion of `TotalCharges` to numeric, handling of missing values.\n"
        "2. **Exploratory analysis:** distributions, churn rates by segment, correlation analysis.\n"
        "3. **Statistical inference:** sampling distributions, confidence intervals, Welch's t-test, "
        "Z-test, one-way ANOVA and IQR-based outlier detection.\n"
        "4. **Preprocessing:** median/most-frequent imputation, standardisation of numeric features and "
        "one-hot encoding of categorical features, fitted on training data only.\n"
        "5. **Modelling:** Logistic Regression, Decision Tree and Random Forest classifiers with balanced "
        "class weights; the best model by F1 score is deployed here.\n"
        "6. **Evaluation:** accuracy, precision, recall, F1 and ROC-AUC on a stratified 20% test set."
    )
    section("Limitations")
    callout("Predicted probabilities are estimates and not certainties. Statistical tests and correlations "
            "show association, not causation. Risk bands are a display convenience, not a calibrated business "
            "policy, and the model reflects the historical dataset it was trained on.", warn=True)
    section("Technology")
    st.write("Python · pandas · scikit-learn · SciPy · Plotly · Streamlit")


# =============================================================================
# ROUTER
# =============================================================================
ROUTES = {
    "Overview": page_overview,
    "Data Explorer": page_explorer,
    "Statistical Analysis": page_stats,
    "Churn Prediction": page_predict,
    "Batch Scoring": page_batch,
    "Model Performance": page_model,
    "About": page_about,
}
ROUTES[page]()