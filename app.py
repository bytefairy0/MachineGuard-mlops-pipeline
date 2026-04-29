"""
MachineGuard — Streamlit GUI
End-to-End ML Dashboard for Predictive Maintenance
Domain: Manufacturing / Industrial Intelligence
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import joblib
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MachineGuard · MLOps Dashboard",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Barlow:wght@300;400;600;800&display=swap');

:root {
    --bg-deep:  #0b0f14;
    --bg-card:  #131920;
    --bg-panel: #1a2332;
    --accent:   #00d4aa;
    --accent2:  #ff6b35;
    --warn:     #ffd166;
    --text:     #e2e8f0;
    --muted:    #64748b;
    --border:   #1e3a5f;
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg-deep) !important;
    color: var(--text) !important;
    font-family: 'Barlow', sans-serif;
}
[data-testid="stSidebar"] {
    background: var(--bg-card) !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

h1 { font-family:'Barlow';font-weight:800;color:var(--accent) !important;letter-spacing:-1px; }
h2 { font-family:'Barlow';font-weight:800;color:var(--text) !important; }
h3 { font-family:'Barlow';font-weight:800;font-size:1rem !important;color:var(--accent) !important;text-transform:uppercase;letter-spacing:2px; }

[data-testid="metric-container"] {
    background:var(--bg-panel) !important;
    border:1px solid var(--border) !important;
    border-radius:8px !important;
    padding:12px !important;
}
[data-testid="metric-container"] label {
    color:var(--muted) !important;font-size:0.75rem !important;
    text-transform:uppercase;letter-spacing:1px;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color:var(--accent) !important;font-family:'JetBrains Mono';font-size:1.8rem !important;
}

.stButton > button {
    background:var(--accent) !important;color:#000 !important;border:none !important;
    border-radius:4px !important;font-weight:700 !important;font-family:'Barlow',sans-serif !important;
    letter-spacing:1px !important;text-transform:uppercase !important;transition:all 0.2s !important;
}
.stButton > button:hover { background:var(--accent2) !important;transform:translateY(-1px); }

.stTabs [data-baseweb="tab-list"] { background:var(--bg-card);border-bottom:1px solid var(--border);gap:0; }
.stTabs [data-baseweb="tab"] { background:transparent;color:var(--muted) !important;border:none;padding:10px 20px;font-family:'Barlow';font-weight:600; }
.stTabs [aria-selected="true"] { color:var(--accent) !important;border-bottom:2px solid var(--accent) !important; }

.info-card {
    background:var(--bg-panel);border:1px solid var(--border);
    border-radius:8px;padding:16px 20px;margin-bottom:10px;
}
.section-header {
    border-left:4px solid var(--accent);padding-left:12px;
    margin:24px 0 16px 0;font-family:'Barlow';font-weight:800;
    font-size:1.4rem;color:var(--text);
}
.badge-ok   { background:#00d4aa22;color:#00d4aa;border:1px solid #00d4aa44;padding:2px 10px;border-radius:100px;font-size:0.75rem;font-family:'JetBrains Mono'; }
.badge-warn { background:#ffd16622;color:#ffd166;border:1px solid #ffd16644;padding:2px 10px;border-radius:100px;font-size:0.75rem;font-family:'JetBrains Mono'; }
.badge-err  { background:#ff6b3522;color:#ff6b35;border:1px solid #ff6b3544;padding:2px 10px;border-radius:100px;font-size:0.75rem;font-family:'JetBrains Mono'; }
</style>
""", unsafe_allow_html=True)

# ── Paths ───────────────────────────────────────────────────────────────────────
BASE        = Path(__file__).parent
MODELS_DIR  = BASE / "models"
DATA_DIR    = BASE / "data"
OUTPUTS_DIR = BASE / "outputs"
PLOTS_DIR   = BASE / "plots"

# ── Loaders ─────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    out = {}
    for key, fname in {
        "xgb":    "xgb_classifier.pkl",
        "svm":    "svm_pipeline.pkl",
        "kmeans": "kmeans.pkl",
        "pca":    "pca.pkl",
        "scaler": "scaler.pkl",
        "le":     "label_encoder.pkl",
    }.items():
        p = MODELS_DIR / fname
        try:
            out[key] = joblib.load(p) if p.exists() else None
        except Exception:
            out[key] = None
    return out

@st.cache_data
def load_raw_data():
    p = DATA_DIR / "raw_data" / "predictive_maintenance_data.csv"
    return pd.read_csv(p) if p.exists() else None

@st.cache_data
def load_cleaned_data():
    e = DATA_DIR / "cleaned_data" / "eda_data.csv"
    c = DATA_DIR / "cleaned_data" / "clustered_data.csv"
    return (pd.read_csv(e) if e.exists() else None,
            pd.read_csv(c) if c.exists() else None)

@st.cache_data
def load_metrics():
    p = OUTPUTS_DIR / "metrics.json"
    return json.load(open(p)) if p.exists() else {}

@st.cache_data
def load_rules():
    # dict: { "Normal": [list of rule dicts], "Degraded": [...], "Critical": [...] }
    p = MODELS_DIR / "rules.json"
    return json.load(open(p)) if p.exists() else {}

@st.cache_data
def load_kb():
    # list of dicts with keys: id, failure_type, age_bin,
    # recommended_actions, part_codes, urgency_level,
    # repair_manual_id, estimated_downtime_hours
    p = MODELS_DIR / "knowledge_base.json"
    return json.load(open(p)) if p.exists() else []

def show_plot(fname, caption=""):
    p = PLOTS_DIR / fname
    if p.exists():
        st.image(str(p), caption=caption, use_container_width=True)
    else:
        st.caption(f"_(plot not found: {fname})_")

def model_badge(m, key):
    ok  = m.get(key) is not None
    cls = "badge-ok" if ok else "badge-err"
    txt = "● LOADED" if ok else "✗ MISSING"
    return f'<span class="{cls}">{txt}</span>'

# ── Feature engineering — mirrors training exactly ──────────────────────────────
def engineer_features(air_temp, process_temp, rot_speed, torque, tool_wear,
                      machine_type_str, scaler, pca_model, kmeans_model):
    """
    XGB was trained on 15 features:
    air_temperature, process_temperature, rotational_speed, torque, tool_wear,
    type_enc, temp_diff, power_w, wear_rate, torque_speed_ratio,
    high_wear_flag, thermal_overload, health_regime_enc, pc1, pc2

    Scaler was fit on the 5 raw sensor features only.
    PCA and KMeans were fit on those same 5 scaled features.
    """
    type_enc           = {"L": 0, "M": 1, "H": 2}[machine_type_str]
    temp_diff          = process_temp - air_temp
    power_w            = (2 * np.pi * rot_speed * torque) / 60.0
    wear_rate          = tool_wear / (rot_speed + 1e-6)
    torque_speed_ratio = torque    / (rot_speed + 1e-6)
    high_wear_flag     = int(tool_wear > 200)
    thermal_overload   = int(temp_diff > 8.6)

    # Scale raw 5 → used for PCA and KMeans
    raw5    = np.array([[air_temp, process_temp, rot_speed, torque, tool_wear]])
    scaled5 = scaler.transform(raw5)

    # PCA components
    pcs = pca_model.transform(scaled5)
    pc1 = float(pcs[0, 0])
    pc2 = float(pcs[0, 1]) if pcs.shape[1] > 1 else 0.0

    # Health regime from KMeans
    health_regime_enc = int(kmeans_model.predict(scaled5)[0])

    return np.array([[
        air_temp, process_temp, rot_speed, torque, tool_wear,
        type_enc, temp_diff, power_w, wear_rate, torque_speed_ratio,
        high_wear_flag, thermal_overload, health_regime_enc, pc1, pc2
    ]])

# ── Bootstrap ───────────────────────────────────────────────────────────────────
models               = load_models()
raw_df               = load_raw_data()
eda_df, clustered_df = load_cleaned_data()
metrics              = load_metrics()
rules                = load_rules()
kb                   = load_kb()

# ── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ MachineGuard")
    st.markdown('<p style="color:#64748b;font-size:0.8rem;margin-top:-8px;">Predictive Maintenance · MLOps</p>',
                unsafe_allow_html=True)
    st.divider()
    page = st.radio("Navigation", [
        "🏠  Overview",
        "📊  EDA & Data",
        "🤖  Classification",
        "📉  Regression",
        "🔵  Clustering",
        "📐  Dimensionality Reduction",
        "🔗  Association Rules",
        "📈  Time Series",
        "💡  Recommender",
        "🚀  Live Prediction",
        "🧪  Model Testing",
    ], label_visibility="collapsed")
    st.divider()
    st.markdown('<p style="color:#64748b;font-size:0.72rem;">AI221 Machine Learning · GIK Institute<br>Domain: Manufacturing Intelligence</p>',
                unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠  Overview":
    st.markdown("# ⚙️ MachineGuard")
    st.markdown("### End-to-End Predictive Maintenance · MLOps Pipeline")
    st.divider()

    c1, c2, c3, c4, c5 = st.columns(5)
    n_rows   = len(raw_df) if raw_df is not None else "–"
    n_loaded = sum(1 for v in models.values() if v is not None)
    xgb_acc  = metrics.get("xgb", {}).get("accuracy", metrics.get("accuracy", None))
    svm_acc  = metrics.get("svm", {}).get("accuracy", None)

    c1.metric("Dataset Rows",  f"{n_rows:,}" if isinstance(n_rows, int) else n_rows)
    c2.metric("Models Loaded", f"{n_loaded} / {len(models)}")
    c3.metric("XGB Accuracy",  f"{xgb_acc:.2%}" if xgb_acc else "–")
    c4.metric("SVM Accuracy",  f"{svm_acc:.2%}" if svm_acc else "–")
    c5.metric("ML Tasks",      "7")
    st.divider()

    col_a, col_b = st.columns([3, 2])
    with col_a:
        st.markdown('<div class="section-header">Project Architecture</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-card">
        <b style="color:#00d4aa">Domain:</b> Manufacturing / Industrial Intelligence<br><br>
        Full MLOps pipeline on the <b>AI4I Predictive Maintenance</b> dataset:<br><br>
        <b>→ Classification</b> — XGBoost &amp; SVM predict machine failure type<br>
        <b>→ Regression</b> — Tool wear (minutes) prediction<br>
        <b>→ Clustering</b> — K-Means operational health regimes<br>
        <b>→ Dimensionality Reduction</b> — PCA feature compression<br>
        <b>→ Association Rules</b> — Apriori failure co-occurrence patterns<br>
        <b>→ Time Series</b> — Sensor trend &amp; anomaly detection<br>
        <b>→ Recommendation</b> — Maintenance action recommender<br>
        </div>""", unsafe_allow_html=True)

        st.markdown('<div class="section-header">Model Status</div>', unsafe_allow_html=True)
        for key in models:
            st.markdown(f"&nbsp;&nbsp;`{key}` &nbsp; {model_badge(models, key)}", unsafe_allow_html=True)

    with col_b:
        st.markdown('<div class="section-header">MLOps Stack</div>', unsafe_allow_html=True)
        for tool, desc in {
            "FastAPI":        "REST inference API",
            "Prefect":        "Workflow orchestration",
            "Docker":         "Containerization",
            "GitHub Actions": "CI/CD pipeline",
            "DeepChecks":     "Automated ML testing",
            "Streamlit":      "Interactive dashboard",
        }.items():
            st.markdown(f"""
            <div class="info-card" style="padding:10px 14px;margin-bottom:6px;">
            <b style="color:#00d4aa">{tool}</b>&nbsp;&nbsp;
            <span style="color:#64748b;font-size:0.85rem">{desc}</span>
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# EDA
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊  EDA & Data":
    st.markdown("# 📊 Exploratory Data Analysis")
    df = eda_df if eda_df is not None else raw_df

    if df is None:
        st.warning("No dataset found. Place CSV in `data/raw_data/`.")
    else:
        tab1, tab2, tab3 = st.tabs(["📋 Dataset", "📈 Distributions", "🔥 Correlation"])

        with tab1:
            c1, c2, c3 = st.columns(3)
            c1.metric("Rows",    df.shape[0])
            c2.metric("Columns", df.shape[1])
            c3.metric("Missing", int(df.isnull().sum().sum()))
            st.dataframe(df.head(200), use_container_width=True)
            with st.expander("Descriptive Statistics"):
                st.dataframe(df.describe(), use_container_width=True)

        with tab2:
            num_cols = df.select_dtypes(include=np.number).columns.tolist()
            col = st.selectbox("Select feature", num_cols)
            fig = px.histogram(df, x=col, nbins=40,
                               color_discrete_sequence=["#00d4aa"],
                               template="plotly_dark")
            fig.update_layout(paper_bgcolor="#131920", plot_bgcolor="#1a2332")
            st.plotly_chart(fig, use_container_width=True)
            show_plot("01_class_distribution.png", "Class Distribution")

        with tab3:
            corr = df.select_dtypes(include=np.number).corr()
            fig2 = px.imshow(corr, color_continuous_scale="teal",
                             template="plotly_dark", text_auto=".2f")
            fig2.update_layout(paper_bgcolor="#131920")
            st.plotly_chart(fig2, use_container_width=True)
            show_plot("pre_pca_correlation.png")

# ══════════════════════════════════════════════════════════════════════════════
# CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖  Classification":
    st.markdown("# 🤖 Failure Classification")
    st.markdown("**Models:** XGBoost Classifier · SVM (SVC)")
    st.divider()

    tab1, tab2, tab3 = st.tabs(["XGBoost", "SVM", "Comparison"])

    with tab1:
        st.markdown("### XGBoost Classifier")
        c1, c2 = st.columns(2)
        with c1: show_plot("xgb_confusion_matrix.png",   "Confusion Matrix")
        with c2: show_plot("xgb_feature_importance.png", "Feature Importance")
        show_plot("04_feature_importance.png", "Full Feature Importance")
        show_plot("05_cv_scores.png",           "Cross-Validation Scores")
        xgb_m = metrics.get("xgb", {})
        if xgb_m:
            cols = st.columns(len(xgb_m))
            for i, (k, v) in enumerate(xgb_m.items()):
                cols[i].metric(k.capitalize(), f"{v:.4f}" if isinstance(v, float) else v)

    with tab2:
        st.markdown("### SVM (SVC)")
        c1, c2 = st.columns(2)
        with c1: show_plot("svm_confusion_matrix.png",  "Confusion Matrix")
        with c2: show_plot("svm_decision_boundary.png", "Decision Boundary (PCA space)")
        svm_m = metrics.get("svm", {})
        if svm_m:
            cols = st.columns(len(svm_m))
            for i, (k, v) in enumerate(svm_m.items()):
                cols[i].metric(k.capitalize(), f"{v:.4f}" if isinstance(v, float) else v)

    with tab3:
        st.markdown("### Model Comparison")
        show_plot("03_model_comparison.png",   "Accuracy / F1 Comparison")
        show_plot("02_confusion_matrices.png",  "Confusion Matrices Side-by-Side")
        show_plot("perclass_f1_comparison.png", "Per-class F1")
        show_plot("cv_comparison.png",          "CV Score Comparison")
        show_plot("test_comparison.png",        "Test Set Comparison")
        rows = []
        for mn in ["xgb", "svm"]:
            m = metrics.get(mn, {})
            if m:
                rows.append({"Model": mn.upper(), **m})
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# REGRESSION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📉  Regression":
    st.markdown("# 📉 Tool Wear Regression")
    st.markdown("Predicting **tool wear (minutes)** from sensor readings.")
    st.divider()
    c1, c2 = st.columns(2)
    with c1: show_plot("actual_vs_predicted.png", "Actual vs Predicted")
    with c2: show_plot("residual_plots.png",      "Residual Analysis")
    show_plot("learning_curve.png",  "Learning Curve")
    show_plot("Tool_wear_trend.png", "Tool Wear Trend")
    reg_m = metrics.get("regression", {})
    if reg_m:
        cols = st.columns(len(reg_m))
        for i, (k, v) in enumerate(reg_m.items()):
            cols[i].metric(k.upper(), f"{v:.4f}" if isinstance(v, float) else v)
    else:
        st.info("Add regression metrics to `outputs/metrics.json` under key `'regression'`.")

# ══════════════════════════════════════════════════════════════════════════════
# CLUSTERING
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔵  Clustering":
    st.markdown("# 🔵 K-Means Clustering")
    st.markdown("Operational health regime discovery.")
    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        show_plot("elbow.png",           "Elbow Method")
        show_plot("silhouette_plot.png", "Silhouette Plot")
    with c2:
        show_plot("cluster_pca_scatter.png", "Cluster PCA Scatter")
        show_plot("silhouette_scores.png",   "Silhouette Scores per K")
    show_plot("cluster_sensor_profiles.png", "Cluster Sensor Profiles")
    show_plot("failure_type_per_regime.png",  "Failure Type per Regime")
    if clustered_df is not None and "Cluster" in clustered_df.columns:
        num = clustered_df.select_dtypes(include=np.number).columns.tolist()
        if len(num) >= 2:
            fig = px.scatter(clustered_df, x=num[0], y=num[1], color="Cluster",
                             color_discrete_sequence=px.colors.qualitative.Set2,
                             template="plotly_dark")
            fig.update_layout(paper_bgcolor="#131920", plot_bgcolor="#1a2332")
            st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PCA
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📐  Dimensionality Reduction":
    st.markdown("# 📐 PCA — Dimensionality Reduction")
    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        show_plot("pca_scree.png",            "Scree Plot")
        show_plot("pca_loadings.png",         "PCA Loadings")
    with c2:
        show_plot("pca_2d_scatter.png",       "2D PCA Scatter")
        show_plot("06_pc_decision_space.png", "Decision Boundaries in PC Space")

    if models["pca"] is not None:
        with st.expander("Live PCA Details from pca.pkl"):
            try:
                pca = models["pca"]
                ev  = pca.explained_variance_ratio_
                cum = np.cumsum(ev)
                fig = go.Figure()
                fig.add_bar(x=[f"PC{i+1}" for i in range(len(ev))], y=ev,
                            name="Explained Variance", marker_color="#00d4aa")
                fig.add_scatter(x=[f"PC{i+1}" for i in range(len(cum))], y=cum,
                                name="Cumulative", line=dict(color="#ff6b35", width=2))
                fig.update_layout(template="plotly_dark", paper_bgcolor="#131920",
                                  plot_bgcolor="#1a2332", title="PCA Explained Variance")
                st.plotly_chart(fig, use_container_width=True)
                c1, c2 = st.columns(2)
                c1.metric("Components",              pca.n_components_)
                c2.metric("Total Variance Explained", f"{cum[-1]:.2%}")
            except Exception as e:
                st.warning(f"Could not extract PCA info: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# ASSOCIATION RULES
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔗  Association Rules":
    st.markdown("# 🔗 Association Rules")
    st.markdown("Apriori failure co-occurrence patterns, grouped by **machine health regime**.")
    st.divider()
    show_plot("association.png", "Association Rule Network")

    if not rules:
        st.info("No rules found at `models/rules.json`.")
    else:
        # rules = { "Normal": [list of dicts], "Degraded": [...], "Critical": [...] }
        regime       = st.selectbox("Health Regime", list(rules.keys()))
        regime_rules = rules[regime]          # list of rule dicts

        c1, c2 = st.columns(2)
        min_conf = c1.slider("Min Confidence", 0.0, 1.0,   0.5,  0.05)
        min_lift = c2.slider("Min Lift",       0.0, 100.0, 1.0,  1.0)

        filtered = [
            r for r in regime_rules
            if r.get("confidence", 0) >= min_conf
            and r.get("lift", 0) >= min_lift
        ]

        st.markdown(f"**{len(filtered)} rules** match for regime `{regime}`")

        if filtered:
            rows = []
            for r in filtered:
                ants = r.get("antecedents", [])
                cons = r.get("consequents", [])
                rows.append({
                    "Antecedents": " + ".join(ants) if isinstance(ants, list) else str(ants),
                    "Consequents": " + ".join(cons) if isinstance(cons, list) else str(cons),
                    "Support":     round(r.get("support",     0), 5),
                    "Confidence":  round(r.get("confidence",  0), 4),
                    "Lift":        round(r.get("lift",        0), 2),
                    "Zhang's":     round(r.get("zhangs_metric", 0), 4),
                })
            rules_df = pd.DataFrame(rows).sort_values("Lift", ascending=False)
            st.dataframe(rules_df, use_container_width=True)

            fig = px.scatter(rules_df, x="Support", y="Confidence",
                             size="Lift", color="Lift",
                             hover_data=["Antecedents", "Consequents"],
                             color_continuous_scale="teal",
                             template="plotly_dark",
                             title=f"Support vs Confidence — {regime} regime")
            fig.update_layout(paper_bgcolor="#131920", plot_bgcolor="#1a2332")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No rules match filters. Try lowering the thresholds.")

# ══════════════════════════════════════════════════════════════════════════════
# TIME SERIES
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈  Time Series":
    st.markdown("# 📈 Time Series Analysis")
    st.markdown("Sensor trend monitoring, anomaly detection, and pre-failure patterns.")
    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        show_plot("Tool_wear_trend.png",           "Tool Wear Trend")
        show_plot("pre_failure_Spike_pattern.png", "Pre-Failure Spike Patterns")
    with c2:
        show_plot("Torque_Shock_pattern.png",      "Torque Shock Patterns")

    df = eda_df if eda_df is not None else raw_df
    if df is not None:
        st.markdown("### Interactive Sensor Plot")
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        cols_sel = st.multiselect("Select sensors", num_cols,
                                  default=num_cols[:2] if len(num_cols) >= 2 else num_cols)
        n_pts = st.slider("Samples to show", 100, min(5000, len(df)), 500, 100)
        if cols_sel:
            colors = ["#00d4aa","#ff6b35","#ffd166","#a78bfa","#38bdf8"]
            fig = go.Figure()
            for i, c in enumerate(cols_sel):
                fig.add_scatter(y=df[c].values[:n_pts], name=c,
                                line=dict(color=colors[i % len(colors)], width=1.5))
            fig.update_layout(template="plotly_dark", paper_bgcolor="#131920",
                              plot_bgcolor="#1a2332",
                              xaxis_title="Sample Index", yaxis_title="Value")
            st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# RECOMMENDER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "💡  Recommender":
    st.markdown("# 💡 MachineGuard Recommender")
    st.markdown("Knowledge-base maintenance action recommendations per failure type and machine age.")
    st.divider()

    if not kb:
        st.warning("Knowledge base not found at `models/knowledge_base.json`.")
    else:
        # kb is a list of dicts:
        # { id, failure_type, age_bin, recommended_actions[], part_codes[],
        #   urgency_level, repair_manual_id, estimated_downtime_hours }
        failure_types = ["All"] + sorted(set(i.get("failure_type","?") for i in kb))
        age_bins      = ["All"] + sorted(set(i.get("age_bin",     "?") for i in kb))

        c1, c2 = st.columns(2)
        sel_fail = c1.selectbox("Failure Type", failure_types)
        sel_age  = c2.selectbox("Machine Age",  age_bins)

        filtered_kb = [
            i for i in kb
            if (sel_fail == "All" or i.get("failure_type") == sel_fail)
            and (sel_age == "All" or i.get("age_bin")      == sel_age)
        ]

        st.markdown(f"**{len(filtered_kb)} recommendation(s)** found")
        st.divider()

        urgency_color = {1: "#00d4aa", 2: "#ffd166", 3: "#ff6b35"}
        urgency_label = {1: "LOW",     2: "MEDIUM",  3: "HIGH"}

        for item in filtered_kb:
            urg     = item.get("urgency_level", 1)
            color   = urgency_color.get(urg, "#64748b")
            ulabel  = urgency_label.get(urg, str(urg))
            actions = item.get("recommended_actions", [])
            parts   = item.get("part_codes", [])
            dt      = item.get("estimated_downtime_hours", "–")
            manual  = item.get("repair_manual_id", "–")

            actions_html = "".join(f"<li style='margin-bottom:4px'>{a}</li>" for a in actions)
            parts_html   = " &nbsp;".join(
                f'<code style="background:#0b0f14;padding:2px 6px;border-radius:4px">{p}</code>'
                for p in parts
            )

            st.markdown(f"""
            <div class="info-card" style="border-color:{color}44">
              <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px">
                <span style="font-size:1.1rem;font-weight:700;color:{color}">
                  [{item.get('id')}] {item.get('failure_type')} — {item.get('age_bin')} Machine
                </span>
                <span style="background:{color}22;color:{color};border:1px solid {color}44;
                             padding:2px 12px;border-radius:100px;font-size:0.75rem;
                             font-family:'JetBrains Mono'">URGENCY: {ulabel}</span>
              </div>

              <b style="color:#64748b;font-size:0.8rem;text-transform:uppercase;letter-spacing:1px">
                Recommended Actions
              </b>
              <ol style="color:#e2e8f0;margin-top:6px;padding-left:20px">{actions_html}</ol>

              <div style="margin-top:10px;display:flex;gap:24px;flex-wrap:wrap">
                <span><b style="color:#64748b">Parts:</b> {parts_html if parts else "–"}</span>
                <span><b style="color:#64748b">Est. Downtime:</b>
                  <span style="color:{color};font-family:'JetBrains Mono'">{dt}h</span></span>
                <span><b style="color:#64748b">Manual:</b>
                  <code style="color:#a78bfa">{manual}</code></span>
              </div>
            </div>""", unsafe_allow_html=True)

        with st.expander("📋 Summary Table"):
            tbl = pd.DataFrame([{
                "ID":           i.get("id"),
                "Failure Type": i.get("failure_type"),
                "Age Bin":      i.get("age_bin"),
                "Urgency":      urgency_label.get(i.get("urgency_level"), "–"),
                "Downtime (h)": i.get("estimated_downtime_hours"),
                "Manual":       i.get("repair_manual_id"),
                "Parts":        ", ".join(i.get("part_codes", [])),
            } for i in filtered_kb])
            st.dataframe(tbl, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# LIVE PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🚀  Live Prediction":
    st.markdown("# 🚀 Live Prediction")
    st.markdown("""
    Enter the **5 raw sensor values**. The app automatically computes all 10 engineered features
    (temp_diff, power_w, wear_rate, torque_speed_ratio, flags, PCA components, health regime)
    exactly as done during training, then feeds all 15 features to XGBoost or SVM.
    """)
    st.divider()

    # Guard: need scaler + pca + kmeans to engineer features
    required = ["xgb", "scaler", "pca", "kmeans"]
    missing  = [k for k in required if models.get(k) is None]
    if missing:
        st.error(f"Cannot run prediction — missing models: {missing}. Check your `models/` directory.")
        st.stop()

    tab1, tab2 = st.tabs(["Manual Input", "CSV Batch"])

    with tab1:
        st.markdown("### Enter Sensor Readings")
        c1, c2, c3 = st.columns(3)
        with c1:
            air_temp     = st.number_input("Air Temperature [K]",     290.0, 320.0, 300.0, 0.1)
            process_temp = st.number_input("Process Temperature [K]", 300.0, 340.0, 310.0, 0.1)
        with c2:
            rot_speed    = st.number_input("Rotational Speed [rpm]",  1000,  3000,  1500,  10)
            torque       = st.number_input("Torque [Nm]",             3.0,   80.0,  40.0,  0.5)
        with c3:
            tool_wear    = st.number_input("Tool Wear [min]",         0,     300,   100,   1)
            machine_type = st.selectbox("Machine Type", ["L", "M", "H"])

        model_choice = st.radio("Model", ["XGBoost", "SVM"], horizontal=True)

        # Preview engineered features
        with st.expander("👁 Preview all 15 engineered features"):
            temp_diff          = process_temp - air_temp
            power_w            = (2 * np.pi * rot_speed * torque) / 60.0
            wear_rate          = tool_wear / (rot_speed + 1e-6)
            torque_speed_ratio = torque / (rot_speed + 1e-6)
            st.json({
                "1_air_temperature":     air_temp,
                "2_process_temperature": process_temp,
                "3_rotational_speed":    rot_speed,
                "4_torque":              torque,
                "5_tool_wear":           tool_wear,
                "6_type_enc":            {"L":0,"M":1,"H":2}[machine_type],
                "7_temp_diff":           round(temp_diff, 4),
                "8_power_w":             round(power_w, 4),
                "9_wear_rate":           round(wear_rate, 6),
                "10_torque_speed_ratio": round(torque_speed_ratio, 6),
                "11_high_wear_flag":     int(tool_wear > 200),
                "12_thermal_overload":   int(temp_diff > 8.6),
                "13_health_regime_enc":  "→ from KMeans on scaled inputs",
                "14_pc1":                "→ from PCA on scaled inputs",
                "15_pc2":                "→ from PCA on scaled inputs",
            })

        if st.button("▶  Run Prediction"):
            try:
                features = engineer_features(
                    air_temp, process_temp, rot_speed, torque, tool_wear,
                    machine_type,
                    models["scaler"], models["pca"], models["kmeans"]
                )

                chosen = models["xgb"] if model_choice == "XGBoost" else models["svm"]
                if chosen is None:
                    st.error(f"{model_choice} not loaded.")
                    st.stop()

                pred  = chosen.predict(features)
                label = pred[0]
                if models["le"] is not None:
                    try:
                        label = models["le"].inverse_transform([int(label)])[0]
                    except Exception:
                        pass

                # Probabilities
                try:
                    proba   = chosen.predict_proba(features)[0]
                    classes = list(models["le"].classes_) if models["le"] is not None \
                              else list(range(len(proba)))
                    fig = px.bar(pd.DataFrame({"Class": classes, "Probability": proba}),
                                 x="Class", y="Probability",
                                 color="Probability", color_continuous_scale="teal",
                                 template="plotly_dark", title="Class Probabilities")
                    fig.update_layout(paper_bgcolor="#131920", plot_bgcolor="#1a2332")
                    st.plotly_chart(fig, use_container_width=True)
                except Exception:
                    pass

                is_failure = str(label).lower() not in ["no failure", "0", "normal"]
                color = "#ff6b35" if is_failure else "#00d4aa"
                st.markdown(f"""
                <div style="background:#1a2332;border:2px solid {color};border-radius:8px;
                            padding:24px;text-align:center;margin-top:16px;">
                  <p style="color:#64748b;text-transform:uppercase;letter-spacing:2px;
                            font-size:0.8rem;margin:0">Prediction ({model_choice})</p>
                  <p style="color:{color};font-size:2.4rem;font-weight:800;margin:8px 0;
                            font-family:'JetBrains Mono'">{label}</p>
                </div>""", unsafe_allow_html=True)

                # Pull matching KB recommendation
                if kb and str(label).lower() not in ["no failure", "0", "normal"]:
                    match = next((i for i in kb if i.get("failure_type") == str(label)), None)
                    if match:
                        urg    = match.get("urgency_level", 1)
                        uc     = {1:"#00d4aa",2:"#ffd166",3:"#ff6b35"}.get(urg,"#64748b")
                        a_html = "".join(f"<li>{a}</li>" for a in match.get("recommended_actions",[]))
                        st.markdown(f"""
                        <div class="info-card" style="border-color:{uc}44;margin-top:12px">
                          <b style="color:{uc}">💡 Recommended Actions — {match.get('failure_type')}</b>
                          <ol style="color:#e2e8f0;margin-top:8px">{a_html}</ol>
                          <span style="color:#64748b">Est. downtime: </span>
                          <span style="color:{uc};font-family:'JetBrains Mono'">{match.get('estimated_downtime_hours')}h</span>
                          &nbsp;&nbsp;
                          <span style="color:#64748b">Manual: </span>
                          <code style="color:#a78bfa">{match.get('repair_manual_id')}</code>
                        </div>""", unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.exception(e)

    with tab2:
        st.markdown("### Batch Prediction via CSV")
        st.markdown("""Upload a CSV with columns matching the original dataset:
        `Air temperature [K]`, `Process temperature [K]`, `Rotational speed [rpm]`,
        `Torque [Nm]`, `Tool wear [min]`, `Type`""")
        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded:
            batch_df     = pd.read_csv(uploaded)
            model_choice2 = st.radio("Model", ["XGBoost","SVM"], horizontal=True, key="b2")
            st.dataframe(batch_df.head(), use_container_width=True)

            if st.button("▶  Run Batch Prediction"):
                chosen = models["xgb"] if model_choice2 == "XGBoost" else models["svm"]
                if chosen is None:
                    st.error("Model not loaded.")
                else:
                    try:
                        preds = []
                        for _, row in batch_df.iterrows():
                            at = float(row.get("Air temperature [K]",     300))
                            pt = float(row.get("Process temperature [K]", 310))
                            rs = float(row.get("Rotational speed [rpm]",  1500))
                            tq = float(row.get("Torque [Nm]",             40))
                            tw = float(row.get("Tool wear [min]",         100))
                            tp = str(row.get("Type","M"))[0].upper()
                            if tp not in ["L","M","H"]:
                                tp = "M"
                            feats = engineer_features(at, pt, rs, tq, tw, tp,
                                                      models["scaler"], models["pca"], models["kmeans"])
                            p = chosen.predict(feats)[0]
                            if models["le"] is not None:
                                try:
                                    p = models["le"].inverse_transform([int(p)])[0]
                                except Exception:
                                    pass
                            preds.append(p)

                        batch_df["Prediction"] = preds
                        st.dataframe(batch_df, use_container_width=True)
                        st.download_button("⬇  Download Results",
                                           batch_df.to_csv(index=False).encode(),
                                           "predictions.csv", "text/csv")
                    except Exception as e:
                        st.error(f"Batch error: {e}")
                        st.exception(e)

# ══════════════════════════════════════════════════════════════════════════════
# MODEL TESTING
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🧪  Model Testing":
    st.markdown("# 🧪 Automated Model Testing")
    st.divider()

    dc_path = OUTPUTS_DIR / "deepchecks_report.json"
    if dc_path.exists():
        dc = json.load(open(dc_path))
        st.markdown("### DeepChecks Report")
        if isinstance(dc, dict):
            for name, result in dc.items():
                status = result.get("status","unknown") if isinstance(result,dict) else "unknown"
                cls    = "badge-ok" if status=="passed" else ("badge-err" if status=="failed" else "badge-warn")
                detail = result.get("details","") if isinstance(result,dict) else ""
                st.markdown(f"""
                <div class="info-card">
                  <b>{name}</b>&nbsp;&nbsp;<span class="{cls}">{status.upper()}</span>
                  {"<br><span style='color:#64748b;font-size:0.85rem'>"+detail+"</span>" if detail else ""}
                </div>""", unsafe_allow_html=True)
        else:
            st.json(dc)
    else:
        st.info("No DeepChecks report at `outputs/deepchecks_report.json`. Run `scripts/deepchecks_runner.py`.")

    st.divider()
    st.markdown("### Metrics (`outputs/metrics.json`)")
    if metrics:
        st.json(metrics)
    else:
        st.info("No metrics found.")

    st.divider()
    st.markdown("### Test Suite")
    st.markdown("""
    <div class="info-card">
    📄 <b>tests/test_api_main.py</b> — FastAPI endpoint tests<br>
    📄 <b>tests/test_main_pipeline.py</b> — Prefect pipeline unit tests<br>
    📄 <b>tests/conftest.py</b> — Shared fixtures
    </div>""", unsafe_allow_html=True)
    st.code("pytest tests/ -v --tb=short", language="bash")