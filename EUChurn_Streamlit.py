# ================= IMPORTS =================
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="European Bank Churn Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================= BRAND COLORS =================
PRIMARY = "#0A3D91"
SECONDARY = "#14B8A6"
ACCENT = "#F59E0B"
BG = "#F4F8FC"
CARD_BG = "#FFFFFF"
TEXT = "#1F2937"
MUTED = "#6B7280"
BORDER = "#D9E6F2"

PLOTLY_SEQ = [PRIMARY, SECONDARY, ACCENT]
MPL_SEQ = [PRIMARY, SECONDARY, ACCENT]
SHAP_SEQ = [PRIMARY, SECONDARY, ACCENT]

# ================= GLOBAL STYLE =================
st.markdown(f"""
<style>
    .stApp {{
        background: linear-gradient(180deg, {BG} 0%, #ffffff 100%);
        color: {TEXT};
    }}

    h1, h2, h3, h4 {{
        color: {PRIMARY} !important;
        font-weight: 800 !important;
    }}

    .main-header {{
        background: linear-gradient(135deg, {PRIMARY}, #123B6F);
        padding: 24px 26px;
        border-radius: 20px;
        color: white;
        box-shadow: 0 12px 30px rgba(10, 61, 145, 0.18);
        margin-bottom: 18px;
        border: 1px solid rgba(255,255,255,0.08);
    }}

    .main-header h1 {{
        color: white !important;
        margin: 0;
        font-size: 2.0rem;
    }}

    .metric-card {{
        background: {CARD_BG};
        padding: 18px 16px;
        border-radius: 18px;
        border: 1px solid {BORDER};
        box-shadow: 0 8px 22px rgba(15, 23, 42, 0.07);
        text-align: center;
        transition: all 0.2s ease;
        min-height: 120px;
    }}

    .metric-card:hover {{
        transform: translateY(-2px);
        box-shadow: 0 12px 28px rgba(15, 23, 42, 0.10);
    }}

    .metric-title {{
        font-size: 0.82rem;
        color: {MUTED};
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 10px;
    }}

    .metric-value {{
        font-size: 1.8rem;
        font-weight: 800;
        color: {PRIMARY};
        line-height: 1.1;
        margin-bottom: 6px;
    }}

    .metric-subtitle {{
        font-size: 0.84rem;
        color: {MUTED};
    }}

    .section-card {{
        background: {CARD_BG};
        border-radius: 20px;
        padding: 18px 18px 10px 18px;
        border: 1px solid {BORDER};
        box-shadow: 0 8px 22px rgba(15, 23, 42, 0.06);
        margin-bottom: 18px;
    }}

    .stSidebar {{
        background: linear-gradient(180deg, #F8FBFF 0%, #EEF5FF 100%);
    }}

    .stButton > button {{
        background: linear-gradient(135deg, {PRIMARY}, {SECONDARY});
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.55rem 1rem;
        font-weight: 700;
        box-shadow: 0 8px 18px rgba(10, 61, 145, 0.18);
    }}

    .stButton > button:hover {{
        opacity: 0.95;
        transform: translateY(-1px);
    }}

    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
        background: #EEF5FF;
        padding: 6px;
        border-radius: 16px;
    }}

    .stTabs [data-baseweb="tab"] {{
        background: transparent;
        border-radius: 12px;
        padding: 10px 16px;
        font-weight: 700;
        color: #4B5563;
    }}

    .stTabs [aria-selected="true"] {{
        background: white;
        color: {PRIMARY};
        box-shadow: 0 6px 16px rgba(10, 61, 145, 0.10);
    }}

    .tab-caption {{
        color: {MUTED};
        font-size: 0.95rem;
        margin-top: -4px;
        margin-bottom: 10px;
    }}

    .sidebar-block {{
        background: rgba(255,255,255,0.72);
        border: 1px solid {BORDER};
        border-radius: 16px;
        padding: 14px 14px 8px 14px;
        box-shadow: 0 8px 22px rgba(15, 23, 42, 0.05);
        margin-bottom: 14px;
    }}

    .sidebar-label {{
        font-size: 0.86rem;
        font-weight: 800;
        color: {PRIMARY};
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 8px;
    }}
</style>
""", unsafe_allow_html=True)

# ================= HELPERS =================
CATEGORICAL_COLS = ["Geography", "Gender"]
BASE_NUMERIC_COLS = [
    "CreditScore",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
]
ENGINEERED_COLS = [
    "Balance_Salary_Ratio",
    "Product_Density",
    "Engagement_Product",
    "Age_Tenure_Interaction",
]


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Balance_Salary_Ratio"] = out["Balance"] / (out["EstimatedSalary"] + 1)
    out["Product_Density"] = out["NumOfProducts"] / (out["Tenure"] + 1)
    out["Engagement_Product"] = out["IsActiveMember"] * out["NumOfProducts"]
    out["Age_Tenure_Interaction"] = out["Age"] * out["Tenure"]
    return out


def prepare_model_frame(raw_df: pd.DataFrame) -> pd.DataFrame:
    data = raw_df.copy()
    data = data.drop(columns=["CustomerId", "Surname"], errors="ignore")
    data = create_features(data)
    return data


def risk_category(score: float) -> str:
    if score < 30:
        return "Low Risk"
    elif score < 60:
        return "Medium Risk"
    return "High Risk"


def align_to_model_columns(df: pd.DataFrame, model_columns: list) -> pd.DataFrame:
    return df.reindex(columns=model_columns, fill_value=0)


def get_feature_names_from_preprocessor(preprocessor) -> list:
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        names = []
        try:
            names.extend(list(preprocessor.transformers_[0][2]))
        except Exception:
            names.extend(BASE_NUMERIC_COLS + ENGINEERED_COLS)

        try:
            cat_encoder = preprocessor.named_transformers_["cat"]
            try:
                names.extend(list(cat_encoder.get_feature_names_out(CATEGORICAL_COLS)))
            except Exception:
                try:
                    names.extend(list(cat_encoder.get_feature_names(CATEGORICAL_COLS)))
                except Exception:
                    names.extend(CATEGORICAL_COLS)
        except Exception:
            pass

        return names


@st.cache_resource
def safe_load_model(path: str):
    try:
        import sklearn.compose._column_transformer as ct
        if not hasattr(ct, "_RemainderColsList"):
            ct._RemainderColsList = list
    except Exception:
        pass
    return joblib.load(path)


def parse_float(value, default):
    try:
        return float(str(value).replace(",", "").strip())
    except Exception:
        return float(default)


def severity_style(risk_score: float):
    if risk_score < 30:
        return {"bar": "#22C55E", "steps": ["#EAFBF0", "#CFF6DE", "#A8E9BE"]}  # Green
    elif risk_score < 60:
        return {"bar": "#F59E0B", "steps": ["#FFF7E6", "#FFE9B8", "#FFD98D"]}  # Orange
    return {"bar": "#EF4444", "steps": ["#FDECEC", "#F8B4B4", "#EF4444"]}      # Red


def metric_card(title, value, subtitle="", accent=PRIMARY):
    st.markdown(f"""
    <div class="metric-card" style="border-left: 6px solid {accent};">
        <div class="metric-title">{title}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-subtitle">{subtitle}</div>
    </div>
    """, unsafe_allow_html=True)


def tab_caption(text: str):
    st.markdown(f'<div class="tab-caption">{text}</div>', unsafe_allow_html=True)


def local_shap_barplot(model, one_row_df):
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    x_proc = preprocessor.transform(one_row_df)
    if hasattr(x_proc, "toarray"):
        x_proc = x_proc.toarray()

    feature_names = get_feature_names_from_preprocessor(preprocessor)

    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(x_proc)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    shap_values = np.array(shap_values)
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]

    local_vals = shap_values[0] if shap_values.ndim == 2 else shap_values

    shap_df = pd.DataFrame({
        "Feature": feature_names[:len(local_vals)],
        "SHAP Value": local_vals
    })
    shap_df["Abs"] = shap_df["SHAP Value"].abs()
    shap_df = shap_df.sort_values("Abs", ascending=False).head(10)

    fig, ax = plt.subplots(figsize=(8, 6))
    ordered = shap_df.sort_values("SHAP Value")
    colors = [SECONDARY if v < 0 else PRIMARY for v in ordered["SHAP Value"]]

    sns.barplot(
        data=ordered,
        x="SHAP Value",
        y="Feature",
        ax=ax,
        palette=colors
    )
    ax.axvline(0, color=MUTED, linewidth=1)
    ax.set_title("Local SHAP Explanation for Selected Customer", fontsize=13, fontweight="bold")
    ax.set_xlabel("SHAP Value")
    ax.set_ylabel("")
    plt.tight_layout()
    return fig


def local_pdp_curve(model, base_raw_row, feature_name, model_feature_frame, grid_size=30):
    lo = float(model_feature_frame[feature_name].quantile(0.01))
    hi = float(model_feature_frame[feature_name].quantile(0.99))

    if lo == hi:
        lo = float(model_feature_frame[feature_name].min())
        hi = float(model_feature_frame[feature_name].max())

    grid = np.linspace(lo, hi, grid_size)
    preds = []

    for value in grid:
        temp = base_raw_row.copy()
        temp.iloc[0, temp.columns.get_loc(feature_name)] = value
        temp = create_features(temp)
        temp = align_to_model_columns(temp, MODEL_INPUT_COLS)
        preds.append(model.predict_proba(temp)[0][1])

    return grid, np.array(preds)


def metric_value_fmt(metric_name, value):
    if metric_name in ["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"]:
        return f"{value:.3f}"
    return f"{value:.2f}"


# ================= HEADER =================
col_logo, col_title = st.columns([1.25, 5])
with col_logo:
    try:
        st.image("EULOGO.png", width=190)
    except Exception:
        st.info("EU Logo")

with col_title:
    st.markdown(
        """
        <div class="main-header">
            <h1>Predictive Modeling and Risk Scoring for Bank Customer Churn</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ================= LOAD DATA / MODELS =================
try:
    rf_model_pipeline = safe_load_model("rf_churn_prediction_model.pkl")
except Exception as e:
    st.error(f"Could not load Random Forest model: {e}")
    st.stop()

try:
    gb_model_pipeline = safe_load_model("gb_churn_prediction_model.pkl")
except Exception as e:
    st.warning(f"Gradient Boosting model not loaded: {e}")
    gb_model_pipeline = None

try:
    df = pd.read_csv("European_Bank_Data.csv")
except Exception as e:
    st.error(f"Could not load dataset: {e}")
    st.stop()

reference_df = prepare_model_frame(df)
MODEL_INPUT_COLS = [c for c in reference_df.columns if c != "Exited"]
X_model = reference_df.drop("Exited", axis=1)
X_model = align_to_model_columns(X_model, MODEL_INPUT_COLS)
y_model = reference_df["Exited"]

# ================= SIDEBAR =================
try:
    st.sidebar.image("UMlogo.png", width=150)
except Exception:
    st.sidebar.info("Mentor Logo")

st.sidebar.markdown("### Input Details")
st.sidebar.caption("Enter customer details to estimate churn risk.")

with st.sidebar.expander("Customer Profile", expanded=True):
    geography = st.selectbox("Geography", ["France", "Spain", "Germany"])
    gender = st.radio("Gender", ["Male", "Female"], horizontal=True)

with st.sidebar.expander("Financial Details", expanded=True):
    credit_score = st.slider("Credit Score", 300, 900, 650)
    balance_txt = st.text_input("Balance", "200000")
    estimated_salary_txt = st.text_input("Estimated Salary", "500000")

with st.sidebar.expander("Relationship Details", expanded=True):
    age = st.slider("Age", 18, 80, 35)
    tenure = st.slider("Tenure", 0, 10, 3)
    num_products = st.selectbox("Number of Products", [1, 2, 3, 4], index=1)
    has_cr_card = st.radio("Has Credit Card (0 = No, 1 = Yes)", [0, 1], horizontal=True)
    is_active_member = st.radio("Active Member (0 = No, 1 = Yes)", [0, 1], horizontal=True)

with st.sidebar.expander("Retention Preferences", expanded=False):
    preferred_channels = st.multiselect(
        "Preferred Retention Channels",
        ["Email", "SMS", "Phone"],
        default=["Email"]
    )

input_raw_df = pd.DataFrame([{
    "CreditScore": int(credit_score),
    "Geography": geography,
    "Gender": gender,
    "Age": int(age),
    "Tenure": int(tenure),
    "Balance": parse_float(balance_txt, 200000.0),
    "NumOfProducts": int(num_products),
    "HasCrCard": int(has_cr_card),
    "IsActiveMember": int(is_active_member),
    "EstimatedSalary": parse_float(estimated_salary_txt, 500000.0),
}])

input_raw_df = create_features(input_raw_df)
input_df = align_to_model_columns(input_raw_df, MODEL_INPUT_COLS)

# ================= MODEL SELECT =================
st.markdown("### Model Selection")
model_choice = st.selectbox(
    "Choose the model to explore",
    ["Random Forest", "Gradient Boosting"],
    index=0
)

if model_choice == "Random Forest":
    selected_model = rf_model_pipeline
    selected_model_name = "RF"
else:
    if gb_model_pipeline is None:
        st.warning("Gradient Boosting model file was not loaded.")
        st.stop()
    selected_model = gb_model_pipeline
    selected_model_name = "GB"

# ================= CORE VALUES =================
prob = selected_model.predict_proba(input_df)[0][1]
risk_score = prob * 100
risk = risk_category(risk_score)
retention_rate = (1 - prob) * 100

# ================= DASHBOARD =================
def render_dashboard(model, model_name, prob, risk_score, risk, retention_rate):
    tabs = st.tabs([
        "Overview",
        "Segmentation",
        "Performance",
        "Explainability",
        "PDP",
        "Actions"
    ])

    # ================= OVERVIEW TAB =================
    with tabs[0]:
        tab_caption("A quick snapshot of the selected customer's churn position, with the score, gauge, and portfolio distribution shown together.")

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            metric_card("Churn Probability", f"{prob:.2f}", "Model output", PRIMARY)
        with c2:
            metric_card("Risk Score", f"{risk_score:.1f}", "Scale: 0–100", SECONDARY)
        with c3:
            metric_card("Risk Category", risk, "Customer segment", ACCENT)
        with c4:
            metric_card("Retention Rate", f"{retention_rate:.1f}%", "Estimated retention", SECONDARY)

        col4, col5 = st.columns(2)

        with col4:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.subheader("Risk Gauge")
            st.caption("Green, orange, and red indicate increasing churn risk.")
            style = severity_style(risk_score)
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=risk_score,
                number={"suffix": "%"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": style["bar"]},
                    "steps": [
                        {"range": [0, 30], "color": style["steps"][0]},
                        {"range": [30, 60], "color": style["steps"][1]},
                        {"range": [60, 100], "color": style["steps"][2]},
                    ],
                },
            ))
            fig.update_layout(
                margin=dict(l=20, r=20, t=25, b=10),
                paper_bgcolor="white",
                plot_bgcolor="white",
                font=dict(color=TEXT),
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col5:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.subheader("Probability Distribution")
            st.caption("The selected customer is compared against the model’s full probability distribution.")
            try:
                probs = model.predict_proba(X_model)[:, 1]
                fig1, ax1 = plt.subplots(figsize=(8, 5))
                sns.histplot(
                    probs,
                    bins=30,
                    kde=True,
                    ax=ax1,
                    color=SECONDARY,
                    edgecolor="white"
                )
                ax1.axvline(prob, color=ACCENT, linewidth=2, linestyle="--", label="Selected customer")
                ax1.set_xlabel("Churn Probability")
                ax1.set_ylabel("Customers")
                ax1.set_title("Distribution of Churn Probabilities")
                ax1.legend(frameon=False)
                st.pyplot(fig1)
                plt.close(fig1)
            except Exception as e:
                st.error(f"Probability distribution error: {e}")
            st.markdown('</div>', unsafe_allow_html=True)

    # ================= SEGMENTATION TAB =================
    with tabs[1]:
        tab_caption("A segmentation view that combines the model distribution pie with the most important churn drivers.")

        col6, col7 = st.columns(2)

        with col6:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.subheader("Customer Segmentation")
            st.caption("The pie shows how the portfolio splits across low, medium, and high churn-risk groups.")
            try:
                seg_probs = model.predict_proba(X_model)[:, 1]
                seg_labels = pd.cut(
                    seg_probs,
                    bins=[0, 0.3, 0.6, 1.0],
                    labels=["Low", "Medium", "High"],
                    include_lowest=True
                )
                seg_counts = seg_labels.value_counts().reindex(["Low", "Medium", "High"], fill_value=0)

                fig_pie = px.pie(
                    names=seg_counts.index,
                    values=seg_counts.values,
                    color_discrete_sequence=[PRIMARY, SECONDARY, ACCENT],
                    hole=0.35
                )
                fig_pie.update_traces(textinfo="percent+label")
                fig_pie.update_layout(
                    template="plotly_white",
                    margin=dict(l=10, r=10, t=30, b=10),
                    legend_title_text="Risk Segment"
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            except Exception as e:
                st.error(f"Segmentation chart error: {e}")
            st.markdown('</div>', unsafe_allow_html=True)

        with col7:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.subheader("Feature Importance")
            st.caption("The ranked bars show the main variables driving churn decisions in the selected model.")
            try:
                model_core = model.named_steps["classifier"]
                preprocess = model.named_steps["preprocessor"]

                feature_names = get_feature_names_from_preprocessor(preprocess)
                importances = getattr(model_core, "feature_importances_", None)

                if importances is not None:
                    feat_df = pd.DataFrame({
                        "Feature": feature_names[:len(importances)],
                        "Importance": importances
                    }).sort_values(by="Importance", ascending=False)

                    top_n = feat_df.head(10).copy()
                    palette = sns.color_palette([PRIMARY, SECONDARY, ACCENT], n_colors=max(len(top_n), 3))

                    fig3, ax3 = plt.subplots(figsize=(7, 5))
                    sns.barplot(
                        data=top_n,
                        x="Importance",
                        y="Feature",
                        ax=ax3,
                        palette=palette[:len(top_n)]
                    )
                    ax3.set_title("Top Features Influencing Churn")
                    ax3.set_xlabel("Importance")
                    ax3.set_ylabel("")
                    st.pyplot(fig3)
                    plt.close(fig3)
                else:
                    st.info("Feature importance is not available for this model.")
            except Exception as e:
                st.error(f"Feature importance error: {e}")
            st.markdown('</div>', unsafe_allow_html=True)


    # ================= PERFORMANCE TAB =================
    with tabs[2]:
        tab_caption("A compact performance view that presents model scores clearly and consistently for decision review.")

        col_left, col_right = st.columns(2)

        # ===== LEFT: METRICS + TABLE =====
        with col_left:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.subheader("Model Performance Metrics")
            st.caption("Higher values indicate better performance. Scores are rounded for readability.")
            try:
                y_pred = model.predict(X_model)
                y_prob = model.predict_proba(X_model)[:, 1]

                metrics = {
                    "Accuracy": accuracy_score(y_model, y_pred),
                    "Precision": precision_score(y_model, y_pred, zero_division=0),
                    "Recall": recall_score(y_model, y_pred, zero_division=0),
                    "F1 Score": f1_score(y_model, y_pred, zero_division=0),
                    "ROC AUC": roc_auc_score(y_model, y_prob),
                }

                metric_cols = st.columns(5)
                for col, (name, value) in zip(metric_cols, metrics.items()):
                    with col:
                        st.metric(name, metric_value_fmt(name, value))

                metrics_df = pd.DataFrame({
                    "Metric": list(metrics.keys()),
                    "Score": [round(v, 3) for v in metrics.values()]
                })

                st.dataframe(metrics_df, use_container_width=True, hide_index=True)

            except Exception as e:
                st.error(f"Performance table error: {e}")
            st.markdown('</div>', unsafe_allow_html=True)

    # ===== RIGHT: GRAPH =====
    with col_right:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Performance Overview")
        st.caption("The horizontal bar chart provides a quick visual comparison of the same scores.")
        try:
            fig4, ax4 = plt.subplots(figsize=(8, 5))
            sns.barplot(
                data=metrics_df,
                x="Score",
                y="Metric",
                ax=ax4,
                palette=[PRIMARY, SECONDARY, ACCENT, PRIMARY, SECONDARY]
            )
            ax4.set_xlim(0, 1)
            ax4.set_title("Model Performance Overview")
            ax4.set_xlabel("Score")
            ax4.set_ylabel("")
            st.pyplot(fig4)
            plt.close(fig4)
        except Exception as e:
            st.error(f"Performance overview error: {e}")
        st.markdown('</div>', unsafe_allow_html=True)

    # ================= EXPLAINABILITY TAB =================
    with tabs[3]:
        tab_caption("A local explanation of the selected customer, plus a geography-level risk view for broader context.")

        col10, col11 = st.columns(2)

        with col10:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.subheader("SHAP Explainability")
            st.caption("Positive bars increase churn risk; negative bars reduce it for the selected customer.")
            try:
                fig5 = local_shap_barplot(model, input_df)
                st.pyplot(fig5)
                plt.close(fig5)
            except Exception as e:
                st.error(f"SHAP error: {e}")
            st.markdown('</div>', unsafe_allow_html=True)

        with col11:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.subheader("Geography-wise Bubble Scatter")
            st.caption("Bubble size reflects churn probability, while color separates the geography groups.")
            try:
                geo_df = prepare_model_frame(df)
                geo_X = align_to_model_columns(geo_df.drop("Exited", axis=1), MODEL_INPUT_COLS)
                geo_df["Churn Probability"] = model.predict_proba(geo_X)[:, 1]
                geo_df["Bubble Size"] = np.clip(geo_df["Churn Probability"] * 60, 8, 45)

                fig_geo = px.scatter(
                    geo_df,
                    x="Age",
                    y="Balance",
                    color="Geography",
                    size="Bubble Size",
                    hover_data=["CreditScore", "NumOfProducts", "Churn Probability"],
                    opacity=0.72,
                    title="Geography-wise Churn Risk",
                    color_discrete_sequence=PLOTLY_SEQ
                )
                fig_geo.update_layout(
                    template="plotly_white",
                    title_font=dict(size=18, color=PRIMARY),
                    legend_title_text="Geography",
                    paper_bgcolor="white",
                    plot_bgcolor="white",
                    margin=dict(l=10, r=10, t=50, b=10),
                )
                st.plotly_chart(fig_geo, use_container_width=True)
            except Exception as e:
                st.error(f"Geography plot error: {e}")
            st.markdown('</div>', unsafe_allow_html=True)

    # ================= PDP TAB =================
    with tabs[4]:
        tab_caption("Partial dependence charts show how core variables move churn risk across the portfolio. All panels use the same size for easy comparison.")

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Partial Dependence Analysis")
        st.caption("The first row covers customer and product factors, while the second row shows engagement and credit effects.")
        try:
            pdp_features = ["Age", "Balance", "NumOfProducts", "IsActiveMember", "CreditScore"]
            fig6, axes = plt.subplots(2, 3, figsize=(18, 10))
            axes_flat = axes.flatten()

            pdp_colors = [PRIMARY, SECONDARY, ACCENT, SECONDARY, PRIMARY]

            for idx, (feature, color) in enumerate(zip(pdp_features, pdp_colors)):
                ax = axes_flat[idx]
                grid, preds = local_pdp_curve(model, input_raw_df, feature, X_model)
                ax.plot(grid, preds, linewidth=2.5, color=color)
                ax.fill_between(grid, preds, alpha=0.12, color=color)
                ax.axvline(float(input_raw_df.iloc[0][feature]), linestyle="--", linewidth=1.5, color=ACCENT)
                ax.scatter([float(input_raw_df.iloc[0][feature])], [prob], s=40, color=PRIMARY)
                ax.set_title(feature, fontweight="bold")
                ax.set_xlabel(feature)
                ax.set_ylabel("Churn Probability")
                ax.grid(alpha=0.2)

            axes_flat[-1].axis("off")
            plt.tight_layout()
            st.pyplot(fig6)
            plt.close(fig6)
        except Exception as e:
            st.error(f"PDP error: {e}")
        st.markdown('</div>', unsafe_allow_html=True)

    # ================= ACTIONS TAB =================
    with tabs[5]:
        tab_caption("Turn the prediction into action with a two-option simulator and a targeted retention plan.")

        col12, col13 = st.columns(2)

        with col12:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.subheader("What-if Simulator")
            st.caption("Only the three most influential levers are exposed here to keep the scenario analysis focused.")
            try:
                scenario_type = st.selectbox(
                    "Scenario Type",
                    ["Baseline", "Explain Everything"],
                    key=f"{model_name}_scenario_type"
                )

                sim_credit_score = st.slider(
                    "Credit Score (Scenario)",
                    300, 900,
                    int(input_raw_df.iloc[0]["CreditScore"]),
                    key=f"{model_name}_sim_credit"
                )
                sim_balance = st.number_input(
                    "Balance (Scenario)",
                    value=float(input_raw_df.iloc[0]["Balance"]),
                    key=f"{model_name}_sim_balance"
                )
                sim_products = st.slider(
                    "Number of Products (Scenario)",
                    1, 4,
                    int(input_raw_df.iloc[0]["NumOfProducts"]),
                    key=f"{model_name}_sim_products"
                )

                sim_raw_df = input_raw_df.copy()
                sim_raw_df["CreditScore"] = sim_credit_score
                sim_raw_df["Balance"] = sim_balance
                sim_raw_df["NumOfProducts"] = sim_products

                sim_raw_df = create_features(sim_raw_df)
                sim_df = align_to_model_columns(sim_raw_df, MODEL_INPUT_COLS)
                new_prob = model.predict_proba(sim_df)[0][1]
                delta = new_prob - prob

                st.metric("Scenario Churn Probability", f"{new_prob:.2f}", delta=f"{delta:+.2f}")

                if scenario_type == "Baseline":
                    st.info("Baseline keeps the current customer profile unchanged.")
                else:
                    st.markdown(
                        f"""
                        **Scenario explanation**
                        - Credit Score: `{sim_credit_score}`
                        - Balance: `{sim_balance:,.0f}`
                        - Number of Products: `{sim_products}`

                        **What this means**
                        - Higher credit quality generally supports lower churn risk.
                        - Balance and product depth affect relationship strength.
                        - More products usually improves stickiness through deeper engagement.
                        """
                    )
            except Exception as e:
                st.error(f"Simulator error: {e}")
            st.markdown('</div>', unsafe_allow_html=True)

        with col13:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.subheader("Retention Process")
            st.caption("Choose a response, then see the best segment where that action should be applied.")
            try:
                retention_action = st.selectbox(
                    "Choose a retention action",
                    [
                        "Call RM",
                        "Send Personalized Email",
                        "Offer Fee Waiver",
                        "Offer Loyalty Bonus"
                    ],
                    key=f"{model_name}_retention_choice"
                )

                retention_timing = st.radio(
                    "Follow-up Timing",
                    ["Immediate", "Within 7 days", "Within 30 days"],
                    horizontal=True,
                    key=f"{model_name}_retention_timing"
                )

                retention_channel = st.multiselect(
                    "Delivery Channel",
                    ["Email", "SMS", "Phone"],
                    default=preferred_channels if preferred_channels else ["Email"],
                    key=f"{model_name}_retention_channel"
                )

                segment_map = {
                    "Call RM": ("High Risk", "Best for customers needing human follow-up and relationship repair."),
                    "Send Personalized Email": ("Medium Risk", "Best for customers who still respond well to targeted communication."),
                    "Offer Fee Waiver": ("High Risk", "Best for customers showing friction or price sensitivity."),
                    "Offer Loyalty Bonus": ("Medium Risk", "Best for customers with room for reinforcement and upsell."),
                }

                target_segment, segment_note = segment_map[retention_action]

                st.info(f"Best applied for: **{target_segment}** segment. {segment_note}")

                if st.button("Apply Retention Plan", key=f"{model_name}_retention_btn"):
                    channel_text = ", ".join(retention_channel) if retention_channel else "Email"
                    if retention_action == "Call RM":
                        st.success(f"Retention plan applied: assign Relationship Manager follow-up ({retention_timing.lower()}).")
                    elif retention_action == "Send Personalized Email":
                        st.success(f"Retention plan applied: send a personalized email via {channel_text} ({retention_timing.lower()}).")
                    elif retention_action == "Offer Fee Waiver":
                        st.success(f"Retention plan applied: offer a fee waiver to reduce friction ({retention_timing.lower()}).")
                    else:
                        st.success(f"Retention plan applied: present a loyalty bonus through {channel_text} ({retention_timing.lower()}).")
            except Exception as e:
                st.error(f"Retention action error: {e}")
            st.markdown('</div>', unsafe_allow_html=True)


render_dashboard(selected_model, selected_model_name, prob, risk_score, risk, retention_rate)

# ================= FOOTER =================
st.markdown("---")
st.markdown(
    """
    <div style="text-align:center; color:#6B7280; font-size:0.95rem; padding-bottom:10px;">
        Submitted By: <b>Ambika Sharnarthi</b> &nbsp;&nbsp;|&nbsp;&nbsp; Guided By: <b>Sai Kagne</b>
    </div>
    """,
    unsafe_allow_html=True
)