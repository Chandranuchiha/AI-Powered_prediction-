import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.inspection import permutation_importance

# ------------------------------------------------------------
# AI-Powered Student Performance Prediction
# Synthetic demo data is used only until the user uploads a CSV.
# Expected target: a numeric academic outcome such as final_score.
# ------------------------------------------------------------

st.set_page_config(
    page_title="Student Performance | Insight Dashboard",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Premium marketplace-inspired visual language: clean white canvas,
# deep navy typography, restrained blue accents, and compact metric cards.
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .stApp { background: #f5f7fb; color: #172033; }
    header[data-testid="stHeader"] { background: rgba(245,247,251,.9); }
    [data-testid="stSidebar"] { background: #101d35; }
    [data-testid="stSidebar"] * { color: #eef3ff !important; }
    .hero {
        background: linear-gradient(115deg,#101d35 0%,#173d70 65%,#2468b5 100%);
        padding: 28px 32px; border-radius: 18px; color: white;
        margin: 4px 0 22px 0; box-shadow: 0 12px 30px rgba(20,45,85,.12);
    }
    .hero h1 { font-family: Manrope,sans-serif; font-size: 29px; margin: 0 0 6px 0; color: white; }
    .hero p { margin: 0; color: #d9e8ff; font-size: 14px; }
    .eyebrow { text-transform: uppercase; letter-spacing: 1.5px; font-size: 10px; font-weight: 700; color: #a9cfff; margin-bottom: 8px; }
    .metric-card {
        background: white; border: 1px solid #e8edf5; border-radius: 14px;
        padding: 17px 18px; min-height: 105px; box-shadow: 0 3px 12px rgba(25,45,80,.035);
    }
    .metric-label { color: #748096; font-size: 12px; font-weight: 600; margin-bottom: 9px; }
    .metric-value { color: #14213a; font-family: Manrope,sans-serif; font-size: 25px; font-weight: 800; }
    .metric-note { color: #8a95a8; font-size: 11px; margin-top: 4px; }
    .section-title { font-family: Manrope,sans-serif; font-weight: 800; font-size: 18px; color: #17243d; margin: 8px 0 3px 0; }
    .section-sub { color: #7a869b; font-size: 12px; margin-bottom: 12px; }
    .panel {
        background: white; border: 1px solid #e8edf5; border-radius: 15px;
        padding: 18px 20px; box-shadow: 0 3px 12px rgba(25,45,80,.03);
    }
    div[data-testid="stMetric"] { background: white; border: 1px solid #e8edf5; padding: 15px; border-radius: 13px; }
    div[data-testid="stMetricLabel"] { color: #758198; }
    .stButton>button, .stDownloadButton>button {
        border-radius: 9px; border: 0; background: #1769c2; color: white;
        font-weight: 700; padding: 8px 16px;
    }
    .stButton>button:hover, .stDownloadButton>button:hover { background: #10559f; color: white; }
    </style>
    """,
    unsafe_allow_html=True,
)

REQUIRED_HINTS = ["study_hours", "attendance", "previous_score", "final_score"]


@st.cache_data
def make_demo_data(n=520, seed=42):
    """Create a clearly labelled, reproducible synthetic teaching dataset."""
    rng = np.random.default_rng(seed)
    study = np.clip(rng.normal(3.4, 1.35, n), 0.2, 8.0)
    attendance = np.clip(rng.normal(82, 11, n), 40, 100)
    previous = np.clip(rng.normal(67, 14, n), 20, 100)
    assignments = np.clip(rng.normal(76, 15, n), 15, 100)
    sleep = np.clip(rng.normal(7.0, 1.0, n), 4, 10)
    internet = rng.choice(["Good", "Average", "Limited"], n, p=[.62, .29, .09])
    support = rng.choice(["Yes", "No"], n, p=[.58, .42])
    internet_effect = pd.Series(internet).map({"Good": 2.0, "Average": 0.0, "Limited": -3.0}).to_numpy()
    support_effect = pd.Series(support).map({"Yes": 2.0, "No": -1.0}).to_numpy()
    score = (19 + study * 3.8 + attendance * .20 + previous * .30
             + assignments * .10 + sleep * .7 + internet_effect
             + support_effect + rng.normal(0, 5.5, n))
    score = np.clip(score, 0, 100)
    return pd.DataFrame({
        "study_hours": study.round(1),
        "attendance": attendance.round(1),
        "previous_score": previous.round(1),
        "assignment_average": assignments.round(1),
        "sleep_hours": sleep.round(1),
        "internet_access": internet,
        "academic_support": support,
        "final_score": score.round(1),
    })


def clean_data(frame):
    """Apply repeatable, conservative preparation and return audit notes."""
    data = frame.copy()
    data.columns = [str(c).strip().lower().replace(" ", "_") for c in data.columns]
    notes = []
    before = len(data)
    data = data.drop_duplicates()
    if len(data) != before:
        notes.append(f"Removed {before - len(data)} duplicate row(s).")
    data = data.replace([np.inf, -np.inf], np.nan)
    for col in data.select_dtypes(include="object").columns:
        data[col] = data[col].astype("string").str.strip()
        data[col] = data[col].replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
    return data, notes


def build_model(data, target):
    X = data.drop(columns=[target])
    y = pd.to_numeric(data[target], errors="coerce")
    valid = y.notna()
    X, y = X.loc[valid], y.loc[valid]
    if len(y) < 12:
        raise ValueError("At least 12 rows with a valid numeric target are required.")
    if y.nunique() < 2:
        raise ValueError("The target column must contain at least two different numeric values.")
    numeric = X.select_dtypes(include=np.number).columns.tolist()
    categorical = X.select_dtypes(exclude=np.number).columns.tolist()
    transformers = []
    if numeric:
        transformers.append(("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median"))
        ]), numeric))
    if categorical:
        transformers.append(("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical))
    if not transformers:
        raise ValueError("At least one usable feature column is required.")
    prep = ColumnTransformer(transformers)
    model = Pipeline([
        ("preprocessor", prep),
        ("regressor", RandomForestRegressor(
            n_estimators=220, min_samples_leaf=2, random_state=42, n_jobs=-1
        ))
    ])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    metrics = {
        "mae": mean_absolute_error(y_test, pred),
        "rmse": np.sqrt(mean_squared_error(y_test, pred)),
        "r2": r2_score(y_test, pred),
    }
    return model, X, y, X_test, y_test, pred, metrics


def metric_card(label, value, note):
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div><div class="metric-note">{note}</div></div>',
        unsafe_allow_html=True,
    )


# Sidebar: data source and controls
st.sidebar.markdown("## 🎓 Student Insights")
st.sidebar.caption("AI-powered academic analytics")
st.sidebar.markdown("---")
uploaded = st.sidebar.file_uploader("Upload academic data (CSV)", type=["csv"])
st.sidebar.caption("Use anonymised, synthetic, or explicitly authorised records.")
st.sidebar.markdown("---")
st.sidebar.markdown("**Dashboard sections**")
st.sidebar.markdown("01  Overview & trends\n\n02  Model evaluation\n\n03  Student prediction\n\n04  Data quality")

if uploaded is not None:
    try:
        raw = pd.read_csv(uploaded)
        source_label = f"Uploaded file · {uploaded.name}"
    except Exception as exc:
        st.error(f"Could not read this CSV file. Please check its format. Details: {exc}")
        st.stop()
else:
    raw = make_demo_data()
    source_label = "Synthetic demo dataset · generated locally"

data, prep_notes = clean_data(raw)
numeric_targets = [
    c for c in data.columns
    if pd.api.types.is_numeric_dtype(data[c]) and data[c].notna().sum() >= 12
]
if not numeric_targets:
    st.error("No numeric target column was found. Add a numeric outcome column such as final_score.")
    st.stop()

st.markdown(
    f"""
    <div class="hero">
      <div class="eyebrow">Academic intelligence platform</div>
      <h1>Student Performance Dashboard</h1>
      <p>Turn academic records into clear signals for timely, evidence-led student support.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
if uploaded is None:
    st.info("Demo mode: these are synthetic records for demonstrating the workflow, not real student results.")
else:
    st.success(f"Data source: {source_label}")

with st.sidebar:
    target = st.selectbox("Prediction target", numeric_targets,
                          index=numeric_targets.index("final_score") if "final_score" in numeric_targets else 0)
    st.caption(f"Selected outcome: **{target}**")

try:
    model, X, y, X_test, y_test, predictions, metrics = build_model(data, target)
except Exception as exc:
    st.error(str(exc))
    st.stop()

# Overview
st.markdown('<div class="section-title">Performance at a glance</div>', unsafe_allow_html=True)
st.markdown('<div class="section-sub">A high-level snapshot of the selected dataset and baseline model.</div>', unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4)
with c1: metric_card("STUDENT RECORDS", f"{len(data):,}", "Rows after duplicate removal")
with c2: metric_card("AVERAGE OUTCOME", f"{y.mean():.1f}", f"Mean {target}")
with c3: metric_card("MODEL MAE", f"{metrics['mae']:.2f}", "Average absolute error · holdout set")
with c4: metric_card("MODEL R²", f"{metrics['r2']:.3f}", "Holdout explanatory score")

st.markdown("")
left, right = st.columns([1.2, 1])
with left:
    st.markdown('<div class="section-title">Outcome distribution</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">How the selected academic outcome is distributed.</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.hist(y, bins=18, edgecolor="white", linewidth=1.2)
    ax.set_xlabel(target.replace("_", " ").title())
    ax.set_ylabel("Number of records")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
with right:
    st.markdown('<div class="section-title">Feature relationships</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Numeric inputs compared with the selected outcome.</div>', unsafe_allow_html=True)
    corr = data.select_dtypes(include=np.number).corr(numeric_only=True)
    if target in corr.columns and len(corr.columns) > 1:
        rel = corr[target].drop(target).sort_values()
        fig, ax = plt.subplots(figsize=(6, 3.2))
        rel.plot(kind="barh", ax=ax)
        ax.set_xlabel("Correlation")
        ax.set_ylabel("")
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    else:
        st.info("Add numeric feature columns to view feature relationships.")

st.markdown("---")
# Model evaluation
st.markdown('<div class="section-title">Model performance & evaluation</div>', unsafe_allow_html=True)
st.markdown('<div class="section-sub">Baseline Random Forest regression · fixed 80/20 random holdout split.</div>', unsafe_allow_html=True)
m1, m2, m3 = st.columns(3)
m1.metric("Mean absolute error (MAE)", f"{metrics['mae']:.2f}")
m2.metric("Root mean squared error (RMSE)", f"{metrics['rmse']:.2f}")
m3.metric("R² score", f"{metrics['r2']:.3f}")
ev1, ev2 = st.columns([1, 1])
with ev1:
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.scatter(y_test, predictions, alpha=.65)
    low = min(float(y_test.min()), float(np.min(predictions)))
    high = max(float(y_test.max()), float(np.max(predictions)))
    ax.plot([low, high], [low, high], linestyle="--")
    ax.set_xlabel("Actual outcome")
    ax.set_ylabel("Predicted outcome")
    ax.set_title("Actual vs predicted")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
with ev2:
    result_view = pd.DataFrame({
        "Actual": y_test.to_numpy(),
        "Predicted": predictions.round(2),
        "Absolute error": np.abs(y_test.to_numpy() - predictions).round(2),
    })
    st.dataframe(result_view.head(12), use_container_width=True, hide_index=True)
    st.caption("Preview of holdout predictions; rows are not student identifiers.")

st.markdown("---")
# Individual prediction
st.markdown('<div class="section-title">Explore an individual prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="section-sub">Enter feature values to estimate the selected academic outcome. This is a decision-support estimate, not a definitive judgement.</div>', unsafe_allow_html=True)
feature_cols = X.columns.tolist()
with st.container(border=True):
    input_values = {}
    cols = st.columns(3)
    for i, col in enumerate(feature_cols):
        series = X[col]
        with cols[i % 3]:
            if pd.api.types.is_numeric_dtype(series):
                default = float(series.median()) if series.notna().any() else 0.0
                input_values[col] = st.number_input(
                    col.replace("_", " ").title(), value=default, key=f"pred_{col}"
                )
            else:
                choices = sorted(series.dropna().astype(str).unique().tolist())
                if not choices:
                    choices = ["Unknown"]
                input_values[col] = st.selectbox(
                    col.replace("_", " ").title(), choices, key=f"pred_{col}"
                )
    if st.button("Generate performance estimate", type="primary"):
        sample = pd.DataFrame([input_values], columns=feature_cols)
        estimate = float(model.predict(sample)[0])
        st.success(f"Estimated {target.replace('_', ' ').title()}: **{estimate:.1f}**")
        st.caption("The estimate is based on patterns in the selected dataset and should be reviewed alongside educator judgement.")

st.markdown("---")
# Data quality and export
st.markdown('<div class="section-title">Data quality & records</div>', unsafe_allow_html=True)
st.markdown('<div class="section-sub">Review input completeness and inspect the prepared data before relying on results.</div>', unsafe_allow_html=True)
q1, q2, q3 = st.columns(3)
q1.metric("Missing cells", f"{int(data.isna().sum().sum()):,}")
q2.metric("Duplicate rows removed", f"{len(raw) - len(data):,}")
q3.metric("Feature columns", f"{len(feature_cols):,}")
if prep_notes:
    for note in prep_notes:
        st.write(f"• {note}")
else:
    st.caption("No duplicate rows were removed.")
missing = data.isna().sum().rename("Missing values").to_frame()
st.dataframe(missing, use_container_width=True)
with st.expander("Preview prepared dataset"):
    st.dataframe(data.head(30), use_container_width=True)
st.download_button(
    "Download prepared data (CSV)",
    data=data.to_csv(index=False).encode("utf-8"),
    file_name="prepared_student_performance.csv",
    mime="text/csv",
)

st.markdown("---")
st.caption(
    "Method note: A baseline Random Forest regressor is trained on the selected numeric outcome. "
    "The random holdout metrics describe this dataset only; they do not establish real-world accuracy, "
    "causality, or suitability for high-stakes student decisions. Use anonymised or authorised data."
)
