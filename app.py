"""
Wine Quality Classification - Streamlit demo app.

Loads 5 pre-trained classifiers and evaluates them on a user-uploaded test CSV.
Models are NEVER retrained here - the app only loads saved pipelines, predicts,
and reports metrics (as required by the assignment).
"""
import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef,
                             confusion_matrix, classification_report)

MODEL_DIR = "model"

st.set_page_config(page_title="Wine Quality Classifier", page_icon="🍷",
                   layout="wide")


@st.cache_resource
def load_artifacts():
    meta = joblib.load(os.path.join(MODEL_DIR, "metadata.joblib"))
    models = {name: joblib.load(os.path.join(MODEL_DIR, f"{fn}.joblib"))
              for name, fn in meta["file_names"].items()}
    return meta, models


def compute_metrics(y_true, y_pred, y_proba):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_proba),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


meta, models = load_artifacts()
feature_cols = meta["feature_cols"]
target = meta["target"]

st.title("🍷 Wine Quality Classifier")
st.markdown(
    "Binary classification of wine as **good (quality ≥ 7)** vs **not good**, "
    "using 11 physicochemical measurements plus wine type (red/white). "
    "Five models were trained offline; this app loads them and scores your "
    "uploaded **test** data — no retraining."
)

with st.sidebar:
    st.header("1 · Upload test data")
    st.caption(
        "CSV must contain all 12 feature columns **and** the "
        f"`{target}` target column. Download `test_data.csv` from the repo "
        "to try it out."
    )
    uploaded = st.file_uploader("Test CSV", type=["csv"])

    st.header("2 · Select model")
    model_choice = st.selectbox(
        "Model", list(models.keys()) + ["Compare all"], index=len(models))

if uploaded is None:
    st.info("⬅️ Upload a test CSV from the sidebar to see results.")
    st.subheader("Expected columns")
    st.code(", ".join(feature_cols + [target]))
    st.stop()

# --- Load + validate the uploaded CSV ---------------------------------------
df = pd.read_csv(uploaded)
missing = [c for c in feature_cols + [target] if c not in df.columns]
if missing:
    st.error(f"Uploaded CSV is missing required columns: {missing}")
    st.stop()

X = df[feature_cols]
y = df[target].astype(int)
st.success(f"Loaded {len(df)} rows. Class balance "
           f"(good=1): {y.mean():.3f}")

with st.expander("Preview uploaded data"):
    st.dataframe(df.head(20), use_container_width=True)


def score(name):
    model = models[name]
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    return y_pred, y_proba


# --- Compare-all view -------------------------------------------------------
if model_choice == "Compare all":
    st.subheader("Evaluation metrics — all models")
    rows = []
    for name in models:
        y_pred, y_proba = score(name)
        m = compute_metrics(y, y_pred, y_proba)
        m["Model"] = name
        rows.append(m)
    table = pd.DataFrame(rows)[
        ["Model", "Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]]
    best_by = {c: table.loc[table[c].idxmax(), "Model"]
               for c in ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]}
    st.dataframe(
        table.set_index("Model").style.format("{:.4f}").highlight_max(axis=0,
                                                                       color="#b7e4c7"),
        use_container_width=True)
    st.caption("Green = best model for that metric.")

    winner = table.loc[table["MCC"].idxmax(), "Model"]
    st.success(f"🏆 Best overall (by MCC): **{winner}**")

    fig, ax = plt.subplots(figsize=(9, 4))
    plot_df = table.set_index("Model")[["Accuracy", "AUC", "F1", "MCC"]]
    plot_df.plot(kind="bar", ax=ax)
    ax.set_ylabel("Score")
    ax.set_title("Model comparison")
    ax.legend(loc="lower right")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    st.pyplot(fig)
    st.stop()

# --- Single-model view ------------------------------------------------------
st.subheader(f"Results — {model_choice}")
y_pred, y_proba = score(model_choice)
m = compute_metrics(y, y_pred, y_proba)

cols = st.columns(6)
for col, (k, v) in zip(cols, m.items()):
    col.metric(k, f"{v:.4f}")

left, right = st.columns(2)

with left:
    st.markdown("**Confusion matrix**")
    cm = confusion_matrix(y, y_pred)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["not good", "good"],
                yticklabels=["not good", "good"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

with right:
    st.markdown("**Classification report**")
    report = classification_report(
        y, y_pred, target_names=["not good (<7)", "good (>=7)"],
        output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(report).T.round(3), use_container_width=True)

with st.expander("Predictions on uploaded rows"):
    out = df.copy()
    out["predicted"] = y_pred
    out["prob_good"] = y_proba.round(3)
    st.dataframe(out.head(100), use_container_width=True)
