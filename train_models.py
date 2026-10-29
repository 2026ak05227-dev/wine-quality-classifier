"""
Wine Quality Classification - trains 5 classifiers and saves artifacts.

Dataset: UCI Wine Quality (red + white combined).
  11 physicochemical features + 1 engineered `wine_type` feature = 12 features.
Target: binary `good_quality` = 1 if quality >= 7 else 0.

Run:  python3 train_models.py
Produces:
  model/*.joblib           one Pipeline(StandardScaler -> classifier) per model
  model/metadata.joblib    feature names, target name, class labels
  test_data.csv            held-out test split (features + target) for the Streamlit app
  metrics.csv              comparison table of all 6 metrics for all 5 models
"""
import os
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef)

RANDOM_STATE = 42
MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)


def load_data():
    red = pd.read_csv("data/winequality-red.csv", sep=";")
    white = pd.read_csv("data/winequality-white.csv", sep=";")
    red["wine_type"] = 0          # 0 = red
    white["wine_type"] = 1        # 1 = white
    df = pd.concat([red, white], ignore_index=True)
    df["good_quality"] = (df["quality"] >= 7).astype(int)
    df = df.drop(columns=["quality"])   # drop raw quality to avoid target leakage
    return df


def build_models():
    """Each model is a Pipeline so the saved artifact is self-contained
    (scaler travels with the classifier - the app never re-fits anything)."""
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced",
                                       random_state=RANDOM_STATE)),
        ]),
        "Decision Tree": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", DecisionTreeClassifier(max_depth=8, min_samples_leaf=20,
                                           class_weight="balanced",
                                           random_state=RANDOM_STATE)),
        ]),
        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=25, weights="distance")),
        ]),
        "Naive Bayes": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GaussianNB()),
        ]),
        "Support Vector Machine": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", C=2.0, gamma="scale", probability=True,
                        class_weight="balanced", random_state=RANDOM_STATE)),
        ]),
    }


def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_proba),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_test, y_pred),
    }


FILE_NAMES = {
    "Logistic Regression": "logistic_regression",
    "Decision Tree": "decision_tree",
    "KNN": "knn",
    "Naive Bayes": "naive_bayes",
    "Support Vector Machine": "svm",
}


def main():
    df = load_data()
    target = "good_quality"
    feature_cols = [c for c in df.columns if c != target]
    print(f"Dataset: {df.shape[0]} rows, {len(feature_cols)} features")
    print(f"Features: {feature_cols}")
    print(f"Class balance (good_quality=1): {df[target].mean():.3f}\n")

    X = df[feature_cols]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    # Save the held-out test split for the Streamlit app (features + target).
    test_df = X_test.copy()
    test_df[target] = y_test.values
    test_df.to_csv("test_data.csv", index=False)
    print(f"Saved test_data.csv ({test_df.shape[0]} rows)\n")

    models = build_models()
    rows = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        joblib.dump(model, os.path.join(MODEL_DIR, f"{FILE_NAMES[name]}.joblib"))
        m = evaluate(model, X_test, y_test)
        m["Model"] = name
        rows.append(m)
        print(f"{name:24s}  acc={m['Accuracy']:.3f}  auc={m['AUC']:.3f}  "
              f"prec={m['Precision']:.3f}  rec={m['Recall']:.3f}  "
              f"f1={m['F1']:.3f}  mcc={m['MCC']:.3f}")

    joblib.dump({
        "feature_cols": feature_cols,
        "target": target,
        "file_names": FILE_NAMES,
        "class_labels": {0: "not good (<7)", 1: "good (>=7)"},
    }, os.path.join(MODEL_DIR, "metadata.joblib"))

    metrics_df = pd.DataFrame(rows)[
        ["Model", "Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]]
    metrics_df.to_csv("metrics.csv", index=False)
    print("\nSaved model/*.joblib, model/metadata.joblib, metrics.csv")
    print("\n" + metrics_df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
