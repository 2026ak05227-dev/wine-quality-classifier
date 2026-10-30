# Wine Quality Classification — ML Assignment 2

**Name:** Sabari Nathan M &nbsp;|&nbsp; **ID:** 2026AK05227 &nbsp;|&nbsp; **Course:** Machine Learning (M.Tech AIML)

An end-to-end classification project: five ML models trained on the UCI Wine
Quality dataset, served through an interactive Streamlit app that scores
user-uploaded **test** data (no retraining in the app).

---

## a. Problem statement

Predict whether a wine is of **good quality** from its physicochemical
properties. The original UCI target is an integer quality score (0–10); we
frame it as a **binary classification** problem:

> `good_quality = 1` if `quality >= 7`, else `0`.

This is a practically useful framing for a producer who wants an automated
"is this batch premium-grade?" flag from lab measurements alone, without a
human taster. The positive class is the minority (~20% of wines), so the task
rewards models that balance precision and recall rather than just chasing
accuracy — which is why **AUC** and **MCC** are reported alongside accuracy.

## b. Dataset description

- **Source:** UCI Machine Learning Repository — [Wine Quality Data Set](https://archive.ics.uci.edu/dataset/186/wine+quality)
  (Cortez et al., 2009). Red and white *vinho verde* wines.
- **Instances:** 6,497 (1,599 red + 4,898 white) — well above the 500 minimum.
- **Features:** **12** (above the 12 minimum) — 11 physicochemical measurements
  plus an engineered `wine_type` flag:

  | # | Feature | Description |
  |---|---------|-------------|
  | 1 | fixed acidity | tartaric acid (g/dm³) |
  | 2 | volatile acidity | acetic acid (g/dm³) |
  | 3 | citric acid | g/dm³ |
  | 4 | residual sugar | g/dm³ |
  | 5 | chlorides | sodium chloride (g/dm³) |
  | 6 | free sulfur dioxide | mg/dm³ |
  | 7 | total sulfur dioxide | mg/dm³ |
  | 8 | density | g/cm³ |
  | 9 | pH | acidity scale |
  | 10 | sulphates | potassium sulphate (g/dm³) |
  | 11 | alcohol | % by volume |
  | 12 | **wine_type** | 0 = red, 1 = white (engineered) |

- **Target:** `good_quality` (binary). Class balance: **19.7% positive**.
- **Split:** stratified 80/20 train/test (`random_state=42`). The 1,300-row test
  split is saved as [`test_data.csv`](test_data.csv) and is what the Streamlit
  app consumes.
- **Preprocessing:** every model is a scikit-learn `Pipeline` that applies
  `StandardScaler` before the classifier, so scaling is fitted on train only and
  travels inside each saved `.joblib` — no leakage, nothing to re-fit at serve time.

## c. GitHub Repository Link

> **https://github.com/2026ak05227-dev/wine-quality-classifier.git**  

**Live Streamlit app:** https://wine-quality-classifier-7rhjeufnghy89v988w2kse.streamlit.app/

Repository contents:

```
project-folder/
├── app.py                 # Streamlit app (loads models, scores uploaded test CSV)
├── train_models.py        # Source code: trains + evaluates + saves all 5 models
├── requirements.txt       # Pinned dependencies
├── README.md              # This file
├── test_data.csv          # Held-out test split (features + target)
├── metrics.csv            # Comparison table (generated)
├── data/                  # Raw UCI red + white CSVs
└── model/                 # Saved artifacts
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    ├── svm.joblib
    └── metadata.joblib     # feature names, target, class labels
```

## d. Models used

All five models share the **same** dataset, split, and `StandardScaler`
preprocessing. Metrics below are computed on the held-out **test** set
(1,300 rows). The positive class is `good_quality = 1`.

### Comparison table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.7123 | 0.8019 | 0.3825 | 0.7500 | 0.5066 | 0.3701 |
| Decision Tree | 0.7400 | 0.8212 | 0.4170 | 0.8047 | 0.5493 | 0.4333 |
| KNN | **0.8792** | **0.9166** | **0.7538** | 0.5742 | **0.6519** | **0.5883** |
| Naive Bayes | 0.7346 | 0.7486 | 0.3901 | 0.6172 | 0.4781 | 0.3268 |
| Support Vector Machine | 0.7431 | 0.8551 | 0.4220 | **0.8242** | 0.5582 | 0.4474 |

*(Bold = best model for that metric.)*

### Observations about model performance

| ML Model Name | Observation about model performance |
|---|---|
| **Logistic Regression** | A linear decision boundary is too simple for this data. With `class_weight='balanced'` it recovers 75% of good wines (recall 0.75) but at poor precision (0.38) — many false alarms. A respectable AUC (0.80) shows it still ranks wines sensibly; it just can't place a sharp threshold. |
| **Decision Tree** | Depth-limited (max_depth=8, min_samples_leaf=20) to curb overfitting. Captures non-linear splits in alcohol/density/sulphates, lifting recall to 0.80 and MCC to 0.43, but precision stays low (0.42). A single tree is higher-variance than the neighbour and kernel methods. |
| **KNN** | **Best overall.** k=25 distance-weighted neighbours on standardized features win 5 of 6 metrics — accuracy 0.88, AUC 0.92, precision 0.75, F1 and MCC. Good wines cluster tightly in the physicochemical space, so local similarity is a very strong signal. It leans toward precision (recall 0.57) because it is not class-balanced. |
| **Naive Bayes** | **Weakest.** Its feature-independence assumption is badly violated here — alcohol, density and residual sugar are strongly correlated — giving the lowest AUC (0.75) and MCC (0.33). Fast and simple, but the wrong model shape for correlated lab measurements. |
| **Support Vector Machine** | RBF kernel with balanced weights gives the **highest recall (0.82)** and the second-best AUC (0.86) — a strong non-linear learner. Like LR and the tree, class balancing trades precision (0.42) for recall, so its F1/MCC trail KNN's. |
| **Overall, Winner for your dataset?** | **KNN.** It dominates accuracy, AUC, precision, F1 and MCC, and its distance-weighted local view matches the way good wines cluster in feature space. If the business goal were *catching every good wine* (maximising recall), SVM would be the pick; for balanced overall quality, KNN is the clear winner. |

---

## Running locally

```bash
pip install -r requirements.txt
python train_models.py      # regenerates model/ , test_data.csv , metrics.csv
streamlit run app.py
```

## Streamlit app features

- **CSV upload** — upload a test CSV (must contain the 12 features + `good_quality`). Please upload test_data.csv from our repo.
- **Model selection dropdown** — pick any one model, or **"Compare all"** for a side-by-side metrics table.
- **Evaluation metrics** — Accuracy, AUC, Precision, Recall, F1, MCC shown per model.
- **Confusion matrix + classification report** for the selected model.
- Models are **loaded from disk and never retrained** on the uploaded data.

