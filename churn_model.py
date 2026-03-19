"""
Customer Churn Prediction Model
================================
XGBoost classifier on Telco Customer Churn data.
Includes: preprocessing, training, evaluation, SHAP interpretability, and saving.

Dataset: IBM Telco Customer Churn (or compatible CSV)
Usage:   python churn_model.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import shap
import warnings
import os

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score
)
from xgboost import XGBClassifier
import joblib

warnings.filterwarnings("ignore")

# ─── CONFIG ───────────────────────────────────────────────────────────────────
DATA_PATH   = "telco_churn.csv"
MODEL_PATH  = "churn_model.pkl"
PLOTS_DIR   = "plots"
RANDOM_SEED = 42
TEST_SIZE   = 0.20

os.makedirs(PLOTS_DIR, exist_ok=True)

# ─── STYLE ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0e1117",
    "axes.facecolor":   "#0e1117",
    "axes.edgecolor":   "#2a2d35",
    "axes.labelcolor":  "#c9cdd4",
    "text.color":       "#c9cdd4",
    "xtick.color":      "#6b7280",
    "ytick.color":      "#6b7280",
    "grid.color":       "#1e2029",
    "grid.linestyle":   "--",
    "font.family":      "monospace",
    "axes.spines.top":  False,
    "axes.spines.right":False,
})
TEAL   = "#00e5c0"
PURPLE = "#7c6ff7"
CORAL  = "#f472b6"
AMBER  = "#f59e0b"

# ─── 1. LOAD DATA ─────────────────────────────────────────────────────────────
print("\n[1/6] Loading data...")
df = pd.read_csv(DATA_PATH)
print(f"      Shape: {df.shape}  |  Churn rate: {df['Churn'].eq('Yes').mean()*100:.1f}%")

# ─── 2. PREPROCESSING ─────────────────────────────────────────────────────────
print("[2/6] Preprocessing...")

# Drop ID column
df.drop(columns=["customerID"], inplace=True, errors="ignore")

# TotalCharges can be blank for new customers — replace with 0
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)

# Target encoding
df["Churn"] = (df["Churn"] == "Yes").astype(int)

# Encode all object columns
cat_cols = df.select_dtypes(include="object").columns.tolist()
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col].astype(str))

# Feature / target split
X = df.drop(columns=["Churn"])
y = df["Churn"]

# Train / test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
)
print(f"      Train: {X_train.shape[0]}  |  Test: {X_test.shape[0]}")

# ─── 3. TRAIN XGBOOST ─────────────────────────────────────────────────────────
print("[3/6] Training XGBoost...")

scale_pos = (y_train == 0).sum() / (y_train == 1).sum()

model = XGBClassifier(
    n_estimators      = 400,
    max_depth         = 5,
    learning_rate     = 0.05,
    subsample         = 0.85,
    colsample_bytree  = 0.80,
    min_child_weight  = 3,
    gamma             = 0.1,
    reg_alpha         = 0.1,
    reg_lambda        = 1.0,
    scale_pos_weight  = scale_pos,
    use_label_encoder = False,
    eval_metric       = "auc",
    random_state      = RANDOM_SEED,
    n_jobs            = -1,
    verbosity         = 0,
)

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False,
)

# Cross-validation AUC
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
cv_scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
print(f"      CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ─── 4. EVALUATE ──────────────────────────────────────────────────────────────
print("[4/6] Evaluating...")

y_pred      = model.predict(X_test)
y_prob      = model.predict_proba(X_test)[:, 1]
auc_score   = roc_auc_score(y_test, y_prob)
ap_score    = average_precision_score(y_test, y_prob)

print(f"\n      ── Test Metrics ──")
print(f"      AUC-ROC  : {auc_score:.4f}")
print(f"      Avg Prec : {ap_score:.4f}")
print(f"\n{classification_report(y_test, y_pred, target_names=['No Churn','Churn'])}")

# ─── 5. PLOTS ─────────────────────────────────────────────────────────────────
print("[5/6] Generating plots...")

# ── 5a. Dashboard: ROC, PR, Confusion Matrix, Feature Importance
fig = plt.figure(figsize=(16, 10), facecolor="#0e1117")
fig.suptitle("Churn Prediction Model  ·  Evaluation Dashboard",
             fontsize=14, color="#e8eaf0", y=0.98, fontfamily="monospace")
gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.4)

# ROC Curve
ax1 = fig.add_subplot(gs[0, :2])
fpr, tpr, _ = roc_curve(y_test, y_prob)
ax1.plot(fpr, tpr, color=TEAL, lw=2, label=f"XGBoost (AUC = {auc_score:.3f})")
ax1.plot([0,1],[0,1], color="#2a2d35", lw=1, linestyle="--", label="Random")
ax1.fill_between(fpr, tpr, alpha=0.08, color=TEAL)
ax1.set_xlabel("False Positive Rate"); ax1.set_ylabel("True Positive Rate")
ax1.set_title("ROC Curve", color="#e8eaf0")
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# PR Curve
ax2 = fig.add_subplot(gs[0, 2:])
prec, rec, _ = precision_recall_curve(y_test, y_prob)
ax2.plot(rec, prec, color=PURPLE, lw=2, label=f"AP = {ap_score:.3f}")
ax2.fill_between(rec, prec, alpha=0.08, color=PURPLE)
ax2.axhline(y_test.mean(), color="#2a2d35", linestyle="--", lw=1, label="Baseline")
ax2.set_xlabel("Recall"); ax2.set_ylabel("Precision")
ax2.set_title("Precision–Recall Curve", color="#e8eaf0")
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# Confusion Matrix
ax3 = fig.add_subplot(gs[1, :2])
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", ax=ax3,
            cmap=sns.light_palette(TEAL, as_cmap=True),
            linecolor="#0e1117", linewidths=2,
            xticklabels=["No Churn","Churn"],
            yticklabels=["No Churn","Churn"])
ax3.set_title("Confusion Matrix", color="#e8eaf0")
ax3.set_xlabel("Predicted"); ax3.set_ylabel("Actual")

# Feature Importance (top 10)
ax4 = fig.add_subplot(gs[1, 2:])
feat_imp = pd.Series(model.feature_importances_, index=X.columns).nlargest(10)
bars = ax4.barh(feat_imp.index[::-1], feat_imp.values[::-1], color=AMBER, height=0.6)
ax4.set_title("Top 10 Feature Importances", color="#e8eaf0")
ax4.set_xlabel("Importance Score")
ax4.grid(True, alpha=0.3, axis="x")
for bar in bars:
    ax4.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
             f"{bar.get_width():.3f}", va="center", fontsize=8, color="#6b7280")

plt.savefig(f"{PLOTS_DIR}/evaluation_dashboard.png", dpi=150, bbox_inches="tight",
            facecolor="#0e1117")
plt.close()
print(f"      Saved: {PLOTS_DIR}/evaluation_dashboard.png")

# ── 5b. SHAP Summary Plot
print("      Computing SHAP values (this may take a moment)...")
explainer  = shap.TreeExplainer(model)
shap_vals  = explainer.shap_values(X_test)

fig2, ax5 = plt.subplots(figsize=(10, 7), facecolor="#0e1117")
shap.summary_plot(shap_vals, X_test, plot_type="bar", show=False,
                  color=TEAL, max_display=12)
plt.title("SHAP Feature Importance", color="#e8eaf0", fontfamily="monospace", pad=14)
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/shap_summary.png", dpi=150, bbox_inches="tight",
            facecolor="#0e1117")
plt.close()
print(f"      Saved: {PLOTS_DIR}/shap_summary.png")

# ── 5c. SHAP Beeswarm
fig3, ax6 = plt.subplots(figsize=(10, 7), facecolor="#0e1117")
shap.summary_plot(shap_vals, X_test, show=False, max_display=12)
plt.title("SHAP Beeswarm — Feature Impact Direction", color="#e8eaf0",
          fontfamily="monospace", pad=14)
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/shap_beeswarm.png", dpi=150, bbox_inches="tight",
            facecolor="#0e1117")
plt.close()
print(f"      Saved: {PLOTS_DIR}/shap_beeswarm.png")

# ── 5d. Churn probability distribution
fig4, ax7 = plt.subplots(figsize=(9, 5), facecolor="#0e1117")
ax7.hist(y_prob[y_test == 0], bins=40, alpha=0.6, color=TEAL,   label="No Churn", density=True)
ax7.hist(y_prob[y_test == 1], bins=40, alpha=0.6, color=CORAL,  label="Churn",    density=True)
ax7.axvline(0.5, color="#ffffff", lw=1, linestyle="--", alpha=0.4, label="Threshold 0.5")
ax7.set_xlabel("Predicted Churn Probability")
ax7.set_ylabel("Density")
ax7.set_title("Score Distribution by True Class", color="#e8eaf0")
ax7.legend()
ax7.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/score_distribution.png", dpi=150, bbox_inches="tight",
            facecolor="#0e1117")
plt.close()
print(f"      Saved: {PLOTS_DIR}/score_distribution.png")

# ─── 6. SAVE MODEL ────────────────────────────────────────────────────────────
print(f"[6/6] Saving model to {MODEL_PATH}...")
joblib.dump({"model": model, "features": list(X.columns)}, MODEL_PATH)
print(f"      Done.\n")

# ─── INFERENCE EXAMPLE ────────────────────────────────────────────────────────
print("── Inference Example ──────────────────────────────────────────────────")
sample = X_test.iloc[:5].copy()
probs  = model.predict_proba(sample)[:, 1]
for i, prob in enumerate(probs):
    label = "⚠ CHURN" if prob > 0.5 else "✓ RETAIN"
    print(f"  Customer {i+1}: {prob*100:.1f}% churn probability  →  {label}")

print("\n✓ Pipeline complete. Outputs:")
print(f"   Model  : {MODEL_PATH}")
print(f"   Plots  : {PLOTS_DIR}/")
print(f"   AUC    : {auc_score:.4f}")
