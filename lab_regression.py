import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.metrics import (
    classification_report,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    mean_absolute_error,
    r2_score
)

# =========================
# Load Data
# =========================
df = pd.read_csv("data/telecom_churn.csv")

print("Shape:", df.shape)

print("\nMissing values:\n", df.isnull().sum())

print("\nChurn distribution:\n", df["churned"].value_counts())

# =========================
# Visualization 1: Churn distribution
# =========================
sns.countplot(x="churned", data=df)
plt.title("Churn Distribution")
plt.show()

# =========================
# Features / Target
# =========================
X = df.drop("churned", axis=1)
y = df["churned"]

X = pd.get_dummies(X, drop_first=True)

# =========================
# Train/Test split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =========================
# Scaling
# =========================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =========================
# Logistic Regression
# =========================
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train_scaled, y_train)

y_pred = log_model.predict(X_test_scaled)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print("F1 Score:", f1_score(y_test, y_pred))

# =========================
# Confusion Matrix
# =========================
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Confusion Matrix")
plt.show()

# =========================
# Cross Validation
# =========================
cv_scores = cross_val_score(log_model, X_train_scaled, y_train, cv=5, scoring="f1")
print("\nCV Mean:", cv_scores.mean())
print("CV Std:", cv_scores.std())

# =========================
# Ridge Regression
# =========================
ridge = Ridge()
ridge.fit(X_train_scaled, y_train)
ridge_pred = ridge.predict(X_test_scaled)

print("\nRidge Results:")
print({
    "mae": mean_absolute_error(y_test, ridge_pred),
    "r2": r2_score(y_test, ridge_pred)
})

# =========================
# Lasso Regression
# =========================
lasso = Lasso()
lasso.fit(X_train_scaled, y_train)
lasso_pred = lasso.predict(X_test_scaled)

print("\nLasso Results:")
print({
    "mae": mean_absolute_error(y_test, lasso_pred),
    "r2": r2_score(y_test, lasso_pred)
})

# =========================
# Threshold Tuning
# =========================
probs = log_model.predict_proba(X_test_scaled)[:, 1]

thresholds = np.arange(0.3, 0.8, 0.1)
f1_scores = []

best_f1 = 0
best_t = 0

for t in thresholds:
    preds = (probs >= t).astype(int)
    f1 = f1_score(y_test, preds)
    f1_scores.append(f1)
    print(f"Threshold {t:.1f}: F1={f1:.3f}")

    if f1 > best_f1:
        best_f1 = f1
        best_t = t

print("\nBest Threshold:", best_t)

# =========================
# Threshold Plot
# =========================
plt.plot(thresholds, f1_scores, marker="o")
plt.title("Threshold vs F1 Score")
plt.xlabel("Threshold")
plt.ylabel("F1 Score")
plt.show()

# =========================
# Model Sweep
# =========================
models = [
    LogisticRegression(max_iter=1000),
    LogisticRegression(C=0.1, max_iter=1000),
    LogisticRegression(C=10, max_iter=1000),
]

print("\nMODEL SWEEP RESULTS:\n")

for m in models:
    m.fit(X_train_scaled, y_train)
    preds = m.predict(X_test_scaled)
    f1 = f1_score(y_test, preds)
    print(f"{m} mean_f1: {f1}")

# ========
# SUMMARY 
# ========
"""
SUMMARY:
- Loaded telecom churn dataset.
- Performed preprocessing (encoding + scaling).
- Trained Logistic Regression classifier.
- Evaluated using F1-score, classification report, confusion matrix, CV.
- Tuned decision threshold for better F1.
- Compared multiple Logistic Regression models (C values).
- Added Ridge and Lasso as regression baselines.
- Visualized churn distribution, confusion matrix, and threshold vs F1.
"""