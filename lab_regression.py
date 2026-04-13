import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_absolute_error, r2_score,
    classification_report, ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt


def load_data(filepath="data/telecom_churn.csv"):
    df = pd.read_csv(filepath)
    return df


def split_data(df, target_col, test_size=0.2, random_state=42):
    X = df.drop(columns=[target_col])
    y = df[target_col]

    stratify = y if y.nunique() <= 10 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify
    )

    print("\nTrain size:", len(X_train))
    print("Test size:", len(X_test))
    print("Train churn rate:", y_train.mean())
    print("Test churn rate:", y_test.mean())

    return X_train, X_test, y_train, y_test


def build_logistic_pipeline():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(random_state=42, max_iter=1000, class_weight="balanced"))
    ])


def evaluate_classifier(pipeline, X_train, X_test, y_train, y_test):
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.show()

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred)
    }


def build_ridge_pipeline():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0))
    ])


def evaluate_regressor(pipeline, X_train, X_test, y_train, y_test):
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    return {
        "mae": mean_absolute_error(y_test, y_pred),
        "r2": r2_score(y_test, y_pred)
    }


def build_lasso_pipeline():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", Lasso(alpha=0.1))
    ])


def run_cross_validation(pipeline, X_train, y_train, cv=5):
    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=cv_splitter,
        scoring="accuracy"
    )

    print("\nCV scores per fold:", scores)
    print("Mean:", scores.mean())
    print("Std:", scores.std())

    return scores


if __name__ == "__main__":
    df = load_data()

    print("\nShape:", df.shape)
    print("\nMissing values:\n", df.isnull().sum())
    print("\nChurn distribution:\n", df["churned"].value_counts())

    numeric_features = [
        "tenure", "monthly_charges", "total_charges",
        "num_support_calls", "senior_citizen",
        "has_partner", "has_dependents"
    ]

    df_cls = df[numeric_features + ["churned"]].dropna()
    X_train, X_test, y_train, y_test = split_data(df_cls, "churned")

    log_pipe = build_logistic_pipeline()
    metrics = evaluate_classifier(log_pipe, X_train, X_test, y_train, y_test)
    print("\nLogistic Metrics:", metrics)

    run_cross_validation(log_pipe, X_train, y_train)

    df_reg = df[
        ["tenure", "total_charges", "num_support_calls",
         "senior_citizen", "has_partner", "has_dependents",
         "monthly_charges"]
    ].dropna()

    X_tr, X_te, y_tr, y_te = split_data(df_reg, "monthly_charges")

    ridge_pipe = build_ridge_pipeline()
    ridge_metrics = evaluate_regressor(ridge_pipe, X_tr, X_te, y_tr, y_te)

    print("\nRidge Metrics:", ridge_metrics)

    lasso_pipe = build_lasso_pipeline()
    lasso_pipe.fit(X_tr, y_tr)

    ridge_model = ridge_pipe.named_steps["model"]
    lasso_model = lasso_pipe.named_steps["model"]

    print("\nFeature Coefficients Comparison:")
    for i, col in enumerate(X_tr.columns):
        print(col, "| Ridge:", ridge_model.coef_[i], "| Lasso:", lasso_model.coef_[i])



    """
TASK 7 - SUMMARY OF FINDINGS

1. Most important features for predicting churn:
The most important features are tenure, monthly_charges, and num_support_calls.

2. Model performance:
The logistic regression model performs reasonably well with balanced class weights.
Recall is more important than precision because we care about detecting customers who will churn.

3. Recommendations:
Improve performance using feature engineering, hyperparameter tuning,
and more advanced models such as Random Forest or XGBoost.
"""