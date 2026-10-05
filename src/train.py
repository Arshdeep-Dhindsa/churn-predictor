"""Steps 2-4 - Preprocessing, model selection, evaluation, export."""
import json
import sqlite3
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "churn.db"
MODEL_PATH = ROOT / "models" / "churn_model.joblib"
METRICS_PATH = ROOT / "models" / "metrics.json"

NUMERIC = ["age", "tenure_months", "monthly_charges", "support_calls"]
CATEGORICAL = ["contract", "internet_service", "payment_method",
               "tech_support", "paperless_billing"]
FEATURES = NUMERIC + CATEGORICAL
TARGET = "churn"


def load_data() -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql("SELECT * FROM customers", conn)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates(subset="customer_id")
    df = df.dropna(subset=[TARGET])
    df = df[df["age"].isna() | df["age"].between(18, 100)]
    return df


def build_preprocessor() -> ColumnTransformer:
    # Imputation lives INSIDE the pipeline, so serving applies the same logic
    num = Pipeline([("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler())])
    cat = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("num", num, NUMERIC), ("cat", cat, CATEGORICAL)])


def main() -> None:
    df = clean(load_data())
    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    candidates = {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "random_forest": RandomForestClassifier(
            n_estimators=200, min_samples_leaf=5, random_state=42, n_jobs=-1),
        "gradient_boosting": GradientBoostingClassifier(random_state=42),
    }

    cv_scores = {}
    for name, clf in candidates.items():
        pipe = Pipeline([("prep", build_preprocessor()), ("model", clf)])
        cv_scores[name] = cross_val_score(
            pipe, X_train, y_train, cv=5, scoring="roc_auc").mean()
        print(f"{name:22s} CV ROC-AUC = {cv_scores[name]:.3f}")

    best = max(cv_scores, key=cv_scores.get)
    final = Pipeline([("prep", build_preprocessor()), ("model", candidates[best])])
    final.fit(X_train, y_train)

    proba = final.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    report = classification_report(y_test, pred, output_dict=True)
    metrics = {
        "best_model": best,
        "cv_roc_auc": {k: round(float(v), 4) for k, v in cv_scores.items()},
        "test_roc_auc": round(float(roc_auc_score(y_test, proba)), 4),
        "test_accuracy": round(report["accuracy"], 4),
        "test_precision_churn": round(report["1"]["precision"], 4),
        "test_recall_churn": round(report["1"]["recall"], 4),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        "train_rows": len(X_train), "test_rows": len(X_test),
    }

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(final, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(f"\nBest: {best}\n{json.dumps(metrics, indent=2)}")


if __name__ == "__main__":
    main()
