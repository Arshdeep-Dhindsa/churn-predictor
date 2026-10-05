"""Step 1 - Data collection.

Simulates pulling customer records from a source system, saves the raw CSV
and loads it into SQLite (the "data warehouse" the training step reads from).
Swap `generate_customers` for a real source (API, CSV export, SQL query).
"""
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "data" / "customers_raw.csv"
DB_PATH = ROOT / "data" / "churn.db"


def generate_customers(n: int = 6000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    tenure = rng.integers(1, 73, n)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.25, 0.20])
    internet = rng.choice(["DSL", "Fiber", "None"], n, p=[0.35, 0.45, 0.20])
    payment = rng.choice(["Card", "UPI", "Bank transfer", "Cash"], n, p=[0.3, 0.35, 0.2, 0.15])
    tech_support = rng.choice(["Yes", "No"], n, p=[0.4, 0.6])
    paperless = rng.choice(["Yes", "No"], n, p=[0.6, 0.4])
    age = rng.integers(18, 80, n)
    support_calls = rng.poisson(1.5, n)

    base = {"DSL": 35, "Fiber": 70, "None": 20}
    monthly = np.array([base[i] for i in internet]) + rng.normal(0, 8, n)
    monthly = np.clip(monthly, 10, None).round(2)

    # Hidden "true" churn process (the model has to rediscover this)
    z = (
        -1.0
        - 0.045 * tenure
        + 0.025 * (monthly - 50)
        + 1.2 * (contract == "Month-to-month")
        - 0.9 * (contract == "Two year")
        + 0.5 * (internet == "Fiber")
        - 0.6 * (tech_support == "Yes")
        + 0.25 * support_calls
        + 0.3 * (payment == "Cash")
        + 0.01 * (50 - age)
        + rng.normal(0, 0.6, n)
    )
    churn = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)

    df = pd.DataFrame({
        "customer_id": [f"C{100000 + i}" for i in range(n)],
        "age": age, "tenure_months": tenure, "monthly_charges": monthly,
        "contract": contract, "internet_service": internet,
        "payment_method": payment, "tech_support": tech_support,
        "paperless_billing": paperless, "support_calls": support_calls,
        "churn": churn,
    })
    # Realistic mess: missing values + duplicate rows to clean later
    for col in ["monthly_charges", "age", "payment_method"]:
        df.loc[rng.random(n) < 0.03, col] = np.nan
    return pd.concat([df, df.sample(40, random_state=seed)], ignore_index=True)


def main() -> None:
    df = generate_customers()
    RAW_CSV.parent.mkdir(exist_ok=True)
    df.to_csv(RAW_CSV, index=False)
    with sqlite3.connect(DB_PATH) as conn:
        df.to_sql("customers", conn, if_exists="replace", index=False)
    print(f"Collected {len(df)} rows -> {RAW_CSV.name}, {DB_PATH.name}")


if __name__ == "__main__":
    main()
