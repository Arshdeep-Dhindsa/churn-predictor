from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

HIGH_RISK = {"age": 25, "tenure_months": 2, "monthly_charges": 90, "support_calls": 6,
             "contract": "Month-to-month", "internet_service": "Fiber",
             "payment_method": "Cash", "tech_support": "No", "paperless_billing": "Yes"}
LOW_RISK = {"age": 60, "tenure_months": 60, "monthly_charges": 30, "support_calls": 0,
            "contract": "Two year", "internet_service": "DSL",
            "payment_method": "Card", "tech_support": "Yes", "paperless_billing": "No"}


def test_health():
    assert client.get("/health").json()["status"] == "ok"


def test_risk_ordering():
    hi = client.post("/predict", json=HIGH_RISK).json()
    lo = client.post("/predict", json=LOW_RISK).json()
    assert hi["churn_probability"] > lo["churn_probability"]


def test_validation_error():
    bad = {**HIGH_RISK, "age": 5}
    assert client.post("/predict", json=bad).status_code == 422
