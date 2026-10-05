# Customer Churn Predictor — End-to-End Data Science Project

Predict whether a telecom/subscription customer will churn, and serve the model
through a FastAPI REST API with a small web UI.

## Pipeline

| Stage | File | What happens |
|---|---|---|
| 1. Collect | `src/data_collection.py` | Generates ~6k customer records, saves CSV, loads into SQLite |
| 2. Preprocess | `src/train.py` | Dedupe, drop bad rows, impute, scale, one-hot (inside a sklearn `Pipeline`) |
| 3. Model | `src/train.py` | 5-fold CV on Logistic Regression, Random Forest, Gradient Boosting; best by ROC-AUC |
| 4. Evaluate | `models/metrics.json` | ROC-AUC, precision, recall, confusion matrix on held-out test set |
| 5. Deploy | `app/main.py` | FastAPI: `/predict`, `/health`, `/metrics`, `/docs`, web form at `/` |

## Tech stack

- Python 3.11, pandas, NumPy, scikit-learn, joblib
- SQLite (data store)
- FastAPI + Pydantic (API and input validation), Uvicorn (server)
- pytest + httpx (tests)
- Docker, Render (deployment)

## Local setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m src.data_collection    # step 1
python -m src.train              # steps 2-4
uvicorn app.main:app --reload    # step 5 -> http://127.0.0.1:8000
pytest                           # optional
```

## Example request

```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{
  "age": 25, "tenure_months": 2, "monthly_charges": 90, "support_calls": 6,
  "contract": "Month-to-month", "internet_service": "Fiber",
  "payment_method": "Cash", "tech_support": "No", "paperless_billing": "Yes"}'
```

## Deploy (free) on Render

1. Push this folder to a GitHub repo.
2. Render → New → Web Service → connect the repo → Runtime: **Docker**.
3. Deploy. Your public URL serves the web UI at `/` and Swagger docs at `/docs`.

(Alternative: Hugging Face Spaces with the Docker SDK, set `app_port: 8000`.)

## Using real data

Replace `generate_customers()` with a real source, e.g. the Kaggle
*Telco Customer Churn* CSV, and update the column lists in `src/train.py`.

## Ideas to extend

- Hyperparameter tuning (`RandomizedSearchCV`), threshold tuning for recall
- SHAP explanations per prediction
- MLflow experiment tracking, GitHub Actions CI
