"""Step 5 - Serving the model with FastAPI."""
import json
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "churn_model.joblib"
METRICS_PATH = ROOT / "models" / "metrics.json"

app = FastAPI(title="Customer Churn Predictor", version="1.0.0",
              description="Predicts the probability that a customer will churn.")

model = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None


class Customer(BaseModel):
    age: int = Field(..., ge=18, le=100, examples=[34])
    tenure_months: int = Field(..., ge=0, le=120, examples=[5])
    monthly_charges: float = Field(..., ge=0, examples=[79.5])
    support_calls: int = Field(..., ge=0, le=50, examples=[3])
    contract: Literal["Month-to-month", "One year", "Two year"] = "Month-to-month"
    internet_service: Literal["DSL", "Fiber", "None"] = "Fiber"
    payment_method: Literal["Card", "UPI", "Bank transfer", "Cash"] = "UPI"
    tech_support: Literal["Yes", "No"] = "No"
    paperless_billing: Literal["Yes", "No"] = "Yes"


class Prediction(BaseModel):
    churn_probability: float
    churn_prediction: bool
    risk_level: Literal["Low", "Medium", "High"]


def _risk(p: float) -> str:
    return "High" if p >= 0.6 else "Medium" if p >= 0.3 else "Low"


@app.get("/health")
def health():
    return {"status": "ok" if model is not None else "model_missing"}


@app.get("/metrics")
def metrics():
    if not METRICS_PATH.exists():
        raise HTTPException(404, "Run `python -m src.train` first.")
    return json.loads(METRICS_PATH.read_text())


@app.post("/predict", response_model=Prediction)
def predict(customer: Customer):
    if model is None:
        raise HTTPException(503, "Model not loaded. Run `python -m src.train`.")
    row = pd.DataFrame([customer.model_dump()])
    p = float(model.predict_proba(row)[0, 1])
    return Prediction(churn_probability=round(p, 4),
                      churn_prediction=p >= 0.5, risk_level=_risk(p))


PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Churn Predictor</title>
<style>
body{font-family:system-ui,sans-serif;max-width:520px;margin:0 auto;padding:16px;background:#f6f7f9;color:#1b1f24}
h1{font-size:1.3rem}label{display:block;margin:10px 0 3px;font-size:.85rem;color:#555}
input,select{width:100%;padding:9px;border:1px solid #ccd;border-radius:8px;font-size:1rem;box-sizing:border-box}
button{margin-top:16px;width:100%;padding:12px;border:0;border-radius:8px;background:#2d5bff;color:#fff;font-size:1rem}
#out{margin-top:16px;padding:14px;border-radius:10px;background:#fff;display:none}
.Low{color:#128a3e}.Medium{color:#b7791f}.High{color:#c53030}
a{color:#2d5bff}
</style></head><body>
<h1>Customer Churn Predictor</h1>
<form id="f">
<label>Age</label><input name="age" type="number" value="34" required>
<label>Tenure (months)</label><input name="tenure_months" type="number" value="5" required>
<label>Monthly charges</label><input name="monthly_charges" type="number" step="0.01" value="79.5" required>
<label>Support calls</label><input name="support_calls" type="number" value="3" required>
<label>Contract</label><select name="contract"><option>Month-to-month</option><option>One year</option><option>Two year</option></select>
<label>Internet service</label><select name="internet_service"><option>Fiber</option><option>DSL</option><option>None</option></select>
<label>Payment method</label><select name="payment_method"><option>UPI</option><option>Card</option><option>Bank transfer</option><option>Cash</option></select>
<label>Tech support</label><select name="tech_support"><option>No</option><option>Yes</option></select>
<label>Paperless billing</label><select name="paperless_billing"><option>Yes</option><option>No</option></select>
<button>Predict churn</button></form>
<div id="out"></div>
<p><a href="/docs">API docs</a> · <a href="/metrics">Model metrics</a></p>
<script>
const NUM=["age","tenure_months","monthly_charges","support_calls"];
document.getElementById("f").addEventListener("submit",async e=>{
  e.preventDefault();
  const d=Object.fromEntries(new FormData(e.target));
  NUM.forEach(k=>d[k]=Number(d[k]));
  const out=document.getElementById("out");out.style.display="block";out.textContent="Predicting...";
  const r=await fetch("/predict",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(d)});
  const j=await r.json();
  if(!r.ok){out.textContent=JSON.stringify(j.detail);return}
  out.innerHTML=`<b>Churn probability: ${(j.churn_probability*100).toFixed(1)}%</b><br>Risk: <b class="${j.risk_level}">${j.risk_level}</b>`;
});
</script></body></html>"""


@app.get("/", response_class=HTMLResponse)
def home():
    return PAGE
