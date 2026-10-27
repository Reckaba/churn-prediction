import pytest
from fastapi.testclient import TestClient

from app.main import EXAMPLE, app, model
from churn.pipeline import FEATURES, load_data

client = TestClient(app)

NEW_CUSTOMER = {**EXAMPLE, "tenure": 1, "Contract": "Month-to-month",
                "InternetService": "Fiber optic", "PaymentMethod": "Electronic check"}
LOYAL_CUSTOMER = {**EXAMPLE, "tenure": 72, "Contract": "Two year", "InternetService": "DSL",
                  "PaymentMethod": "Bank transfer (automatic)", "OnlineSecurity": "Yes",
                  "TechSupport": "Yes", "MonthlyCharges": 60.0}


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_returns_probability():
    response = client.post("/predict", json=EXAMPLE)
    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["high_risk"] == (body["churn_probability"] >= body["threshold"])


def test_new_customer_riskier_than_loyal_customer():
    new = client.post("/predict", json=NEW_CUSTOMER).json()["churn_probability"]
    loyal = client.post("/predict", json=LOYAL_CUSTOMER).json()["churn_probability"]
    assert new > 0.5 > loyal


@pytest.mark.parametrize("field, value", [("Contract", "Monthly"), ("tenure", -1), ("MonthlyCharges", 0)])
def test_invalid_input_rejected(field, value):
    response = client.post("/predict", json={**EXAMPLE, field: value})
    assert response.status_code == 422


def test_missing_field_rejected():
    payload = {k: v for k, v in EXAMPLE.items() if k != "Contract"}
    assert client.post("/predict", json=payload).status_code == 422


def test_api_matches_model_on_real_rows():
    """Guards against train/serve skew: the API must reproduce the pipeline's predictions."""
    X, _ = load_data()
    sample = X[FEATURES].sample(50, random_state=0)
    expected = model.predict_proba(sample)[:, 1]
    for record, p in zip(sample.to_dict(orient="records"), expected):
        got = client.post("/predict", json=record).json()["churn_probability"]
        assert got == pytest.approx(p, abs=1e-4)
