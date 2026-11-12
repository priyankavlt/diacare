import pytest

from app.app import app
from diacare.config import MODEL_PATH
from diacare.predict import predict
from diacare.preprocessing import load_data

pytestmark = pytest.mark.skipif(not MODEL_PATH.exists(), reason="run `python -m diacare.train` first")

HEALTHY = {"Pregnancies": 1, "Glucose": 85, "BloodPressure": 66, "SkinThickness": 29,
           "Insulin": 0, "BMI": 26.6, "DPF": 0.351, "Age": 31}
AT_RISK = {"Pregnancies": 8, "Glucose": 183, "BloodPressure": 64, "SkinThickness": 0,
           "Insulin": 0, "BMI": 23.3, "DPF": 0.672, "Age": 32}


def test_data_is_deduplicated():
    df = load_data()
    assert not df.duplicated().any()
    assert len(df) == 744


def test_predictions_are_sensible():
    healthy, p_healthy = predict(HEALTHY)
    at_risk, p_risk = predict(AT_RISK)
    assert not healthy and at_risk
    assert 0 <= p_healthy < p_risk <= 1


def test_web_app():
    client = app.test_client()
    assert client.get("/").status_code == 200
    form = {"pregnancies": 8, "glucose": 183, "bloodpressure": 64, "skinthickness": 0,
            "insulin": 0, "bmi": 23.3, "dpf": 0.672, "age": 32}
    r = client.post("/predict", data=form)
    assert r.status_code == 200 and b"Higher risk" in r.data
    r = client.post("/predict", data={**form, "glucose": "abc"})
    assert r.status_code == 400
