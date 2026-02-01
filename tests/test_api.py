from fastapi.testclient import TestClient

from api.app import app


def test_health():
    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"


def test_predict():
    payload = {
        "features": {
            "age": 0.05,
            "sex": -0.04,
            "bmi": 0.06,
            "bp": 0.02,
            "s1": -0.04,
            "s2": -0.03,
            "s3": -0.02,
            "s4": -0.01,
            "s5": 0.04,
            "s6": 0.03,
        }
    }

    with TestClient(app) as client:
        r = client.post("/predict", json=payload)
        # If it fails, print the server's error detail
        assert r.status_code == 200, r.text
        body = r.json()
        assert "prediction" in body
        assert isinstance(body["prediction"], (int, float))
