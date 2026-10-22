import pytest
from fastapi.testclient import TestClient
from src.main import app
from src.database.connection import Base, engine
from src.services.ml_service import MODEL_PATH, FEATURES, load_model_on_startup, train_model


@pytest.fixture(scope="session", autouse=True)
def setup_database_and_model():
    """Garante schema atualizado no SQLite e artefato de Pipeline compatível com as features atuais."""
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)

    retrain = not MODEL_PATH.exists()
    if not retrain:
        try:
            model = load_model_on_startup()
            if getattr(model, "n_features_in_", None) != len(FEATURES):
                retrain = True
        except Exception:
            retrain = True

    if retrain:
        train_model(n_estimators=50, max_depth=12)


@pytest.fixture(scope="module")
def client():
    """Inicializa o TestClient executando o ciclo de vida (lifespan) da aplicação FastAPI."""
    with TestClient(app) as test_client:
        yield test_client


def test_root_endpoint(client: TestClient):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert data["docs_url"] == "/docs"


def test_health_check_endpoint(client: TestClient):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert data["model_loaded"] is True


def test_predict_success(client: TestClient):
    payload = {
        "machine_id": 1,
        "model_type": "model1",
        "volt": 175.5,
        "rotate": 425.0,
        "pressure": 102.3,
        "vibration": 43.1,
        "age": 12.0,
        "recent_errors": 1,
        "days_since_maint": 10.5
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["machine_id"] == 1
    assert "predicted_rul_hours" in data
    assert data["predicted_rul_hours"] > 0
    assert data["status"] in ["CRÍTICO", "ALERTA", "NORMAL"]
    assert "recommendation" in data


def test_predict_v1_alias_success(client: TestClient):
    payload = {
        "machine_id": 2,
        "model_type": "model3",
        "volt": 168.0,
        "rotate": 440.0,
        "pressure": 99.0,
        "vibration": 39.5
    }
    response = client.post("/v1/predict", json=payload)
    assert response.status_code == 200
    assert response.json()["machine_id"] == 2


def test_predict_idempotency(client: TestClient):
    payload = {
        "machine_id": 77,
        "volt": 180.0,
        "rotate": 410.0,
        "pressure": 108.0,
        "vibration": 44.0,
        "age": 9.0,
        "recent_errors": 2,
        "days_since_maint": 20.0
    }
    headers = {"Idempotency-Key": "sensor-event-77-20261020T2300"}

    # Primeira chamada grava no banco
    resp1 = client.post("/predict", json=payload, headers=headers)
    assert resp1.status_code == 200

    # Segunda chamada (retry com mesma chave) retorna exatamente o mesmo resultado sem duplicar no banco
    resp2 = client.post("/predict", json=payload, headers=headers)
    assert resp2.status_code == 200
    assert resp1.json() == resp2.json()

    # Verifica no histórico que apenas 1 registro foi gravado para a máquina 77
    logs_resp = client.get("/predictions?machine_id=77")
    assert logs_resp.status_code == 200
    logs = logs_resp.json()
    assert len(logs) == 1
    assert logs[0]["idempotency_key"] == "sensor-event-77-20261020T2300"


def test_list_predictions_endpoint(client: TestClient):
    response = client.get("/predictions?machine_id=1&limit=10")
    assert response.status_code == 200
    logs = response.json()
    assert isinstance(logs, list)
    assert len(logs) >= 1
    assert logs[0]["machine_id"] == 1
    assert "predicted_rul_hours" in logs[0]
    assert "timestamp" in logs[0]


def test_predict_negative_physical_values_rejected(client: TestClient):
    payload = {
        "machine_id": 1,
        "volt": -170.0,
        "rotate": 425.0,
        "pressure": 100.0,
        "vibration": 40.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_invalid_payload(client: TestClient):
    payload = {"machine_id": 1}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
