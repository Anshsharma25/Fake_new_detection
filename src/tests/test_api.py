from fastapi.testclient import TestClient
from src.api.main import app
from src.config.logger import logger

client = TestClient(app)

def test_home():
    logger.info("Testing home endpoint...")
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Fake News Detection API"}

def test_predict():
    logger.info("Testing prediction endpoint...")
    response = client.post("/predict", json={"text": "This is a test news article."})
    assert response.status_code == 200
    assert "prediction" in response.json()
