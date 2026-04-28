
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock

# from src.main import app  # Check if app is exposed here, or create a throwaway app with router
from src.prediction.presentation.api.routes import router, get_predictor, init_predictor
from src.prediction.application.predictor import CongestionPredictor
from src.prediction.presentation.api.schemas import PredictionInput, HistoricalDataPoint

# Setup standalone app for testing router if main app has complex deps
from fastapi import FastAPI
test_app = FastAPI()
test_app.include_router(router)

client = TestClient(test_app)

@pytest.fixture
def mock_predictor():
    predictor = MagicMock(spec=CongestionPredictor)
    return predictor

@pytest.fixture
def override_dependency(mock_predictor):
    test_app.dependency_overrides[get_predictor] = lambda: mock_predictor
    yield
    test_app.dependency_overrides = {}

def test_predict_traffic_success(mock_predictor, override_dependency):
    # Mock return value
    mock_result = MagicMock()
    mock_result.current_congestion_level = "Normal"
    mock_result.predicted_congestion_15min = "High"
    mock_result.predicted_congestion_30min = "Heavy"
    mock_result.predicted_vehicles_15min = 100
    mock_result.predicted_vehicles_30min = 150
    mock_result.predicted_congestion_45min = "Normal"
    mock_result.predicted_vehicles_45min = 80
    mock_result.confidence_score = 0.95
    mock_result.meta_info = {"model": "test_model"}
    # Add other required fields if schema validation is strict, or let it slide if Pydantic model is flexible
    # However, create a real object to satisfy Pydantic response model if needed? 
    # Or just dict if mock_result works.
    
    # Better to mock the methods return plain objects/dict compatible with Pydantic
    mock_predictor.predict_congestion.return_value = mock_result

    payload = {
        "camera_id": "CAM_001",
        "total_vehicles": 50,
        "occupancy_rate": 0.2,
        "flow_rate_per_min": 10,
        "avg_speed": 40,
        "avg_density": 10
    }
    
    response = client.post("/predictions/predict", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    assert data["alert"] is True # Normal -> High (worse)
    assert data["data"]["predicted_congestion_15min"] == "High"

def test_get_history_success(mock_predictor, override_dependency):
    # Mock history
    mock_predictor.get_traffic_history.return_value = [
        HistoricalDataPoint(
            timestamp="10:00", 
            total_vehicles=20, 
            congestion_level="Normal", 
            is_prediction=False
        )
    ]
    # Mock prediction (last log)
    mock_predictor.predict_future_from_last_log.return_value = {
        "current_congestion_level": "Normal",
        "predicted_congestion_15min": "Normal",
        "predicted_vehicles_15min": 20,
        "predicted_congestion_30min": "Normal",
        "predicted_vehicles_30min": 20,
        "predicted_congestion_45min": "Normal",
        "predicted_vehicles_45min": 20,
        "confidence_score": 0.9,
        "meta_info": {}
    }

    response = client.get("/predictions/history/CAM_001?interval=5")
    
    assert response.status_code == 200
    data = response.json()
    assert data["camera_id"] == "CAM_001"
    assert len(data["history"]) == 1
    assert data["prediction"]["predicted_congestion_15min"] == "Normal"

def test_predict_service_unavailable(override_dependency):
    # Test what happens if dependency raises or is None (though override handles injection)
    # To test actual 503 from get_predictor, we'd NOT override it and ensure it's not init.
    # But clean test_app might not have it init anyway.
    
    # Reset overrides to test default behavior
    test_app.dependency_overrides = {}
    
    # Manually ensure it's None (might be global state)
    # The routes.py uses a global variable. This is hard to reset if other tests touched it.
    # But we can try.
    
    response = client.post("/predictions/predict", json={
        "camera_id": "CAM_001", "total_vehicles": 0, "occupancy_rate":0, 
        "flow_rate_per_min":0, "avg_speed":0, "avg_density":0
    })
    
    # If not init, should be 503
    assert response.status_code == 503
