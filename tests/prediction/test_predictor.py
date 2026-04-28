import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime
import pandas as pd

from src.prediction.application.predictor import CongestionPredictor
from src.prediction.presentation.api.schemas import PredictionInput

@pytest.fixture
def mock_engine():
    with patch('src.prediction.application.predictor.TrafficModelEngine') as MockEngine:
        engine_instance = MockEngine.return_value
        # Mock predict return
        engine_instance.predict.return_value = {
            "predicted_congestion_current": "Normal",
            "predicted_congestion_15min": "High",
            "predicted_vehicles_15min": 100,
            "predicted_congestion_30min": "Heavy",
            "predicted_vehicles_30min": 150,
            "predicted_congestion_45min": "Normal",
            "predicted_vehicles_45min": 80
        }
        yield engine_instance

@pytest.fixture
def mock_loader():
    with patch('src.prediction.application.predictor.CSVLoader') as MockLoader:
        loader_instance = MockLoader.return_value
        yield loader_instance

@pytest.fixture
def predictor(mock_engine, mock_loader):
    return CongestionPredictor()

def test_predict_congestion_structure(predictor, mock_engine):
    """Test that predict_congestion returns correct structure and values mapped from engine."""
    input_data = PredictionInput(
        camera_id="CAM_001",
        total_vehicles=50,
        occupancy_rate=0.2,
        flow_rate_per_min=10,
        avg_speed=40,
        avg_density=10,
        hour=10,
        day_of_week=0
    )
    
    output = predictor.predict_congestion(input_data)
    
    # Check that engine.predict was called with correct dict
    mock_engine.predict.assert_called_once()
    
    # Check mapping
    assert output.current_congestion_level == "Normal"
    assert output.predicted_congestion_15min == "High"
    assert output.predicted_vehicles_15min == 100
    assert output.meta_info['model'] == "RandomForest_v2_Reg"

def test_get_traffic_history_empty(predictor, mock_loader):
    """Test behavior when no logs are found."""
    mock_loader.load_all_logs.return_value = pd.DataFrame()
    
    history = predictor.get_traffic_history("CAM_001")
    assert history == []

def test_get_traffic_history_filtering(predictor, mock_loader):
    """Test that it filters by camera_id and aggregates correctly."""
    # Create sample dataframe
    data = {
        'camera_id': ['CAM_001', 'CAM_001', 'CAM_002'],
        'timestamp': [
            datetime.now().timestamp(), 
            (datetime.now()).timestamp() - 60, # 1 min ago
            datetime.now().timestamp()
        ],
        'total_vehicles': [10, 20, 50],
        'occupancy_rate': [0.1, 0.2, 0.5]
    }
    df = pd.DataFrame(data)
    mock_loader.load_all_logs.return_value = df
    
    history = predictor.get_traffic_history("CAM_001", interval=1)
    
    assert len(history) > 0
    # Should filtered out CAM_002
    assert all(h.total_vehicles != 50 for h in history) 
    # Check structure
    assert history[0].congestion_level in ["Normal", "High", "Heavy"]

def test_predict_future_from_last_log_success(predictor, mock_loader, mock_engine):
    """Test predicting from the last available log."""
    mock_loader.get_latest_log_entry.return_value = {
        'total_vehicles': 100,
        'occupancy_rate': 0.5,
        'flow_rate_per_min': 20,
        'avg_speed': 30,
        'avg_density': 15
    }
    
    output = predictor.predict_future_from_last_log("CAM_001")
    
    assert output is not None
    assert output.predicted_congestion_15min == "High" # Mapped from mock_engine

def test_predict_future_from_last_log_none(predictor, mock_loader):
    """Test handling when no last log exists."""
    mock_loader.get_latest_log_entry.return_value = None
    
    output = predictor.predict_future_from_last_log("CAM_001")
    assert output is None
