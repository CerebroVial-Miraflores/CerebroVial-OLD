from fastapi import APIRouter, HTTPException, Depends
from typing import List, Optional

from ...application.predictor import CongestionPredictor
from .schemas import PredictionInput, PredictionResponse, HistoryResponse

# Pydantic Schemas for API Response Wrapper
# Note: PredictionResponse is defined in schemas but we might need a wrapper for alerts as per legacy

# Router definition
router = APIRouter(prefix="/predictions", tags=["predictions"])

# Dependency Injection Placeholder
_predictor_instance: Optional[CongestionPredictor] = None

def get_predictor() -> CongestionPredictor:
    if _predictor_instance is None:
        raise HTTPException(status_code=503, detail="Predictor service not initialized")
    return _predictor_instance

def init_predictor(predictor: CongestionPredictor):
    global _predictor_instance
    _predictor_instance = predictor

@router.post("/predict", response_model=PredictionResponse)
async def predict_traffic(input_data: PredictionInput, predictor: CongestionPredictor = Depends(get_predictor)):
    """
    Get congestion prediction based on input features.
    """
    try:
        result = predictor.predict_congestion(input_data)
        
        # Logic for notification/alert (Ported from legacy)
        severity = {"Normal": 0, "High": 1, "Heavy": 2}
        
        current_sev = severity.get(result.current_congestion_level, 0)
        future_sev_15 = severity.get(result.predicted_congestion_15min, 0)
        future_sev_30 = severity.get(result.predicted_congestion_30min, 0)
        
        alert = False
        msg = "Traffic situation is stable."
        
        if future_sev_15 > current_sev:
            alert = True
            msg = f"Warning: Traffic expected to worsen to {result.predicted_congestion_15min} in 15 mins."
        elif future_sev_30 > current_sev:
            alert = True
            msg = f"Caution: Traffic expected to worsen to {result.predicted_congestion_30min} in 30 mins."
        elif current_sev == 2 and future_sev_15 == 2:
            alert = True
            msg = "Alert: Heavy traffic persisting."
            
        # We need a schema for this wrapper or define it inline/schemas
        # Using a dict for now compatible with Pydantic if defined properly or return JSON
        return {
            "data": result,
            "alert": alert,
            "message": msg
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/history/{camera_id}", response_model=HistoryResponse)
async def get_history(
    camera_id: str, 
    interval: int = 5, 
    predictor: CongestionPredictor = Depends(get_predictor)
):
    """
    Get historical data and a future prediction based on the last log.
    Interval can be specified in minutes (1, 2, 5, 10, 15).
    """
    try:
        history = predictor.get_traffic_history(camera_id, interval=interval)
        prediction = predictor.predict_future_from_last_log(camera_id)
        
        return HistoryResponse(
            camera_id=camera_id, 
            history=history, 
            prediction=prediction
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
