from pydantic import BaseModel, Field
from typing import Optional, List

class PredictionInput(BaseModel):
    """
    Input features for the traffic prediction model.
    Matches the columns available in the CSV:
    total_vehicles, occupancy_rate, flow_rate_per_min, avg_speed, avg_density
    """
    camera_id: str = Field(..., description="ID of the camera/node")
    total_vehicles: float = Field(..., description="Total count of vehicles")
    occupancy_rate: float = Field(..., description="Occupancy rate (0.0 to 1.0)")
    flow_rate_per_min: float = Field(..., description="Flow rate (vehicles per minute)")
    avg_speed: float = Field(..., description="Average speed in km/h")
    avg_density: float = Field(..., description="Average density")
    
    # Optional context features
    hour: Optional[int] = Field(None, description="Hour of the day (0-23)")
    day_of_week: Optional[int] = Field(None, description="Day of the week (0-6)")

class PredictionOutput(BaseModel):
    """
    Structured output for traffic predictions.
    """
    current_congestion_level: str = Field(..., description="Current detected congestion: Normal, High, Heavy")
    predicted_congestion_15min: str = Field(..., description="Predicted congestion in 15 mins")
    predicted_vehicles_15min: int = Field(..., description="Predicted vehicle count in 15 mins")
    predicted_congestion_30min: str = Field(..., description="Predicted congestion in 30 mins")
    predicted_vehicles_30min: int = Field(..., description="Predicted vehicle count in 30 mins")
    predicted_congestion_45min: str = Field(..., description="Predicted congestion in 45 mins")
    predicted_vehicles_45min: int = Field(..., description="Predicted vehicle count in 45 mins")
    confidence_score: float = Field(..., description="Model confidence score (0.0 to 1.0)")
    meta_info: dict = Field(default_factory=dict, description="Debug or extra info")

class HistoricalDataPoint(BaseModel):
    timestamp: str
    total_vehicles: int
    congestion_level: str
    is_prediction: bool = False

class HistoryResponse(BaseModel):
    camera_id: str
    history: List[HistoricalDataPoint]
    prediction: Optional[PredictionOutput] = None

class PredictionResponse(BaseModel):
    data: PredictionOutput
    alert: bool = Field(False, description="True if predicted congestion is worse than current")
    message: str = Field("Situation stable", description="Human readable alert message")
