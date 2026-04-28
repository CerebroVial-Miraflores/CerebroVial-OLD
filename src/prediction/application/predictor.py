from datetime import datetime, timedelta
import logging
from typing import Optional, List

from ..infrastructure.engine import TrafficModelEngine
from ..infrastructure.csv_loader import CSVLoader
from ..presentation.api.schemas import PredictionInput, PredictionOutput, HistoricalDataPoint

logger = logging.getLogger(__name__)

class CongestionPredictor:
    def __init__(self, model_dir: str = "models", data_dir: str = "data/traffic_logs"):
        self.engine = TrafficModelEngine(model_dir=model_dir)
        self.engine.load_models()
        self.csv_loader = CSVLoader(data_dir=data_dir)
        
    def predict_congestion(self, input_data: PredictionInput) -> PredictionOutput:
        """
        Main entry point for getting traffic predictions.
        """
        # Prepare features
        if input_data.hour is None:
            now = datetime.now()
            input_data.hour = now.hour
            input_data.day_of_week = now.weekday()
            
        features = input_data.model_dump()
        
        raw_predictions = self.engine.predict(features)
        
        # Map raw predictions to schema
        current = raw_predictions.get("predicted_congestion_current", "Unknown")
        p15 = raw_predictions.get("predicted_congestion_15min", "Normal") 
        v15 = raw_predictions.get("predicted_vehicles_15min", -1)
        p30 = raw_predictions.get("predicted_congestion_30min", "Normal")
        v30 = raw_predictions.get("predicted_vehicles_30min", -1)
        p45 = raw_predictions.get("predicted_congestion_45min", "Normal")
        v45 = raw_predictions.get("predicted_vehicles_45min", -1)
        
        return PredictionOutput(
            current_congestion_level=current,
            predicted_congestion_15min=p15,
            predicted_vehicles_15min=v15,
            predicted_congestion_30min=p30,
            predicted_vehicles_30min=v30,
            predicted_congestion_45min=p45,
            predicted_vehicles_45min=v45,
            confidence_score=0.85, # Placeholder
            meta_info={"model": "RandomForest_v2_Reg"}
        )

    def get_traffic_history(self, camera_id: str, interval: int = 5) -> List[HistoricalDataPoint]:
        """
        Retrieves aggregated traffic data sampled every 'interval' minutes.
        The window size is automatically calculated to maintain roughly 30 data points (interval * 30).
        """
        # Calculate window size based on desired points (approx 30)
        minutes = interval * 30
        
        # Get latest logs
        df = self.csv_loader.load_all_logs()
        if df.empty:
            return []
            
        try:
            # Filter by camera_id
            if 'camera_id' in df.columns:
                df = df[df['camera_id'] == camera_id]
                
            if df.empty:
                return []

            # Ensure valid timestamp
            df['timestamp'] = df['timestamp'].astype(float)
            df['datetime'] = df['timestamp'].apply(lambda x: datetime.fromtimestamp(x))
            df = df.set_index('datetime').sort_index()

            # Filter last N minutes
            now = datetime.now()
            start_time = now - timedelta(minutes=minutes)
            df = df[df.index >= start_time]

            # Resample to dynamic interval
            # Using 'min' as 'T' is deprecated
            resampled = df.resample(f'{interval}min').agg({
                'total_vehicles': 'mean',
                'occupancy_rate': 'mean'
            }).fillna(0) # or interpolate

            # Limit to exactly the requested window size if resampling produced more
            
            history = []
            for dt, row in resampled.iterrows():
                # Determine congestion level heuristic on AGGREGATED data
                occ = float(row.get('occupancy_rate', 0))
                level = "Normal"
                if occ > 0.7:
                    level = "Heavy"
                elif occ > 0.4:
                    level = "High"
                    
                ts_str = dt.strftime("%H:%M") # Just Hour:Minute for cleaner X-axis
                
                history.append(HistoricalDataPoint(
                    timestamp=ts_str,
                    total_vehicles=int(row.get('total_vehicles', 0)),
                    congestion_level=level,
                    is_prediction=False
                ))
                
            return history
            
        except Exception as e:
            logger.error(f"Error reading traffic log: {e}")
            return []

    def predict_future_from_last_log(self, camera_id: str) -> Optional[PredictionOutput]:
        """
        Retrieves the very last log entry for the camera and generates a prediction.
        """
        last_entry = self.csv_loader.get_latest_log_entry(camera_id)
        
        if not last_entry:
            return None
            
        try:
            # Construct Input
            input_data = PredictionInput(
                camera_id=camera_id,
                total_vehicles=float(last_entry.get('total_vehicles', 0)),
                occupancy_rate=float(last_entry.get('occupancy_rate', 0)),
                flow_rate_per_min=float(last_entry.get('flow_rate_per_min', 0)),
                avg_speed=float(last_entry.get('avg_speed', 0)),
                avg_density=float(last_entry.get('avg_density', 0)),
                hour=datetime.now().hour,
                day_of_week=datetime.now().weekday()
            )
            
            return self.predict_congestion(input_data)
            
        except Exception as e:
            logger.error(f"Error predicting from log: {e}")
            return None
