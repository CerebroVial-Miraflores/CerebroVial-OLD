from typing import List, Optional
from ..domain import CongestionPrediction

class PredictionRepository:
    """
    Repository for storing and retrieving congestion predictions.
    """
    def __init__(self):
        self._storage: List[CongestionPrediction] = []

    def save_prediction(self, prediction: CongestionPrediction):
        """
        Saves a single prediction.
        """
        # In a real app, this would write to DB
        self._storage.append(prediction)
        # Keep only recent predictions in memory
        if len(self._storage) > 1000:
            self._storage.pop(0)
            
    def get_latest_prediction(self, intersection_id: str) -> Optional[CongestionPrediction]:
        """
        Retrieves the most recent prediction for a given intersection.
        """
        for pred in reversed(self._storage):
            if pred.intersection_id == intersection_id:
                return pred
        return None
