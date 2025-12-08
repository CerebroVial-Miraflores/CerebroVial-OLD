import torch
from typing import List, Tuple
from ..domain import VisionTrafficData, WazeTrafficData, NodeFeatures, TrafficGraph
import numpy as np

class TrafficDataLoader:
    """
    Handles data ingestion from various sources and formatting for the model.
    """
    def __init__(self, vision_source: VisionTrafficData, waze_source: WazeTrafficData):
        self.vision_source = vision_source
        self.waze_source = waze_source
        self._history_buffer: List[List[NodeFeatures]] = []
        
    def load_realtime_features(self) -> List[NodeFeatures]:
        """
        Fetches current data from sources and combines them into NodeFeatures.
        """
        # TODO: Implement actual data fetching and mapping logic
        # For now, returning dummy data
        # visual_metrics = self.vision_source.get_zone_metrics()
        # waze_data = self.waze_source.get_jams()
        
        # Placeholder
        return []

    def update_history(self, features: List[NodeFeatures], window_size: int = 12):
        """
        Updates the sliding window buffer of historical data.
        """
        self._history_buffer.append(features)
        if len(self._history_buffer) > window_size:
            self._history_buffer.pop(0)
            
    def get_sequence_tensor(self) -> torch.Tensor:
        """
        Returns the current history buffer as a tensor (Batch, Time, Nodes, Features).
        """
        if not self._history_buffer:
            return torch.empty(0)
            
        # Stack all features: (Time, Nodes, Features)
        sequence = torch.stack([
            torch.stack([n.to_tensor() for n in step]) 
            for step in self._history_buffer
        ])
        
        # Add batch dimension: (1, Time, Nodes, Features)
        return sequence.unsqueeze(0)
