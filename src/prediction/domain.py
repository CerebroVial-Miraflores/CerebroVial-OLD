"""
Domain entities for the Congestion Prediction module.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Protocol, Any
from datetime import datetime
import torch

@dataclass
class NodeFeatures:
    """Features representing a node (intersection) in the traffic graph."""
    node_id: str
    speed: float = 0.0
    density: float = 0.0
    occupancy: float = 0.0
    vehicle_count: int = 0
    historical_avg: float = 0.0
    
    def to_tensor(self) -> torch.Tensor:
        return torch.tensor([
            self.speed, 
            self.density, 
            self.occupancy, 
            float(self.vehicle_count), 
            self.historical_avg
        ], dtype=torch.float32)

@dataclass
class EdgeFeatures:
    """Features representing an edge (road segment) in the traffic graph."""
    edge_id: str
    source_node_id: str
    target_node_id: str
    length: float = 1.0
    lanes: int = 1
    road_type: str = "urban"
    current_flow: int = 0
    max_speed: float = 60.0

@dataclass
class TrafficGraph:
    """
    Spatio-temporal graph representing traffic state at a specific timestamp.
    """
    timestamp: datetime
    nodes: List[NodeFeatures]
    edges: List[EdgeFeatures]
    adjacency_matrix: Optional[torch.Tensor] = None
    node_features_tensor: Optional[torch.Tensor] = None
    
    def __post_init__(self):
        if self.nodes and self.node_features_tensor is None:
            self.node_features_tensor = torch.stack([n.to_tensor() for n in self.nodes])

@dataclass
class CongestionPrediction:
    """
    Prediction result for future congestion levels.
    """
    intersection_id: str
    timestamp: datetime
    horizon_minutes: int
    predicted_speed: float
    predicted_density: float
    congestion_level: int # 0: Low, 1: Medium, 2: High
    confidence: float

# Interfaces for Data Sources

class VisionTrafficData(Protocol):
    """Interface for traffic data coming from the Computer Vision module."""
    def get_zone_metrics(self) -> List[Any]: # Todo: Define specific return type
        ...

class WazeTrafficData(Protocol):
    """Interface for traffic data coming from Waze/External sources."""
    def get_jams(self) -> List[Any]:
        ...
    def get_alerts(self) -> List[Any]:
        ...
