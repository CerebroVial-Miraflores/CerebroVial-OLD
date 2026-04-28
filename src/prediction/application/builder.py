from typing import List, Any
from ..infrastructure.graph_builder import TrafficGraphBuilder
from ..infrastructure.data_loader import TrafficDataLoader
from ..infrastructure.models import STGCNModel
from ..infrastructure.repository import PredictionRepository
from .predictor import CongestionPredictor

class PredictionApplicationBuilder:
    """
    Builder for wiring up the Prediction Application components.
    """
    def __init__(self):
        self._nodes_config: List[dict] = []
        self._edges_config: List[dict] = []
        self._vision_source = None
        self._waze_source = None
        
    def with_graph_topology(self, nodes: List[dict], edges: List[dict]):
        self._nodes_config = nodes
        self._edges_config = edges
        return self
        
    def with_vision_source(self, source: Any):
        self._vision_source = source
        return self
        
    def with_waze_source(self, source: Any):
        self._waze_source = source
        return self
        
    def build(self) -> CongestionPredictor:
        if not self._nodes_config or not self._edges_config:
            raise ValueError("Graph topology not configured")
            
        # 1. Infrastructure
        graph_builder = TrafficGraphBuilder(self._nodes_config, self._edges_config)
        data_loader = TrafficDataLoader(self._vision_source, self._waze_source)
        repository = PredictionRepository()
        
        # 2. Model
        num_nodes = len(self._nodes_config)
        # Using hardcoded dims for MVP
        model = STGCNModel(
            num_nodes=num_nodes,
            input_dim=5, # Features in NodeFeatures
            hidden_dim=32,
            output_dim=2 # Speed, Density
        )
        
        # 3. Application Service
        predictor = CongestionPredictor(
            graph_builder,
            data_loader,
            model,
            repository
        )
        
        return predictor
