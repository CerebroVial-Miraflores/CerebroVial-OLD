import torch
import numpy as np
from typing import List, Tuple, Optional
from ..domain import NodeFeatures, EdgeFeatures, TrafficGraph
from datetime import datetime

class TrafficGraphBuilder:
    """
    Constructs the spatio-temporal graph from basic node and edge definitions.
    """
    def __init__(self, nodes_config: List[dict], edges_config: List[dict]):
        self.nodes_config = nodes_config
        self.edges_config = edges_config
        self._adj_matrix = self._build_adjacency_matrix()
    
    def _build_adjacency_matrix(self) -> torch.Tensor:
        """
        Builds the normalized adjacency matrix (D^-1/2 * (A + I) * D^-1/2).
        """
        num_nodes = len(self.nodes_config)
        adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        
        # Create mapping from node_id to index
        node_to_idx = {node['id']: idx for idx, node in enumerate(self.nodes_config)}
        
        # Fill adjacency matrix (directed)
        for edge in self.edges_config:
            src = edge['source']
            dst = edge['target']
            if src in node_to_idx and dst in node_to_idx:
                src_idx = node_to_idx[src]
                dst_idx = node_to_idx[dst]
                adj[src_idx, dst_idx] = 1.0
                
        # Self-loops
        adj = adj + np.eye(num_nodes)
        
        # Normalization
        d = np.sum(adj, axis=1)
        d_inv_sqrt = np.power(d, -0.5)
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = np.diag(d_inv_sqrt)
        
        norm_adj = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
        return torch.FloatTensor(norm_adj)

    def build_graph(self, node_data: List[NodeFeatures], timestamp: datetime) -> TrafficGraph:
        """
        Constructs a TrafficGraph from current realtime node features.
        """
        # Ensure node_data order matches config order (or sort/map them)
        # For simplicity, assuming node_data handles all nodes in config order or we map them here
        
        # Create domain edges from config
        edges = [
            EdgeFeatures(
                edge_id=f"{e['source']}_{e['target']}",
                source_node_id=e['source'],
                target_node_id=e['target']
            ) for e in self.edges_config
        ]
        
        return TrafficGraph(
            timestamp=timestamp,
            nodes=node_data,
            edges=edges,
            adjacency_matrix=self._adj_matrix
        )
