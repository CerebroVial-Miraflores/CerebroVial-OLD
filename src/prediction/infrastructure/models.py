import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GraphConvolution(nn.Module):
    """
    Simple Graph Convolution Layer: H_out = ReLU(D^-1/2 * (A + I) * D^-1/2 * H_in * W)
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj):
        # x: (batch_size, num_nodes, in_features)
        # adj: (num_nodes, num_nodes)
        support = torch.matmul(x, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        return output

class STGCNModel(nn.Module):
    """
    Spatio-Temporal Graph Convolutional Network.
    Combines Graph Convolution (Spatial) and LSTM (Temporal).
    """
    def __init__(self, num_nodes, input_dim, hidden_dim, output_dim):
        super(STGCNModel, self).__init__()
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Spatial Block
        self.gc1 = GraphConvolution(input_dim, hidden_dim)
        self.gc2 = GraphConvolution(hidden_dim, hidden_dim)
        
        # Temporal Block (LSTM)
        # Input shape: (batch, time_steps, nodes * hidden_dim) -> handled by flattening nodes
        self.lstm = nn.LSTM(
            input_size=num_nodes * hidden_dim,
            hidden_size=num_nodes * hidden_dim,
            batch_first=True
        )
        
        # Output Layer
        self.output_layer = nn.Linear(num_nodes * hidden_dim, num_nodes * output_dim)

    def forward(self, x, adj):
        """
        x: (batch, time, nodes, features)
        adj: (nodes, nodes)
        """
        batch_size, time_steps, num_nodes, features = x.size()
        
        # Process each time step through GCN
        spatial_out = []
        for t in range(time_steps):
            xt = x[:, t, :, :] # (batch, nodes, features)
            h = F.relu(self.gc1(xt, adj))
            h = F.dropout(h, 0.5, training=self.training)
            h = F.relu(self.gc2(h, adj))
            spatial_out.append(h)
            
        # Stack back: (batch, time, nodes, hidden)
        spatial_seq = torch.stack(spatial_out, dim=1)
        
        # Flatten nodes for LSTM: (batch, time, nodes * hidden)
        lstm_in = spatial_seq.view(batch_size, time_steps, -1)
        
        # LSTM forward
        lstm_out, _ = self.lstm(lstm_in)
        
        # Take last time step
        last_out = lstm_out[:, -1, :] # (batch, nodes * hidden)
        
        # Output prediction
        prediction = self.output_layer(last_out) # (batch, nodes * output)
        
        # Reshape back to nodes
        return prediction.view(batch_size, num_nodes, -1)
