from torch_geometric.nn import MessagePassing
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

# Approach 1: Learnable Transformation for Edge Weights
class EdgeWeightedGNN(MessagePassing):
    def __init__(self, in_channels, op_channels, out_channels, num_layers=2, hidden_dim=64, dropout=0.2):
        super(EdgeWeightedGNN, self).__init__(aggr="add")  # Aggregation type
        self.num_layers = num_layers
        self.node_mlp = torch.nn.ModuleList(
            [torch.nn.Linear(in_channels if i == 0 else hidden_dim, hidden_dim) for i in range(num_layers)]
        )
        self.edge_mlp = torch.nn.ModuleList(
            [torch.nn.Linear(1, hidden_dim) for _ in range(num_layers)]  # One for each layer
        )
        self.projection = torch.nn.Linear(hidden_dim, in_channels)  # Project back to initial embedding size
        self.fc = torch.nn.Linear(in_channels + op_channels, out_channels)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, batch, return_embeddings=False):
        x = batch.x
        edge_index = batch.edge_index
        edge_attr = batch.edge_attr

        for i in range(self.num_layers):
            # Transform node features
            x = self.node_mlp[i](x)
            x = F.relu(x)
            x = self.dropout(x)

            # Transform edge attributes
            edge_weight = self.edge_mlp[i](edge_attr)

            # Message passing
            x = self.propagate(edge_index, x=x, edge_weight=edge_weight)

        # Project embeddings back to initial embedding size
        final_embeddings = self.projection(x)

        if return_embeddings:
            return final_embeddings  # Return node embeddings only

        # Concatenate OP embedding before the linear layer
        x = torch.cat([final_embeddings, batch.op], dim=1)
        x = self.fc(x)
        return x

    def message(self, x_j, edge_weight):
        return x_j * edge_weight  # Scale source node features by edge weight

# Approach 2: Predefined Distance Decay for Edge Weights
class DistanceWeightedGNN(torch.nn.Module):
    def __init__(self, in_channels, op_channels, out_channels, dropout=0.2):
        super(DistanceWeightedGNN, self).__init__()
        self.conv1 = GCNConv(in_channels, 64)
        self.conv2 = GCNConv(64, 64)
        self.projection = torch.nn.Linear(64, in_channels)  # Project back to initial embedding size
        self.fc = torch.nn.Linear(in_channels + op_channels, out_channels)  # Adjust size
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, batch, return_embeddings=False):
        # Compute edge weights using distance
        edge_weight = 1 / (batch.edge_attr + 1)  # Distance decay function

        # Apply GNN layers
        x = self.conv1(batch.x, batch.edge_index, edge_weight=edge_weight.squeeze())
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, batch.edge_index, edge_weight=edge_weight.squeeze())
        x = F.relu(x)
        x = self.dropout(x)

        # Project embeddings back to initial embedding size
        final_embeddings = self.projection(x)

        if return_embeddings:
            return final_embeddings  # Return node embeddings only

        # Concatenate OP embedding before the linear layer
        x = torch.cat([final_embeddings, batch.op], dim=1)
        x = self.fc(x)
        return x
    
# Approach 3: Baseline Node Classifier
class BaselineNodeClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super(BaselineNodeClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)  # For classification

    def forward(self, data):
        # Concatenate the initial node embedding (x) and the op embedding
        inputs = torch.cat((data.x, data.op), dim=1)
        x = self.fc1(inputs)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return self.softmax(x)

class GraphPerBatchDataset(torch.utils.data.Dataset):
    def __init__(self, graphs, node_embeddings, op_embeddings, labels, global_node_to_idx):
        """
        Dataset where each batch corresponds to a single graph.
        """
        self.graphs = graphs
        self.node_embeddings = node_embeddings
        self.op_embeddings = op_embeddings
        self.labels = labels
        self.global_node_to_idx = global_node_to_idx

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        graph = self.graphs[idx]

        # Extract nodes and edges from the graph
        graph_nodes = list(graph.nodes)

        # Map global indices to local indices
        local_node_to_idx = {node: i for i, node in enumerate(graph_nodes)}
        batch_indices = [self.global_node_to_idx[node] for node in graph_nodes]

        # Map edges from global to local indices
        edge_index = torch.tensor(
            [
                [local_node_to_idx[src], local_node_to_idx[dst]]
                for src, dst in graph.edges
            ],
            dtype=torch.long,
        ).t().contiguous()

        # Extract edge attributes (e.g., distance)
        edge_attr = torch.tensor(
            [graph[src][dst]["distance"] for src, dst in graph.edges],
            dtype=torch.float,
        ).unsqueeze(1)  # Shape: [num_edges, 1]

        node_types = torch.tensor(
            [graph.nodes[node]["node_type"] for node in graph_nodes],
            dtype=torch.long,
        )  # Assign 1 for "predict" and 0 for "author" for op nodes

        # Prepare features and labels
        embeddings = self.node_embeddings[batch_indices]
        op_feats = self.op_embeddings[batch_indices]
        graph_labels = torch.tensor([self.labels[node] for node in graph_nodes], dtype=torch.long)

        # Create a `Data` object for the graph
        return Data(x=embeddings, edge_index=edge_index, edge_attr=edge_attr, op=op_feats, node_type=node_types, y=graph_labels)