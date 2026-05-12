import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch_geometric.data import Batch, Data


class CustomGNNFeaturesExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor for Stable Baselines 3 that handles padded graph data 
    from the Dict observation space, un-pads them, constructs a PyTorch Geometric Batch,
    and runs them through the TestGraphNetwork.
    """
    def __init__(self, observation_space, gnn_model, features_dim=128):
        # We assume the output of the GNN is features_dim
        super().__init__(observation_space, features_dim)
        self.gnn = gnn_model

    def forward(self, observations):
        # observations is a dict of batched tensors
        # SB3 automatically batches the inputs (batch_size, ...)
        device = next(self.gnn.parameters()).device
        batch_size = observations["x"].shape[0]

        # Single-env (batch_size == 1): skip PyG Batch construction overhead.
        if batch_size == 1:
            num_nodes = int(observations["num_nodes"][0].item())
            num_edges = int(observations["num_edges"][0].item())
            x = observations["x"][0, :num_nodes, :].to(device)
            edge_index = observations["edge_index"][0, :, :num_edges].long().to(device)
            global_features = observations["global_features"][0, :].unsqueeze(0).to(device)
            batch_index = torch.zeros(x.size(0), dtype=torch.long, device=device)
            return self.gnn(x, edge_index, batch_index, global_features)

        data_list = []
        for i in range(batch_size):
            # Extract the actual number of nodes and edges for this sample
            num_nodes = int(observations["num_nodes"][i].item())
            num_edges = int(observations["num_edges"][i].item())
            
            # Slice the valid parts of the arrays
            # We must use torch.clone or contiguous to avoid referencing the full padded tensor
            x = observations["x"][i, :num_nodes, :]
            # SB3 casts all Box observations to float32, but PyG needs long for edge_index
            edge_index = observations["edge_index"][i, :, :num_edges].long()
            global_features = observations["global_features"][i, :].unsqueeze(0)
            
            data = Data(x=x, edge_index=edge_index, global_features=global_features)
            data_list.append(data)
            
        pyg_batch = Batch.from_data_list(data_list)
        pyg_batch = pyg_batch.to(device)
        
        # Forward pass through the GNN
        shared_rep = self.gnn(
            pyg_batch.x, 
            pyg_batch.edge_index, 
            pyg_batch.batch, 
            pyg_batch.global_features
        )
        
        return shared_rep
