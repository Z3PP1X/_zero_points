import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch_geometric.data import Batch, Data


class CustomGNNFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, gnn_model, features_dim=128):
        super().__init__(observation_space, features_dim)
        self.gnn = gnn_model

    def forward(self, observations):
        device = next(self.gnn.parameters()).device
        batch_size = observations["x"].shape[0]

        if batch_size == 1:
            num_nodes = int(observations["num_nodes"][0].item())
            num_edges = int(observations["num_edges"][0].item())
            x = observations["x"][0, :num_nodes, :].to(device)
            edge_index = observations["edge_index"][0, :, :num_edges].long().to(device)
            global_features = observations["global_features"][0, :].unsqueeze(0).to(device)
            batch_index = torch.zeros(x.size(0), dtype=torch.long, device=device)
            return self.gnn(x, edge_index, batch_index, global_features)

        data_list = []
        for index in range(batch_size):
            num_nodes = int(observations["num_nodes"][index].item())
            num_edges = int(observations["num_edges"][index].item())
            x = observations["x"][index, :num_nodes, :]
            edge_index = observations["edge_index"][index, :, :num_edges].long()
            global_features = observations["global_features"][index, :].unsqueeze(0)
            data_list.append(
                Data(x=x, edge_index=edge_index, global_features=global_features)
            )

        pyg_batch = Batch.from_data_list(data_list).to(device)
        return self.gnn(
            pyg_batch.x,
            pyg_batch.edge_index,
            pyg_batch.batch,
            pyg_batch.global_features,
        )
