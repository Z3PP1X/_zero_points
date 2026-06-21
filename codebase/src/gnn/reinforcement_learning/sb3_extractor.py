import logging

import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch_geometric.data import Batch, Data

from observation_sanitize import sanitize_torch_features

logger = logging.getLogger(__name__)


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
            batch_index = torch.zeros(x.size(0), dtype=torch.long, device=device)
            edge_attr = None
            if "edge_attr" in observations:
                edge_attr = observations["edge_attr"][0, :num_edges, :].to(device)
            features = self.gnn(x, edge_index, batch_index, edge_attr=edge_attr)
            return self._sanitize_features(features)

        data_list = []
        for index in range(batch_size):
            num_nodes = int(observations["num_nodes"][index].item())
            num_edges = int(observations["num_edges"][index].item())
            x = observations["x"][index, :num_nodes, :]
            edge_index = observations["edge_index"][index, :, :num_edges].long()
            data_kwargs = {"x": x, "edge_index": edge_index}
            if "edge_attr" in observations:
                data_kwargs["edge_attr"] = observations["edge_attr"][index, :num_edges, :]
            data_list.append(Data(**data_kwargs))

        pyg_batch = Batch.from_data_list(data_list).to(device)
        features = self.gnn(
            pyg_batch.x,
            pyg_batch.edge_index,
            pyg_batch.batch,
            edge_attr=getattr(pyg_batch, "edge_attr", None),
        )
        return self._sanitize_features(features)

    def _sanitize_features(self, features: torch.Tensor) -> torch.Tensor:
        if torch.isfinite(features).all():
            return features
        bad_rows = (~torch.isfinite(features)).any(dim=1).nonzero(as_tuple=False).flatten()
        logger.warning(
            "Non-finite GNN features for batch indices %s; replacing with zeros.",
            bad_rows.tolist(),
        )
        return sanitize_torch_features(features)
