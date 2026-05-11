import torch
from sklearn.model_selection import train_test_split
from dataset import DatasetLoader
from feature_engineering import FeatureEngineering, GraphConversionPipeline
from torch.utils.data import Dataset


class GraphPipeline:
    def __init__(self, dataset_name: str, experiments_dir: str, seed: int = 42):
        self.seed = seed
        self.loader = DatasetLoader(dataset_name)
        fe = FeatureEngineering(self.loader)
        fe._tag_faster_algorithm()
        self.graph_pipeline = GraphConversionPipeline(experiments_dir)
        self.train_loader = None
        self.test_loader = None
        self.class_weights = None
        self.Y_train = None
        self.Y_test = None
        self.train_pids = None
        self.test_pids = None

    def pipe(
        self, test_size=0.2, batch_size=32, stratify: bool = True, num_workers: int = 0
    ):
        df = self.loader.data
        graph_ids = set(self.graph_pipeline.graphs.keys())
        df_matched = df[df["problem_id"].isin(graph_ids)].copy()

        unique_problem_ids = df_matched["problem_id"].unique()

        unique_problem_labels = [
            df_matched[df_matched["problem_id"] == p]["faster_algorithm"].iloc[0]
            for p in unique_problem_ids
        ]

        train_pids, test_pids = train_test_split(
            unique_problem_ids,
            test_size=test_size,
            random_state=self.seed,
            stratify=unique_problem_labels if stratify else None,
        )

        train_pids_set = set(train_pids)
        test_pids_set = set(test_pids)

        train_df = df_matched[df_matched["problem_id"].isin(train_pids_set)]
        test_df = df_matched[df_matched["problem_id"].isin(test_pids_set)]

        class_counts = train_df["faster_algorithm"].value_counts()
        total_train = len(train_df)
        num_classes = len(class_counts)

        weight_0 = total_train / (num_classes * class_counts.get(0, 1))
        weight_1 = total_train / (num_classes * class_counts.get(1, 1))
        self.class_weights = torch.tensor([weight_0, weight_1], dtype=torch.float)

        train_dataset = ProblemRunDataset(train_df, self.graph_pipeline.graphs)
        test_dataset = ProblemRunDataset(test_df, self.graph_pipeline.graphs)

        from torch_geometric.loader import DataLoader

        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=batch_size, num_workers=num_workers
        )

        print("-" * 40)
        print(
            f"IDs gesamt: {len(unique_problem_ids)} "
            f"(Train-IDs: {len(train_pids)}, Test-IDs: {len(test_pids)})"
        )
        print(
            f"Runs gesamt: {len(df_matched)} "
            f"(Train: {len(train_dataset)}, Test: {len(test_dataset)})"
        )
        print(
            f"Trainings-Verteilung: 0: {class_counts.get(0, 0)}, "
            f"1: {class_counts.get(1, 0)}"
        )
        print(f"Berechnete Gewichte: 0: {weight_0:.4f}, 1: {weight_1:.4f}")
        print("-" * 40)

        return self.train_loader, self.test_loader, self.class_weights

    @property
    def input_dim(self) -> int:
        return self.graph_pipeline.input_dim

    @property
    def global_dim(self) -> int:
        return 2


class ProblemRunDataset(Dataset):

    def __init__(self, df, base_graphs):
        self.df = df.reset_index(drop=True)
        self.base_graphs = base_graphs

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pid = row["problem_id"]

        data = self.base_graphs[pid].clone()

        data.y = torch.tensor([row["faster_algorithm"]], dtype=torch.long)
        data.global_features = torch.tensor(
            [row["startwert"], row["zielwert"]], dtype=torch.float
        )
        data.pid = pid

        return data
