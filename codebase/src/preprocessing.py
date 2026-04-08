import torch
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from dataset import DatasetLoader
from feature_engineering import FeatureEngineering, GraphConversionPipeline


class GraphPipeline:
    def __init__(self, dataset_name: str, experiments_dir: str, seed: int = 42):
        self.seed = seed

        # Tabulare Daten + Feature Engineering
        self.loader = DatasetLoader(dataset_name)
        fe = FeatureEngineering(self.loader)
        fe._tag_faster_algorithm()

        # Graphen konvertieren
        self.graph_pipeline = GraphConversionPipeline(experiments_dir)

        # Werden in pipe() gesetzt
        self.train_loader = None
        self.test_loader = None
        self.Y_train = None
        self.Y_test = None
        self.train_pids = None
        self.test_pids = None

    def pipe(self, test_size=0.2, batch_size=32, stratify: bool = True):
        df = self.loader.data
        graph_ids = set(self.graph_pipeline.graphs.keys())
        df_matched = df[df["problem_id"].isin(graph_ids)].copy()

        unique_pids = df_matched["problem_id"].unique()

        pid_labels = [
            df_matched[df_matched["problem_id"] == p]["faster_algorithm"].iloc[0]
            for p in unique_pids
        ]

        train_pids, test_pids = train_test_split(
            unique_pids,
            test_size=test_size,
            random_state=self.seed,
            stratify=pid_labels if stratify else None,
        )

        train_pids_set = set(train_pids)

        train_graphs = []
        test_graphs = []

        # 3. Schritt: Alle 40.0000 Zeilen durchgehen und zuordnen
        for _, row in df_matched.iterrows():
            pid = row["problem_id"]

            # Erstelle ein neues Datenobjekt für JEDE Zeile
            data = self.graph_pipeline.graphs[pid].clone()
            data.y = torch.tensor([row["faster_algorithm"]], dtype=torch.long)
            data.global_features = torch.tensor(
                [row["startwert"], row["zielwert"]], dtype=torch.float
            )
            data.pid = pid

            if pid in train_pids_set:
                train_graphs.append(data)
            else:
                test_graphs.append(data)

        self.train_loader = DataLoader(
            train_graphs, batch_size=batch_size, shuffle=True
        )
        self.test_loader = DataLoader(test_graphs, batch_size=batch_size)

        print(
            f"IDs gesamt: {len(unique_pids)} "
            f"(Train-IDs: {len(train_pids)}, Test-IDs: {len(test_pids)})"
        )
        print(
            f"Graphen-Objekte gesamt: {len(train_graphs) + len(test_graphs)} "
            f"(Train: {len(train_graphs)}, Test: {len(test_graphs)})"
        )

        return self.train_loader, self.test_loader

    @property
    def input_dim(self) -> int:
        return self.graph_pipeline.input_dim

    @property
    def global_dim(self) -> int:
        return 2
