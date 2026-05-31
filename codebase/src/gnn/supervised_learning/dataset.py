import sys
import os
import pandas as pd
from pathlib import Path
import json
import shutil

# Dynamic sys.path resolution to support package imports when run as scripts
gnn_root = Path(__file__).resolve().parents[2]
if str(gnn_root) not in sys.path:
    sys.path.insert(0, str(gnn_root))
src_root = Path(__file__).resolve().parents[3]
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))


class DatasetLoader:
    def __init__(self, dataset_name: str, run_key: str = None, addTraces=False):
        # Auto-extract run_key if passed in dataset_name like "run_20260408_160456/dataset_4"
        if "/" in dataset_name and run_key is None:
            run_key, dataset_name = dataset_name.split("/", 1)
        
        self.dataset_name = dataset_name
        self.run_key = run_key
        self.working_directory = self._set_working_directory()
        self._data = None
        self.addTraces = addTraces

    def _set_working_directory(self):
        # Parents resolved:
        # [0] supervised_learning, [1] gnn, [2] src, [3] codebase, [4] _zero_points (repo root)
        base = Path(__file__).resolve().parents[4]
        return base / "_datasets" / self.run_key

    def _load_dataset_from_csv(self):
        filepath = self.working_directory / f"{self.dataset_name}.csv"
        self._data = pd.read_csv(filepath, sep=",")
        self._data["problem_id"] = self._data["problem_id"].astype(str)
        self._data["point_index"] = self._data["point_index"].astype(int)

    def _import_traces(self):
        traces_dir = self.working_directory / "traces"
        temp_dir = self.working_directory / "temp_parts"
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        temp_dir.mkdir(parents=True)

        batch = []
        batch_size = 5000
        part_num = 0

        columns = [
            "problem_id",
            "point_index",
            "y_target_json",
            "x0_json",
            "algorithm",
            "solver_converged",
            "solver_iterations",
            "solver_time_s",
            "iter",
            "x",
            "step_gain",
        ]

        for chunk_file in Path(traces_dir).glob("*.jsonl"):
            with open(chunk_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    pid = str(data.get("problemId"))
                    pix = int(data.get("pointIndex"))
                    yt = data.get("yTarget")
                    x0 = data.get("x0")

                    for solver in ["newton", "gmgf"]:
                        if solver not in data:
                            continue
                        s_obj = data[solver]

                        s_meta = [
                            solver,
                            s_obj.get("converged"),
                            s_obj.get("iterations"),
                            s_obj.get("time_s"),
                        ]

                        for t in s_obj.get("trace", []):
                            row = (
                                [pid, pix, yt, x0]
                                + s_meta
                                + [t.get("iter"), t.get("x"), t.get("step_gain")]
                            )
                            batch.append(row)

                            if len(batch) >= batch_size:
                                pd.DataFrame(batch, columns=columns).to_parquet(
                                    temp_dir / f"p_{part_num}.pq"
                                )
                                batch, part_num = [], part_num + 1

        if batch:
            pd.DataFrame(batch, columns=columns).to_parquet(
                temp_dir / f"p_{part_num}.pq"
            )

        return (
            pd.read_parquet(temp_dir)
            if any(temp_dir.iterdir())
            else pd.DataFrame(columns=columns)
        )

    def _join_traces(self):
        self._load_dataset_from_csv()
        raw_data = self._import_traces()

        raw_data["problem_id"] = raw_data["problem_id"].astype(str)
        raw_data["point_index"] = raw_data["point_index"].astype(int)

        self._data = pd.merge(
            self._data, raw_data, on=["problem_id", "point_index"], how="left"
        )

    @property
    def data(self):
        if self._data is None:
            if self.addTraces:
                self._join_traces()
            else:
                self._load_dataset_from_csv()
        return self._data

    def add_column(self, name: str, values):
        """Adds a new column to the loaded DataFrame."""
        self.data[name] = values


class DatasetDescriptor:
    def __init__(self, dataset_name, dataset: DatasetLoader = None):
        self.dataset_name = dataset_name
        self.dataset = dataset
        self.pandas_dataframe = None

    def _load_dataset(self):
        """Stellt sicher, dass das Dataframe geladen ist."""
        if self.pandas_dataframe is None:
            if self.dataset is None:
                self.dataset = DatasetLoader(self.dataset_name)
            self.pandas_dataframe = self.dataset.data

    def print_distribution(self):
        self._load_dataset()

        counts = self.pandas_dataframe["faster_algorithm"].value_counts()
        total = len(self.pandas_dataframe)

        newton_count = counts.get(0, 0)
        gmgf_count = counts.get(1, 0)

        perc_newton = (newton_count / total) * 100
        perc_gmgf = (gmgf_count / total) * 100

        print(f"--- Verteilung für Dataset: {self.dataset_name} ---")
        print(f"Gesamtanzahl Samples: {total}")
        print(f"Klasse 0 (Newton): {newton_count:>5} ({perc_newton:>5.2f}%)")
        print(f"Klasse 1 (gMGF):   {gmgf_count:>5} ({perc_gmgf:>5.2f}%)")
        print("-" * (30 + len(self.dataset_name)))


if __name__ == "__main__":
    # Test loading
    try:
        data_loader = DatasetLoader(
            run_key="run_20260419_110821", addTraces=True, dataset_name="dataset1"
        )
        print("DatasetLoader initialized successfully!")
    except Exception as e:
        print(f"DatasetLoader test failed (expected if datasets not present in this sandbox context): {e}")
