import pandas as pd
from pathlib import Path
import json
import shutil


class DatasetLoader:
    def __init__(self, dataset_name: str, run_key: str, addTraces=False):
        self.dataset_name = dataset_name
        self.run_key = run_key
        self.working_directory = self._set_working_directory()
        self._data = None
        self.addTraces = addTraces

    def _set_working_directory(self):
        base = Path(__file__).parent.parent.parent
        return base / "_datasets" / self.run_key

    def _load_dataset_from_csv(self):
        filepath = self.working_directory / f"{self.dataset_name}.csv"
        self._data = pd.read_csv(filepath, sep=",")
        # Typen für den Merge vorbereiten
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

        # Merge-Typen sicherstellen
        raw_data["problem_id"] = raw_data["problem_id"].astype(str)
        raw_data["point_index"] = raw_data["point_index"].astype(int)

        # Join
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


data = DatasetLoader(
    run_key="run_20260419_110821", addTraces=True, dataset_name="dataset1"
)
print(data.data.head())
print(list(data.data.columns.values))