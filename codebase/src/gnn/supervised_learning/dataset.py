import sys

import pandas as pd
from pathlib import Path
import json
import shutil


gnn_root = Path(__file__).resolve().parents[2]
if str(gnn_root) not in sys.path:
    sys.path.insert(0, str(gnn_root))
src_root = Path(__file__).resolve().parents[3]
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))


class DatasetLoader:
    def __init__(
        self, dataset_name: str, run_key: str = None, addTraces=False, base_dir=None
    ):
        if "/" in dataset_name and run_key is None:
            run_key, dataset_name = dataset_name.split("/", 1)

        self.dataset_name = dataset_name
        self.run_key = run_key
        self.base_dir = base_dir
        self.working_directory = self._set_working_directory()
        self._data = None
        self.addTraces = addTraces

    def _set_working_directory(self):
        if self.base_dir is not None:
            p = Path(self.base_dir)
            return p if p.is_dir() else p.parent

        base = Path(__file__).resolve().parents[4]
        primary = base / "datasets" / self.run_key
        if primary.exists():
            return primary
        return base / "_datasets" / self.run_key

    def _normalize_headers(self):
        if self._data is None:
            return

        mapping = {
            "problem_id": "problem_id",
            "problemid": "problem_id",
            "y_target": "y_target",
            "ytarget": "y_target",
            "zielwert": "y_target",
            "startwert": "x0",
            "x0": "x0",
            "newton_abstime": "Newton_absTime",
            "newtonabstime": "Newton_absTime",
            "avg_abs_time_newton": "Newton_absTime",
            "gmgf_abstime": "GMGF_absTime",
            "gmgfabstime": "GMGF_absTime",
            "avg_abs_time_gmgf": "GMGF_absTime",
            "newton_itersteps": "Newton_iterSteps",
            "newtonitersteps": "Newton_iterSteps",
            "schritte_newton": "Newton_iterSteps",
            "gmgf_itersteps": "GMGF_iterSteps",
            "gmgfitersteps": "GMGF_iterSteps",
            "schritte_gmgf": "GMGF_iterSteps",
            "point_index": "point_index",
            "pointindex": "point_index",
            # Function value f(x0) at the current iterate.
            "fx": "fx",
            "fx0": "fx",
            "f_x0": "fx",
            "f_x_0": "fx",
            "fval": "fx",
            "fvalue": "fx",
            # First derivative f'(x0) — Newton step driver.
            "d1x": "d1x",
            "f1": "d1x",
            "fprime": "d1x",
            "fprime_x0": "d1x",
            "dfx": "d1x",
            "derivative1": "d1x",
            "first_derivative": "d1x",
            # Second derivative f''(x0) — curvature / Halley-style driver.
            "d2x": "d2x",
            "f2": "d2x",
            "fsecond": "d2x",
            "fdoubleprime": "d2x",
            "d2fx": "d2x",
            "derivative2": "d2x",
            "second_derivative": "d2x",
        }

        rename_dict = {}
        for col in self._data.columns:
            clean_col = col.strip().replace('"', "").replace("'", "").lower()
            if clean_col in mapping:
                rename_dict[col] = mapping[clean_col]

        if rename_dict:
            self._data = self._data.rename(columns=rename_dict)

    def _load_dataset_from_csv(self):
        filepath = self.working_directory / f"{self.dataset_name}.csv"
        self._data = pd.read_csv(filepath, sep=",")
        self._normalize_headers()
        if "problem_id" in self._data.columns:
            self._data["problem_id"] = self._data["problem_id"].astype(str)
        if "point_index" in self._data.columns:
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


if __name__ == "__main__":
    try:
        data_loader = DatasetLoader(
            run_key="run_20260419_110821", addTraces=True, dataset_name="dataset1"
        )
        print("DatasetLoader initialized successfully!")
    except Exception as e:
        print(
            "DatasetLoader test failed (expected if datasets not present in this "
            f" sandbox context): {e}"
        )
