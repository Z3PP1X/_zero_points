import pandas as pd
from pathlib import Path


class DatasetLoader:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self._data = None

    def _get_dataset_filepath(self):
        base = Path(__file__).parent.parent.parent
        filepath = base / "_datasets" / f"{self.dataset_name}.csv"
        if filepath.exists():
            return filepath
        else:
            raise FileNotFoundError(
                f"File {self.dataset_name} does not exist in {filepath}"
            )

    # FIX:
    def _load_dataset_from_csv(self):
        self._data = pd.read_csv(
            filepath_or_buffer=self._get_dataset_filepath(), sep=","
        )

    @property
    def data(self):
        """ "Lazy-loads the dataframe upon first access."""
        if self._data is None:
            self._load_dataset_from_csv()
        return self._data

    @property
    def newton(self):
        self.data
        if self._data is not None:
            return self._data[
                [
                    "problem_id",
                    "schritte_newton",
                    "loesung_newton",
                ]
            ]

    @property
    def gMGF(self):
        self.data
        if self._data is not None:
            return self._data[["problem_id", "schritte_gmgf", "loesung_gmgf"]]

    @property
    def problem_config(self):
        self.data
        if self._data is not None:
            return self._data[
                [
                    "problem_id",
                    "problem",
                    "startwert",
                    "zielwert",
                    "conserved_step_rel",
                ]
            ]

    @property
    def history(self):
        self.data
        if self._data is not None:
            return self._data[
                [
                    "problem_id",
                    "newton_abs_err_hist",
                    "newton_rel_err_hist",
                    "newton_diag_status",
                    "gmgf_abs_err_hist",
                    "gmgf_rel_err_hist",
                    "gmgf_kappa_raw_hist",
                    "gmgf_kappa_clamp_hist",
                    "gmgf_diag_status",
                ]
            ]

    def add_column(self, name: str, values):
        self.data[name] = values

    def get_view(self, view: str):
        allowed = {"data", "newton", "gmgf", "config"}
        if view not in allowed:
            raise ValueError(f"'{view}' ist ungültig. Erlaubt: {allowed}")
        return getattr(self, view)


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

            # Wir nutzen die .data Property des Loaders
            self.pandas_dataframe = self.dataset.data

    def print_distribution(self):
        """Gibt die Verteilung der Zielklasse (faster_algorithm) aus."""
        self._load_dataset()

        if "faster_algorithm" not in self.pandas_dataframe.columns:
            print(
                "⚠️ Spalte 'faster_algorithm' nicht gefunden. "
                "Führe zuerst FeatureEngineering._tag_faster_algorithm() aus."
            )
            return

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
