import yaml

from . import processing
from .load_data import load_dataset

CATALOG_FILE = "data_catalog.yaml"


class DataCatalog:
    def __init__(self, dataset):
        self.data = load_dataset(dataset)

        self.catalog = self._load_catalog(CATALOG_FILE, dataset)

        self._data_normalized = None

    @property
    def categoricals(self):
        return self.catalog["categorical"]

    @property
    def continous(self):
        return self.catalog["continous"]

    @property
    def immutables(self):
        return self.catalog["immutable"]

    @property
    def target(self):
        return self.catalog["target"]

    @property
    def normalized(self):
        if not self._data_normalized:
            self._data_normalized = processing.normalize(self.data)

        return self._data_normalized

    def _load_catalog(self, filename, dataset):
        with open(filename, "r") as f:
            catalog = yaml.safe_load(f)

        if dataset not in catalog:
            raise KeyError("Dataset not in catalog.")

        return catalog[dataset]
