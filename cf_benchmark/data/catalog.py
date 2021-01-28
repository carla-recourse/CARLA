import pandas as pd
import yaml

from . import processing
from .load_data import load_dataset

CATALOG_FILE = "data_catalog.yaml"


class DataCatalog:
    def __init__(self, dataset):
        self.data = load_dataset(dataset)

        self.catalog = self._load_catalog(CATALOG_FILE, dataset)

        self._data_normalized = None
        self._data_encoded = None

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
        if self._data_normalized is None:
            self._data_normalized = processing.normalize(self.data, self.continous)

        return self._data_normalized

    @property
    def encoded(self):
        if self._data_encoded is None:
            self._data_encoded = pd.get_dummies(self.data, drop_first=True)

        return self._data_encoded

    def _load_catalog(self, filename, dataset):
        with open(filename, "r") as f:
            catalog = yaml.safe_load(f)

        if dataset not in catalog:
            raise KeyError("Dataset not in catalog.")

        return catalog[dataset]
