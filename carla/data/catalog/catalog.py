import pandas as pd
import yaml

from ..helper import processing
from .load_data import load_dataset
from ..api import Data


class DataCatalog(Data):
    def __init__(self, data_name, catalog_file):
        self.name = data_name
        self.catalog = self._load_catalog(catalog_file, data_name)

        self._raw = load_dataset(data_name)

        self._data_normalized = None
        self._data_encoded = None
        self._data_encoded_normalized = None

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
    def raw(self):
        return self._raw

    @property
    def normalized(self):
        if self._data_normalized is None:
            self._data_normalized = processing.normalize(self.raw, self.continous)

        return self._data_normalized

    @property
    def encoded(self):
        if self._data_encoded is None:
            self._data_encoded = pd.get_dummies(self.raw, drop_first=True)

        return self._data_encoded

    @property
    def encoded_normalized(self):
        if self._data_encoded_normalized is None:
            self._data_encoded_normalized = processing.normalize(
                self.encoded, self.continous
            )

        return self._data_encoded_normalized

    def _load_catalog(self, filename, dataset):
        with open(filename, "r") as f:
            catalog = yaml.safe_load(f)

        if dataset not in catalog:
            raise KeyError("Dataset not in catalog.")

        return catalog[dataset]
