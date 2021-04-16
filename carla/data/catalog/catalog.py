import pandas as pd
import yaml

from ..api import Data
from ..processing import processing
from .load_data import load_dataset


class DataCatalog(Data):
    def __init__(self, data_name, catalog_file, drop_first_encoding):
        """
        Constructor for catalog datasets.

        Parameters
        ----------
        data_name : String
            Used to get the correct dataset from online repository
        catalog_file : String
            yaml file
        drop_first_encoding : Bool
            Decides if the first column of one-hot-encoding should be dropped
        """
        self.name = data_name
        self.catalog = self._load_catalog(catalog_file, data_name)

        self._raw = load_dataset(data_name)

        self._drop_first = drop_first_encoding
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
        return self._raw.copy()

    @property
    def normalized(self):
        if self._data_normalized is None:
            self._data_normalized = processing.normalize(self.raw, self.continous)

        return self._data_normalized.copy()

    @property
    def encoded(self):
        if self._data_encoded is None:
            self._data_encoded = pd.get_dummies(self.raw, drop_first=self._drop_first)

        return self._data_encoded.copy()

    @property
    def encoded_normalized(self):
        if self._data_encoded_normalized is None:
            self._data_encoded_normalized = processing.normalize(
                self.encoded, self.continous
            )

        return self._data_encoded_normalized.copy()

    def _load_catalog(self, filename, dataset):
        with open(filename, "r") as f:
            catalog = yaml.safe_load(f)

        if dataset not in catalog:
            raise KeyError("Dataset not in catalog.")

        return catalog[dataset]
