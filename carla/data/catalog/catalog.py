from carla.data.load_catalog import load_catalog

from ..api import Data
from .load_data import load_dataset


class DataCatalog(Data):
    def __init__(self, data_name, catalog_file):
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
        self.catalog_file = catalog_file
        self.catalog = load_catalog(self.catalog_file, data_name)

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
        return self._raw.copy()
