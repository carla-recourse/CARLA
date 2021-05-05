from typing import Any, Dict, List

import pandas as pd
import yaml

from ..api import Data
from .load_data import load_dataset


class DataCatalog(Data):
    def __init__(self, data_name: str, catalog_file: str):
        """
        Constructor for catalog datasets.

        Parameters
        ----------
        data_name : String
            Used to get the correct dataset from online repository
        catalog_file : String
            yaml file
        """
        self.name: str = data_name
        self.catalog: Dict[str, Any] = self._load_catalog(catalog_file, data_name)

        self._raw: pd.DataFrame = load_dataset(data_name)

    @property
    def categoricals(self) -> List[str]:
        return self.catalog["categorical"]

    @property
    def continous(self) -> List[str]:
        return self.catalog["continous"]

    @property
    def immutables(self) -> List[str]:
        return self.catalog["immutable"]

    @property
    def target(self) -> str:
        return self.catalog["target"]

    @property
    def raw(self) -> pd.DataFrame:
        return self._raw.copy()

    def _load_catalog(self, filename: str, dataset: str):
        with open(filename, "r") as f:
            catalog = yaml.safe_load(f)

        if dataset not in catalog:
            raise KeyError("Dataset not in catalog.")

        return catalog[dataset]
