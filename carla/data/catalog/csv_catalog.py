from typing import List

import pandas as pd
from sklearn.model_selection import train_test_split

from .catalog import DataCatalog


class CsvCatalog(DataCatalog):
    def __init__(
        self,
        file_path: str,
        categorical: List[str],
        continuous: List[str],
        immutables: List[str],
        target: str,
        scaling_method: str = "MinMax",
        encoding_method: str = "OneHot_drop_binary",
    ):
        self._categorical = categorical
        self._continuous = continuous
        self._immutables = immutables
        self._target = target

        # Load the raw data
        raw = pd.read_csv(file_path)
        train_raw, test_raw = train_test_split(raw)

        super().__init__(
            "custom", raw, train_raw, test_raw, scaling_method, encoding_method
        )

    @property
    def categorical(self) -> List[str]:
        return self._categorical

    @property
    def continuous(self) -> List[str]:
        return self._continuous

    @property
    def immutables(self) -> List[str]:
        return self._immutables

    @property
    def target(self) -> str:
        return self._target
