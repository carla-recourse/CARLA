from typing import List

import pandas as pd
from sklearn.model_selection import train_test_split

from .catalog import DataCatalog


class CsvCatalog(DataCatalog):
    """
    Implements DataCatalog using local csv files. Using this class is the easiest way to use your own dataset.
    Besides data transformation, no other preprocessing is done. E.g. the user should remove NaNs.

    Parameters
    ----------
    file_path: str
        Path of the csv file.
    categorical: list[str]
        List containing the column names of the categorical features.
    continuous: list[str]
        List containing the column names of the continuous features.
    immutables: list[str]
        List containing the column names of the immutable features.
    target: str
        Column name of the target.

    Returns
    -------
    DataCatalog
    """

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
