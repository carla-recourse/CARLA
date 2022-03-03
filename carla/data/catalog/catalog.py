from abc import ABC

import pandas as pd

from ..api import Data


class DataCatalog(Data, ABC):
    """
    Framework for datasets, using sklearn processing.

    Parameters
    ----------
    data_name: str
        What name the dataset should have.
    raw: pd.DataFrame
        Complete dataframe.
    train_raw: pd.DataFrame
        Training portion of the complete dataframe.
    test_raw: pd.DataFrame
        Testing portion of the complete dataframe.

    Returns
    -------
    None
    """

    def __init__(
        self,
        data_name: str,
        raw,
        train_raw,
        test_raw,
    ):
        self.name = data_name
        self._raw: pd.DataFrame = raw
        self._train_raw: pd.DataFrame = train_raw
        self._test_raw: pd.DataFrame = test_raw

    @property
    def raw(self) -> pd.DataFrame:
        return self._raw.copy()

    @property
    def train_raw(self) -> pd.DataFrame:
        return self._train_raw.copy()

    @property
    def test_raw(self) -> pd.DataFrame:
        return self._test_raw.copy()
