from typing import Any, Callable, Dict, List

import pandas as pd
from sklearn.base import BaseEstimator

from carla.data.load_catalog import load_catalog

from ..api import Data
from .load_data import load_dataset


class DataCatalog(Data):
    """
    Use already implemented datasets.

    Parameters
    ----------
    data_name : {'adult', 'compas', 'give_me_some_credit'}
        Used to get the correct dataset from online repository.
    scaling_method: str, default: MinMax
        Type of used sklearn scaler. Can be set with property setter to any sklearn scaler.
    encoding_method: str, default: OneHot
        Type of OneHotEncoding [OneHot, OneHot_drop_binary]. Additional drop binary decides if one column
        is dropped for binary features. Can be set with property setter to any sklearn encoder.

    Returns
    -------
    None
    """

    def __init__(
        self,
        data_name: str,
        scaling_method: str = "MinMax",
        encoding_method: str = "OneHot",
    ):
        # TODO there is a very large overlap with the SCM dataclass. Probably should use a superclass for this.
        self.name = data_name

        catalog_content = ["continuous", "categorical", "immutable", "target"]
        self.catalog: Dict[str, Any] = load_catalog(  # type: ignore
            "data_catalog.yaml", data_name, catalog_content
        )

        for key in ["continuous", "categorical", "immutable"]:
            if self.catalog[key] is None:
                self.catalog[key] = []

        # Load the raw data
        raw, train_raw, test_raw = load_dataset(data_name)
        self._raw: pd.DataFrame = raw
        self._train_raw: pd.DataFrame = train_raw
        self._test_raw: pd.DataFrame = test_raw

        # Fit scaler and encoder
        self.scaler: BaseEstimator = self.__fit_scaler(scaling_method)
        self.encoder: BaseEstimator = self.__fit_encoder(encoding_method)

        # Preparing pipeline components
        self._pipeline = self.__init_pipeline()
        self._inverse_pipeline = self.__init_inverse_pipeline()

        # Process the data
        self._processed: pd.DataFrame = self.perform_pipeline(self.raw)
        self._train_processed: pd.DataFrame = self.perform_pipeline(self.train_raw)
        self._test_processed: pd.DataFrame = self.perform_pipeline(self.test_raw)

    @property
    def categorical(self) -> List[str]:
        return self.catalog["categorical"]

    @property
    def continuous(self) -> List[str]:
        return self.catalog["continuous"]

    @property
    def immutables(self) -> List[str]:
        return self.catalog["immutable"]

    @property
    def target(self) -> str:
        return self.catalog["target"]

    @property
    def raw(self) -> pd.DataFrame:
        return self._raw.copy()

    @property
    def train_raw(self) -> pd.DataFrame:
        return self._train_raw.copy()

    @property
    def test_raw(self) -> pd.DataFrame:
        return self._test_raw.copy()

    def processed(self, with_target=True) -> pd.DataFrame:
        df = self._processed.copy()
        if with_target:
            return df
        else:
            df = df[list(set(df.columns) - {self.target})]
            return df

    def train_processed(self, with_target=True) -> pd.DataFrame:
        df = self._train_processed.copy()
        if with_target:
            return df
        else:
            df = df[list(set(df.columns) - {self.target})]
            return df

    def test_processed(self, with_target=True) -> pd.DataFrame:
        df = self._test_processed.copy()
        if with_target:
            return df
        else:
            df = df[list(set(df.columns) - {self.target})]
            return df

    def get_pipeline_element(self, key: str) -> Callable:
        """
        Returns a specific element of the pipeline

        Parameters
        ----------
        key : str
            Element of the pipeline we want to return

        Returns
        -------
        Pipeline element
        """
        key_idx = list(zip(*self._pipeline))[0].index(key)  # find key in pipeline
        return self._pipeline[key_idx][1]

    @property
    def scaler(self) -> BaseEstimator:
        """
        Contains a fitted sklearn scaler.

        Returns
        -------
        sklearn.preprocessing.BaseEstimator
        """
        return self._scaler

    @scaler.setter
    def scaler(self, scaler: BaseEstimator):
        """
        Sets a new fitted sklearn scaler.

        Parameters
        ----------
        scaler : sklearn.preprocessing.Scaler
            Fitted scaler for ML model.

        Returns
        -------
        sklearn.preprocessing.BaseEstimator
        """
        self._scaler = scaler

    @property
    def encoder(self) -> BaseEstimator:
        """
        Contains a fitted sklearn encoder:

        Returns
        -------
        sklearn.preprocessing.BaseEstimator
        """
        return self._encoder

    @encoder.setter
    def encoder(self, encoder: BaseEstimator):
        """
        Sets a new fitted sklearn encoder.

        Parameters
        ----------
        encoder: sklearn.preprocessing.Encoder
            Fitted encoder for ML model.
        """
        self._encoder = encoder

    def perform_pipeline(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms input for prediction into correct form.
        Only possible for DataFrames without preprocessing steps.

        Recommended to use to keep correct encodings and normalization

        Parameters
        ----------
        df : pd.DataFrame
            Contains raw (unnormalized and not encoded) data.

        Returns
        -------
        output : pd.DataFrame
            Prediction input normalized and encoded

        """
        output = df.copy()

        for trans_name, trans_function in self._pipeline:
            output = trans_function(output)

        return output

    def perform_inverse_pipeline(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms output after prediction back into original form.
        Only possible for DataFrames with preprocessing steps.

        Parameters
        ----------
        df : pd.DataFrame
            Contains normalized and encoded data.

        Returns
        -------
        output : pd.DataFrame
            Prediction output denormalized and decoded

        """
        output = df.copy()

        for trans_name, trans_function in self._inverse_pipeline:
            output = trans_function(output)

        return output
