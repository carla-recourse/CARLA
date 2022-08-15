from abc import ABC
from typing import Callable, List, Tuple

import pandas as pd
from sklearn.base import BaseEstimator

from carla.data.pipelining import (
    decode,
    descale,
    encode,
    fit_encoder,
    fit_scaler,
    scale,
)

from ..api import Data


class DataCatalog(Data, ABC):
    """
    Generic framework for datasets, using sklearn processing. This class is implemented by OnlineCatalog and CsvCatalog.
    OnlineCatalog allows the user to easily load online datasets, while CsvCatalog allows easy use of local datasets.

    Parameters
    ----------
    data_name: str
        What name the dataset should have.
    df: pd.DataFrame
        The complete Dataframe. This is equivalent to the combination of df_train and df_test, although not shuffled.
    df_train: pd.DataFrame
        Training portion of the complete Dataframe.
    df_test: pd.DataFrame
        Testing portion of the complete Dataframe.
    scaling_method: str, default: MinMax
        Type of used sklearn scaler. Can be set with the property setter to any sklearn scaler.
        Set to "Identity" for no scaling.
    encoding_method: str, default: OneHot_drop_binary
        Type of OneHotEncoding {OneHot, OneHot_drop_binary}. Additional drop binary decides if one column
        is dropped for binary features. Can be set with the property setter to any sklearn encoder.
        Set to "Identity" for no encoding.

    Returns
    -------
    Data
    """

    def __init__(
        self,
        data_name: str,
        df,
        df_train,
        df_test,
        scaling_method: str = "MinMax",
        encoding_method: str = "OneHot_drop_binary",
    ):
        self.name = data_name
        self._df = df
        self._df_train = df_train
        self._df_test = df_test

        # Fit scaler and encoder
        self.scaler: BaseEstimator = fit_scaler(
            scaling_method, self.df[self.continuous]
        )
        self.encoder: BaseEstimator = fit_encoder(
            encoding_method, self.df[self.categorical]
        )
        self._identity_encoding = (
            encoding_method is None or encoding_method == "Identity"
        )

        # Preparing pipeline components
        self._pipeline = self.__init_pipeline()
        self._inverse_pipeline = self.__init_inverse_pipeline()

        # Process the data
        self._df = self.transform(self.df)
        self._df_train = self.transform(self.df_train)
        self._df_test = self.transform(self.df_test)

    @property
    def df(self) -> pd.DataFrame:
        return self._df.copy()

    @property
    def df_train(self) -> pd.DataFrame:
        return self._df_train.copy()

    @property
    def df_test(self) -> pd.DataFrame:
        return self._df_test.copy()

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

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms input for prediction into correct form.
        Only possible for DataFrames without preprocessing steps.

        Recommended to keep correct encodings and normalization

        Parameters
        ----------
        df : pd.DataFrame
            Contains raw (not normalized and not encoded) data.

        Returns
        -------
        output : pd.DataFrame
            Prediction input normalized and encoded

        """
        output = df.copy()

        for trans_name, trans_function in self._pipeline:
            if trans_name == "encoder" and self._identity_encoding:
                continue
            else:
                output = trans_function(output)

        return output

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
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

    def get_pipeline_element(self, key: str) -> Callable:
        """
        Returns a specific element of the transformation pipeline.

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

    def __init_pipeline(self) -> List[Tuple[str, Callable]]:
        return [
            ("scaler", lambda x: scale(self.scaler, self.continuous, x)),
            ("encoder", lambda x: encode(self.encoder, self.categorical, x)),
        ]

    def __init_inverse_pipeline(self) -> List[Tuple[str, Callable]]:
        return [
            ("encoder", lambda x: decode(self.encoder, self.categorical, x)),
            ("scaler", lambda x: descale(self.scaler, self.continuous, x)),
        ]
