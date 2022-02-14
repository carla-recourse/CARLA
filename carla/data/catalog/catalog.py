from abc import ABC
from typing import Callable, List, Tuple

import pandas as pd
from sklearn import preprocessing
from sklearn.base import BaseEstimator

from carla.data.pipelining import decode, descale, encode, scale

from ..api import Data


class DataCatalog(Data, ABC):
    """
    Framework for datasets, using sklearn processing.

    Parameters
    ----------
    data_name: str
        What name the dataset should have.
    df: pd.DataFrame
        Complete dataframe.
    df_train: pd.DataFrame
        Training portion of the complete dataframe.
    df_test: pd.DataFrame
        Testing portion of the complete dataframe.
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
        df,
        df_train,
        df_test,
        scaling_method: str = "MinMax",
        encoding_method: str = "OneHot",
    ):
        self.name = data_name
        self._df = df
        self._df_train = df_train
        self._df_test = df_test

        # TODO rather then passing a string this could accept a function
        # Fit scaler and encoder
        self.scaler: BaseEstimator = self.__fit_scaler(scaling_method)
        self.encoder: BaseEstimator = self.__fit_encoder(encoding_method)

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

    # def processed(self, with_target=True) -> pd.DataFrame:
    #     df = self._processed.copy()
    #     if with_target:
    #         return df
    #     else:
    #         df = df[list(set(df.columns) - {self.target})]
    #         return df

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

    def __fit_scaler(self, scaling_method):
        # If needed another scaling method can be added here
        if scaling_method == "MinMax":
            fitted_scaler = preprocessing.MinMaxScaler().fit(self.df[self.continuous])
        elif scaling_method == "Standard":
            fitted_scaler = preprocessing.StandardScaler().fit(self.df[self.continuous])
        elif scaling_method is None or "Identity":
            fitted_scaler = preprocessing.FunctionTransformer(
                func=None, inverse_func=None
            )
        else:
            raise ValueError("Scaling Method not known")
        return fitted_scaler

    def __fit_encoder(self, encoding_method):
        if encoding_method == "OneHot":
            fitted_encoder = preprocessing.OneHotEncoder(
                handle_unknown="error", sparse=False
            ).fit(self.df[self.categorical])
        elif encoding_method == "OneHot_drop_binary":
            fitted_encoder = preprocessing.OneHotEncoder(
                drop="if_binary", handle_unknown="error", sparse=False
            ).fit(self.df[self.categorical])
        elif encoding_method is None or "Identity":
            fitted_encoder = preprocessing.FunctionTransformer(
                func=None, inverse_func=None
            )
        else:
            raise ValueError("Encoding Method not known")
        return fitted_encoder

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
