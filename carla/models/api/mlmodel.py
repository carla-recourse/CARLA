from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.base import BaseEstimator

from carla.data.api import Data


class MLModel(ABC):
    def __init__(
        self,
        data: Data,
        scaling_method: str = "MinMax",
        encoding_method: str = "OneHot",
    ) -> None:
        self.data: Data = data

        if scaling_method == "MinMax":
            fitted_scaler = preprocessing.MinMaxScaler().fit(data.raw[data.continous])
            self.scaler: BaseEstimator = fitted_scaler

        if encoding_method == "OneHot":
            fitted_encoder = preprocessing.OneHotEncoder(
                handle_unknown="ignore", sparse=False
            ).fit(data.raw[data.categoricals])
            self.encoder: BaseEstimator = fitted_encoder

    @property
    def data(self) -> Data:
        return self._data

    @data.setter
    def data(self, data: Data):
        self._data = data

    @property
    def scaler(self) -> BaseEstimator:
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

        """
        self._scaler = scaler

    @property
    def encoder(self) -> BaseEstimator:
        return self._encoder

    @encoder.setter
    def encoder(self, encoder: BaseEstimator):
        """
        Sets a new fitted sklearn encoder.

        Parameters
        ----------
        encoder : sklearn.preprocessing.Encoder
            Fitted encoder for ML model.

        Returns
        -------

        """
        self._encoder = encoder

    @property
    @abstractmethod
    def feature_input_order(self):
        """
        Saves the required order of feature as list.

        Prevents confusion about correct order of input features in evaluation

        Returns
        -------
        ordered_features : List of String
            Correct order of input features for ml model
        """
        pass

    @property
    @abstractmethod
    def backend(self):
        """
        Describes the type of backend which is used for the ml model.

        E.g., tensorflow, pytorch, sklearn, ...

        Returns
        -------
        backend : String
            Used framework
        """
        pass

    @property
    @abstractmethod
    def raw_model(self):
        """
        Returns the raw ml model built on its framework

        Returns
        -------
        ml_model : tensorflow, pytorch, sklearn model type
            Loaded model
        """
        pass

    @abstractmethod
    def predict(self, x: Union[np.ndarray, pd.DataFrame]):
        """
        One-dimensional prediction of ml model for an output interval of [0, 1].

        Shape of input dimension has to be always two-dimensional (e.g., (1, m), (n, m))

        Parameters
        ----------
        x : np.Array or pd.DataFrame
            Tabular data of shape N x M (N number of instances, M number of features)

        Returns
        -------
        output : np.Array
            Ml model prediction for interval [0, 1] with shape N x 1
        """
        pass

    @abstractmethod
    def predict_proba(self, x: Union[np.ndarray, pd.DataFrame]):
        """
        Two-dimensional probability prediction of ml model

        Shape of input dimension has to be always two-dimensional (e.g., (1, m), (n, m))

        Parameters
        ----------
        x : np.Array or pd.DataFrame
            Tabular data of shape N x M (N number of instances, M number of features)

        Returns
        -------
        output : float
            Ml model prediction with shape N x 2
        """
        pass
