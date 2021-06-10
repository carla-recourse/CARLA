from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.base import BaseEstimator

from carla.data.api import Data


class MLModel(ABC):
    """
    Abstract class to implement custom black-box-model for a given dataset with encoding and scaling processing.

    Parameters
    ----------
    data: Data
        Dataset inherited from Data-wrapper
    scaling_method: str, default: MinMax
        Type of used sklearn scaler. Can be set with property setter to any sklearn scaler.
    encoding_method: str, default: OneHot
        Type of OneHotEncoding [OneHot, OneHot_drop_binary]. Additional drop binary decides if one column
        is dropped for binary features. Can be set with property setter to any sklearn encoder.

    Methods
    -------
    predict:
        One-dimensional prediction of ml model for an output interval of [0, 1].
    predict_proba:
        Two-dimensional probability prediction of ml model

    Returns
    -------
    None
    """

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
                handle_unknown="error", sparse=False
            ).fit(data.raw[data.categoricals])
        elif encoding_method == "OneHot_drop_binary":
            fitted_encoder = preprocessing.OneHotEncoder(
                drop="if_binary", handle_unknown="error", sparse=False
            ).fit(data.raw[data.categoricals])
        else:
            raise ValueError("Encoding Method not known")

        self.encoder: BaseEstimator = fitted_encoder

    @property
    def data(self) -> Data:
        """
        Contains the data.api.Data dataset.

        Returns
        -------
        carla.data.Data
        """
        return self._data

    @data.setter
    def data(self, data: Data) -> None:
        self._data = data

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

    @property
    @abstractmethod
    def feature_input_order(self):
        """
        Saves the required order of feature as list.

        Prevents confusion about correct order of input features in evaluation

        Returns
        -------
        list of str
        """
        pass

    @property
    @abstractmethod
    def backend(self):
        """
        Describes the type of backend which is used for the classifier.

        E.g., tensorflow, pytorch, sklearn, ...

        Returns
        -------
        str
        """
        pass

    @property
    @abstractmethod
    def raw_model(self):
        """
        Contains the raw ml model built on its framework

        Returns
        -------
        object
            Classifier, depending on used framework
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
        iterable object
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
        iterable object
            Ml model prediction with shape N x 2
        """
        pass
