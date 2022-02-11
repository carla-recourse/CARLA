from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import pandas as pd

from carla.data.api import Data


class MLModel(ABC):
    """
    Abstract class to implement custom black-box-model for a given dataset with encoding and scaling processing.

    Parameters
    ----------
    data: Data
        Dataset inherited from Data-wrapper

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
    ) -> None:
        self.data: Data = data

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
