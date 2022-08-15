import warnings
from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import pandas as pd

from carla.data.api import Data
from carla.data.pipelining import order_data


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
        Two-dimensional probability prediction of ml model.

    Returns
    -------
    None
    """

    def __init__(
        self,
        data: Data,
    ) -> None:
        self._data: Data = data

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
        Saves the required order of features as list.

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

        E.g., tensorflow, pytorch, sklearn, xgboost

        Returns
        -------
        str
        """
        pass

    @property
    @abstractmethod
    def raw_model(self):
        """
        Contains the raw ML model built on its framework

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
        Two-dimensional probability prediction of ml model.

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

    def get_ordered_features(self, x):
        """
        Restores the correct input feature order for the ML model, this also drops the columns not in the
        feature order. So it drops the target column, and possibly other features, e.g. categorical.

        Only works for encoded data

        Parameters
        ----------
        x : pd.DataFrame
            Data we want to order

        Returns
        -------
        output : pd.DataFrame
            Whole DataFrame with ordered feature
        """
        if isinstance(x, pd.DataFrame):
            return order_data(self.feature_input_order, x)
        else:
            warnings.warn(
                f"cannot re-order features for non dataframe input: {type(x)}"
            )
            return x

    def get_mutable_mask(self):
        """
        Get mask of mutable features.

        For example with mutable feature "income" and immutable features "age", the
        mask would be [True, False] for feature_input_order ["income", "age"].

        This mask can then be used to index data to only get the columns that are (im)mutable.

        Returns
        -------
        mutable_mask: np.array(bool)
        """
        # get categorical features
        categorical = self.data.categorical
        # get the binary encoded categorical features
        encoded_categorical = self.data.encoder.get_feature_names(categorical)
        # get the immutables, where the categorical features are in encoded format
        immutable = [
            encoded_categorical[categorical.index(i)] if i in categorical else i
            for i in self.data.immutables
        ]
        # find the index of the immutables in the feature input order
        immutable = [self.feature_input_order.index(col) for col in immutable]
        # make a mask
        mutable_mask = np.ones(len(self.feature_input_order), dtype=bool)
        # set the immutables to False
        mutable_mask[immutable] = False
        return mutable_mask
