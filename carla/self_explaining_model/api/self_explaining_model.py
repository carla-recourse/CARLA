from abc import ABC,abstractmethod
import pandas as pd

from carla.models.api import MLModel
from carla.data import Data

class SelfExplainingModel(MLModel,ABC):
    """
    Abstract class to implement custom self explaining methods.

    Parameters
    ----------
    data: Data
        Dataset inherited from Data-wrapper

    Methods
    -------
    get_counterfactuals:
        Generate counterfactual examples for given factuals.

    Returns
    -------
    None
    """
    def _init__(self,data:Data):
        super().__init__(data)

    @abstractmethod
    def get_counterfactuals(self, factuals: pd.DataFrame):
        """
        Generate counterfactual examples for given factuals.

        Parameters
        ----------
        factuals: pd.DataFrame
            Not encoded and not normalised factual examples in two-dimensional shape (m, n).

        Returns
        -------
        pd.DataFrame
            Encoded and normalised counterfactual examples.
        """
        pass