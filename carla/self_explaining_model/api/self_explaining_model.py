from abc import abstractmethod
import pandas as pd

from carla.models.api import MLModel
from carla.data.catalog import DataCatalog

class SelfExplainingModel(MLModel):

    def _init__(self,data:DataCatalog):
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