from abc import ABC, abstractmethod

import pandas as pd


class Evaluation(ABC):
    def __init__(self, mlmodel, hyperparameters: dict = None):
        """

        Parameters
        ----------
        mlmodel:
            Classification model. (optional)
        hyperparameters:
            Dictionary with hyperparameters, could be used to pass other things. (optional)
        """
        self.mlmodel = mlmodel
        self.hyperparameters = hyperparameters

    @abstractmethod
    def get_evaluation(
        self, factuals: pd.DataFrame, counterfactuals: pd.DataFrame
    ) -> pd.DataFrame:
        """Compute evaluation measure"""
