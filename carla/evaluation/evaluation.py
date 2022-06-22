from abc import ABC, abstractmethod

import pandas as pd


def remove_nans(counterfactuals: pd.DataFrame, factuals: pd.DataFrame = None):
    """

    Parameters
    ----------
    counterfactuals:
        Has to be the same shape as factuals
    factuals:
        Has to be the same shape as counterfactuals

    Returns
    -------

    """
    # get indices of unsuccessful counterfactuals
    nan_idx = counterfactuals.index[counterfactuals.isnull().any(axis=1)]
    output_counterfactuals = counterfactuals.copy()
    output_counterfactuals = output_counterfactuals.drop(index=nan_idx)

    if factuals is not None:
        if factuals.shape[0] != counterfactuals.shape[0]:
            raise ValueError(
                "Counterfactuals and factuals should contain the same amount of samples"
            )

        output_factuals = factuals.copy()
        output_factuals = output_factuals.drop(index=nan_idx)

        return output_counterfactuals, output_factuals

    return output_counterfactuals


class Evaluation(ABC):
    def __init__(self, mlmodel, hyperparameters=None):
        self.mlmodel = mlmodel
        self.hyperparameters = hyperparameters

    @abstractmethod
    def get_evaluation(self, factuals, counterfactuals):
        """Compute evaluation measure"""
