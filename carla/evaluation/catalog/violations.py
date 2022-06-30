from typing import List

import numpy as np
import pandas as pd

from carla.data.api import Data
from carla.evaluation import remove_nans
from carla.evaluation.api import Evaluation


def _intersection(list1: List, list2: List):
    """Compute the intersection between two lists"""
    return list(set(list1) & set(list2))


def constraint_violation(
    data: Data, counterfactuals: pd.DataFrame, factuals: pd.DataFrame
) -> List[List[float]]:
    """
    Counts constraint violation per counterfactual

    Parameters
    ----------
    data:

    counterfactuals:
        Normalized and encoded counterfactual examples.
    factuals:
        Normalized and encoded factuals.

    Returns
    -------

    """
    df_decoded_cfs = data.inverse_transform(counterfactuals.copy())
    df_factuals = data.inverse_transform(factuals.copy())

    """
    continuous features
    """
    # check continuous using np.isclose to allow for very small numerical differences
    cfs_continuous_immutable = df_decoded_cfs[
        _intersection(data.continuous, data.immutables)
    ]
    factual_continuous_immutable = df_factuals[
        _intersection(data.continuous, data.immutables)
    ]

    continuous_violations = np.invert(
        np.isclose(cfs_continuous_immutable, factual_continuous_immutable)
    )
    continuous_violations = np.sum(continuous_violations, axis=1).reshape(
        (-1, 1)
    )  # sum over features

    """
    categorical features
    """
    # check categorical by boolean comparison
    cfs_categorical_immutable = df_decoded_cfs[
        _intersection(data.categorical, data.immutables)
    ]
    factual_categorical_immutable = df_factuals[
        _intersection(data.categorical, data.immutables)
    ]

    categorical_violations = cfs_categorical_immutable != factual_categorical_immutable
    categorical_violations = np.sum(categorical_violations.values, axis=1).reshape(
        (-1, 1)
    )  # sum over features

    total_violations = continuous_violations + categorical_violations
    return total_violations.tolist()


class ConstraintViolation(Evaluation):
    """
    Computes the constraint violation per factual as dataframe
    """

    def __init__(self, mlmodel):
        super().__init__(mlmodel)
        self.columns = ["Constraint_Violation"]

    def get_evaluation(self, factuals, counterfactuals):
        counterfactuals_without_nans, factual_without_nans = remove_nans(
            counterfactuals, factuals
        )

        if counterfactuals_without_nans.empty:
            violations = []
        else:
            violations = constraint_violation(
                self.mlmodel.data, counterfactuals_without_nans, factual_without_nans
            )

        return pd.DataFrame(violations, columns=self.columns)
