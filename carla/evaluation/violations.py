from typing import List

import numpy as np
import pandas as pd

from carla.data.api import Data


def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))


def constraint_violation(
    data: Data, counterfactuals: pd.DataFrame, factuals: pd.DataFrame
) -> List[List[float]]:
    """
    Counts constraint violation per counterfactual

    Parameters
    ----------
    data: Data object
    counterfactuals: Normalized and encoded counterfactual examples
    factuals: Normalized and encoded factuals

    Returns
    -------

    """
    df_decoded_cfs = data.inverse_transform(counterfactuals.copy())
    df_factuals = data.inverse_transform(factuals.copy())

    # check continuous using np.isclose to allow for very small numerical differences
    cfs_continuous_immutable = df_decoded_cfs[
        intersection(data.continuous, data.immutables)
    ]
    factual_continuous_immutable = df_factuals[
        intersection(data.continuous, data.immutables)
    ]

    continuous_violations = np.invert(
        np.isclose(cfs_continuous_immutable, factual_continuous_immutable)
    )
    continuous_violations = np.sum(continuous_violations, axis=1).reshape(
        (-1, 1)
    )  # sum over features

    # check categorical by boolean comparison
    cfs_categorical_immutable = df_decoded_cfs[
        intersection(data.categorical, data.immutables)
    ]
    factual_categorical_immutable = df_factuals[
        intersection(data.categorical, data.immutables)
    ]

    categorical_violations = cfs_categorical_immutable != factual_categorical_immutable
    categorical_violations = np.sum(categorical_violations.values, axis=1).reshape(
        (-1, 1)
    )  # sum over features

    total_violations = continuous_violations + categorical_violations
    return total_violations.tolist()
