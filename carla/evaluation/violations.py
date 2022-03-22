from typing import List

import numpy as np
import pandas as pd

from carla.data.api import Data


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
    immutables = data.immutables

    df_decoded_cfs = data.inverse_transform(counterfactuals.copy())
    df_decoded_cfs[data.continuous] = df_decoded_cfs[data.continuous].astype("int64")
    df_decoded_cfs = df_decoded_cfs[immutables]

    df_factuals = data.inverse_transform(factuals)
    df_factuals[data.continuous] = df_factuals[data.continuous].astype("int64")
    df_factuals = df_factuals[immutables]

    # todo add pandas testing rather then casting as ints above?
    logical = df_factuals != df_decoded_cfs
    logical = np.sum(logical.values, axis=1).reshape((-1, 1))

    return logical.tolist()
