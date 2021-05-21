from typing import List

import numpy as np
import pandas as pd

from carla.models.api import MLModel
from carla.models.pipelining import decode


def constraint_violation(
    mlmodel: MLModel, counterfactuals: pd.DataFrame, factuals: pd.DataFrame
) -> List[List[float]]:
    """
    Counts constraint violation per counterfactual

    Parameters
    ----------
    mlmodel: Black-box-model we want to discover
    counterfactuals: Normalized and encoded counterfactual examples
    factuals: Not normalized and encoded factuals

    Returns
    -------

    """
    immutables = mlmodel.data.immutables

    # Decode counterfactuals to compare immutables with not encoded factuals
    df_decoded_cfs = counterfactuals.copy()
    df_decoded_cfs = decode(mlmodel.encoder, mlmodel.data.categoricals, df_decoded_cfs)
    df_decoded_cfs[mlmodel.data.continous] = mlmodel.scaler.inverse_transform(
        df_decoded_cfs[mlmodel.data.continous]
    )
    df_decoded_cfs[mlmodel.data.continous] = df_decoded_cfs[
        mlmodel.data.continous
    ].astype(
        "int64"
    )  # avoid precision error

    df_decoded_cfs = df_decoded_cfs[immutables]
    df_factuals = factuals[immutables]

    logical = df_factuals != df_decoded_cfs
    logical = np.sum(logical.values, axis=1).reshape((-1, 1))

    return logical.tolist()
