from typing import List, Union

import numpy as np
import pandas as pd

from carla.models.api import MLModel


def check_counterfactuals(
    mlmodel: MLModel,
    counterfactuals: Union[List, pd.DataFrame],
    negative_label: int = 0,
) -> pd.DataFrame:
    """
    Takes the generated list of counterfactuals from recourse methods and checks if these samples are able
    to flip the label from 0 to 1. Every counterfactual which still has a negative label, will be replaced with an
    empty row.

    Parameters
    ----------
    mlmodel: Black-box-model we want to discover
    counterfactuals: List or DataFrame of generated samples from recourse method
    negative_label: Defines the negative label.

    Returns
    -------
    pd.DataFrame
    """
    if isinstance(counterfactuals, list):
        df_cfs = pd.DataFrame(
            np.array(counterfactuals), columns=mlmodel.feature_input_order
        )
    else:
        df_cfs = counterfactuals.copy()

    df_cfs[mlmodel.data.target] = np.argmax(mlmodel.predict_proba(df_cfs), axis=1)
    # Change all wrong counterfactuals to nan
    df_cfs.loc[df_cfs[mlmodel.data.target] == negative_label, :] = np.nan

    return df_cfs


def get_drop_columns_binary(categoricals: List[str], columns: List[str]) -> List[str]:
    """
    Selects the columns which can be dropped for one-hot-encoded dfs without drop_first.

    Is mainly needed to transform into drop_first encoding

    Parameters
    ----------
    categoricals: non encoded categorical feature names
    columns: one-hot-encoded features without drop_first

    Returns
    -------
    List of features to drop
    """
    list_drop = [
        c for c in columns if c.split("_")[0] in [c.split("_")[0] for c in categoricals]
    ]
    return list_drop[::2]
