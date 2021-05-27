from typing import List

import numpy as np
import pandas as pd

from carla.models.api import MLModel


def check_counterfactuals(
    mlmodel: MLModel, counterfactuals: List, negative_label: int = 0
) -> pd.DataFrame:
    """
    Takes the generated list of counterfactuals from recourse methods and checks if these samples are able
    to flip the label from 0 to 1. Every counterfactual which still has a negative label, will be replaced with an
    empty row.

    Parameters
    ----------
    mlmodel: Black-box-model we want to discover
    counterfactuals: List of generated samples from recourse method
    negative_label: Defines the negative label.

    Returns
    -------
    pd.DataFrame
    """
    df_cfs = pd.DataFrame(
        np.array(counterfactuals), columns=mlmodel.feature_input_order
    )
    df_cfs[mlmodel.data.target] = np.argmax(mlmodel.predict_proba(df_cfs), axis=1)
    # Change all wrong counterfactuals to nan
    df_cfs.loc[df_cfs[mlmodel.data.target] == negative_label, :] = np.nan

    return df_cfs
