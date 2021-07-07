from typing import Dict, List, Union

import numpy as np
import pandas as pd
import torch

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


def reconstruct_encoding_constraints(
    x: torch.Tensor, feature_pos: List[int], binary_cat: bool
) -> torch.Tensor:
    """
    Reconstructing one-hot-encoded data, such that its values are either 0 or 1,
    and features do not contradict (e.g., sex_female = 1, sex_male = 1)

    Parameters
    ----------
    x: instance where we want to reconstruct categorical constraints
    feature_pos: list with positions of categorical features in x
    binary_cat: If true, categorical datas are encoded with drop_if_binary

    Returns
    -------
    Tensor with reconstructed constraints
    """
    x_enc = x.clone()

    if binary_cat:
        for pos in feature_pos:
            x_enc[:, pos] = torch.round(x_enc[:, pos])
    else:
        binary_pairs = list(zip(feature_pos[:-1], feature_pos[1:]))[0::2]
        for pair in binary_pairs:
            # avoid overwritten inconsistent results
            temp = (x_enc[:, pair[0]] >= x_enc[:, pair[1]]).float()

            x_enc[:, pair[1]] = (x_enc[:, pair[0]] < x_enc[:, pair[1]]).float()
            x_enc[:, pair[0]] = temp

            if (x_enc[:, pair[0]] == x_enc[:, pair[1]]).any():
                raise ValueError(
                    "Reconstructing encoded features lead to an error. Feature {} and {} have the same value".format(
                        pair[0], pair[1]
                    )
                )

    return x_enc


def merge_default_parameters(hyperparams: Dict, default: Dict) -> Dict:
    """
    Checks if the input parameter hyperparams contains every necessary key and if not, uses default values or
    raises a ValueError if no default value is given.

    Parameters
    ----------
    hyperparams: dict
        Hyperparameter as passed to the recuorse method.
    default: dict
        Dictionary with every necessary key and default value.
        If key has no default value and hyperparams has no value for it, raise a ValueError

    Returns
    -------
    dict
        Dictionary with every necessary key.
    """
    keys = default.keys()
    dict_output = dict()

    for key in keys:
        if isinstance(default[key], dict):
            hyperparams[key] = (
                dict() if key not in hyperparams.keys() else hyperparams[key]
            )
            sub_dict = merge_default_parameters(hyperparams[key], default[key])
            dict_output[key] = sub_dict
            continue
        if key not in hyperparams.keys():
            default_val = default[key]
            if default_val is None:
                # None value for key depicts that user has to pass this value in hyperparams
                raise ValueError(
                    "For {} is no default value defined, please pass this key and its value in hyperparams".format(
                        key
                    )
                )
            elif isinstance(default_val, str) and default_val == "_optional_":
                # _optional_ depicts that value for this key is optional and therefore None
                default_val = None
            dict_output[key] = default_val
        else:
            if hyperparams[key] is None:
                raise ValueError("For {} in hyperparams is a value needed".format(key))
            dict_output[key] = hyperparams[key]

    return dict_output
