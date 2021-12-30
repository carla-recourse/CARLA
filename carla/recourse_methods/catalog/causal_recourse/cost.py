import numpy as np
import pandas as pd
import torch


def action_set_cost(
    factual_instance: pd.Series,
    action_set: dict,
    ranges: pd.Series,
    norm_type=2,
) -> float:
    """
    Compute cost of action

    Parameters
    ----------
    factual_instance: pd.Series
        Contains a single factual instance, where each element corresponds to a feature.
    action_set: dict
        Contains perturbation of features.
    ranges: pd.Series
        Contains the feature ranges of the original dataset.
    norm_type: int
        Norm to be used, choose either 1 or 2.

    Returns
    -------
    float

    """

    factual_instance = factual_instance.to_dict()
    ranges = ranges.to_dict()

    factual_float_int = np.all(
        [isinstance(elem, (int, float)) for elem in factual_instance.values()]
    )
    action_float_int = np.all(
        [isinstance(elem, (int, float)) for elem in action_set.values()]
    )

    factual_torch = np.all(
        [isinstance(elem, torch.Tensor) for elem in factual_instance.values()]
    )
    action_torch = np.all(
        [isinstance(elem, torch.Tensor) for elem in action_set.values()]
    )

    deltas = [
        (action_set[key] - factual_instance[key]) / ranges[key]
        for key in action_set.keys()
    ]

    if factual_float_int and action_float_int:
        return np.linalg.norm(deltas, norm_type)
    elif factual_torch and action_torch:
        return torch.norm(torch.stack(deltas), p=norm_type)
    else:
        raise Exception("Mismatching or unsupported datatypes.")
