from typing import Tuple

import pandas as pd


def remove_nans(
    factuals: pd.DataFrame, counterfactuals: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """

    Parameters
    ----------
    factuals: Has to be the same shape as counterfactuals
    counterfactuals: Has to be the same shape as factuals

    Returns
    -------

    """
    if factuals.shape[0] != counterfactuals.shape[0]:
        raise ValueError(
            "Counterfactuals and factuals should contain the same amount of samples"
        )

    nan_idx = counterfactuals.index[counterfactuals.isnull().any(axis=1)]

    output_factuals = factuals.copy()
    output_counterfactuals = counterfactuals.copy()

    output_factuals = output_factuals.drop(index=nan_idx)
    output_counterfactuals = output_counterfactuals.drop(index=nan_idx)

    return output_factuals, output_counterfactuals
