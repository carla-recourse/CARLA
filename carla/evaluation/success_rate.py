import pandas as pd


def success_rate(counterfactuals: pd.DataFrame) -> float:
    """
    Computes success rate for all counterfactuals

    Parameters
    ----------
    counterfactuals: All counterfactual examples inclusive nan values

    Returns
    -------

    """
    return (counterfactuals.dropna().shape[0]) / counterfactuals.shape[0]
