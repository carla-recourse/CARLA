import pandas as pd

from carla.evaluation.api.evaluation import Evaluation


def _success_rate(counterfactuals: pd.DataFrame) -> float:
    """
    Computes success rate for all counterfactuals.

    Parameters
    ----------
    counterfactuals:
        All counterfactual examples inclusive nan values.

    Returns
    -------

    """
    total_num_counterfactuals = len(counterfactuals)
    successful_counterfactuals = len(counterfactuals.dropna())
    success_rate = successful_counterfactuals / total_num_counterfactuals
    return success_rate


class SuccessRate(Evaluation):
    """
    Computes success rate for the whole recourse method.
    """

    def __init__(self):
        super().__init__(None)
        self.columns = ["Success_Rate"]

    def get_evaluation(self, factuals, counterfactuals):
        rate = _success_rate(counterfactuals)
        return pd.DataFrame([[rate]], columns=self.columns)
