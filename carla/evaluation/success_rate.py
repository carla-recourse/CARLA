import pandas as pd

from carla.evaluation.evaluation import Evaluation


class SuccessRate(Evaluation):
    def __init__(self, mlmodel):
        super().__init__(mlmodel)
        self.columns = ["Success_Rate"]

    def compute_success_rate(self, counterfactuals: pd.DataFrame) -> float:
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

    def get_evaluation(self, counterfactuals, factuals):
        rate = self.compute_success_rate(counterfactuals)

        return pd.DataFrame([[rate]], columns=self.columns)
