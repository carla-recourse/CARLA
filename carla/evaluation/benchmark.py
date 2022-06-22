import timeit
from typing import Union

import pandas as pd

from carla.evaluation import YNN, ConstraintViolation, Distance, Redundancy, SuccessRate
from carla.models.api import MLModel
from carla.models.catalog import MLModelCatalog
from carla.recourse_methods.api import RecourseMethod


class Benchmark:
    """
    The benchmarking class contains all measurements.
    It is possible to run only individual evaluation metrics or all via one single call.

    For every given factual, the benchmark object will generate one counterfactual example with
    the given recourse method.

    Parameters
    ----------
    mlmodel: carla.models.MLModel
        Black Box model we want to explain
    recmodel: carla.recourse_methods.RecourseMethod
        Recourse method we want to benchmark
    factuals: pd.DataFrame
        Instances we want to find counterfactuals

    Methods
    -------
    compute_ynn:
        Computes y-Nearest-Neighbours for generated counterfactuals
    compute_average_time:
        Computes average time for generated counterfactual
    compute_distances:
        Calculates the distance measure and returns it as dataframe
    compute_constraint_violation:
        Computes the constraint violation per factual as dataframe
    compute_redundancy:
        Computes redundancy for each counterfactual
    compute_success_rate:
        Computes success rate for the whole recourse method.
    run_benchmark:
        Runs every measurement and returns every value as dict.
    """

    def __init__(
        self,
        mlmodel: Union[MLModel, MLModelCatalog],
        recourse_method: RecourseMethod,
        factuals: pd.DataFrame,
    ) -> None:

        self._mlmodel = mlmodel
        self._recourse_method = recourse_method
        self._factuals = self._mlmodel.get_ordered_features(factuals.copy())

        start = timeit.default_timer()
        self._counterfactuals = recourse_method.get_counterfactuals(factuals)
        stop = timeit.default_timer()
        self._timer = stop - start

    def compute_ynn(self) -> pd.DataFrame:
        """
        Computes y-Nearest-Neighbours for generated counterfactuals

        Returns
        -------
        pd.DataFrame
        """
        return YNN(self._mlmodel, {"y": 5, "cf_label": 1}).get_evaluation(
            counterfactuals=self._counterfactuals, factuals=self._factuals
        )

    def compute_average_time(self) -> pd.DataFrame:
        """
        Computes average time for generated counterfactual

        Returns
        -------
        pd.DataFrame
        """

        avg_time = self._timer / self._counterfactuals.shape[0]

        columns = ["Average_Time"]

        return pd.DataFrame([[avg_time]], columns=columns)

    def compute_distances(self) -> pd.DataFrame:
        """
        Calculates the distance measure and returns it as dataframe

        Returns
        -------
        pd.DataFrame
        """
        return Distance().get_evaluation(
            factuals=self._factuals,
            counterfactuals=self._counterfactuals,
        )

    def compute_constraint_violation(self) -> pd.DataFrame:
        """
        Computes the constraint violation per factual as dataframe

        Returns
        -------
        pd.Dataframe
        """
        return ConstraintViolation(self._mlmodel).get_evaluation(
            counterfactuals=self._counterfactuals,
            factuals=self._factuals,
        )

    def compute_redundancy(self) -> pd.DataFrame:
        """
        Computes redundancy for each counterfactual

        Returns
        -------
        pd.Dataframe
        """
        return Redundancy(self._mlmodel, {"cf_label": 1}).get_evaluation(
            counterfactuals=self._counterfactuals,
            factuals=self._factuals,
        )

    def compute_success_rate(self) -> pd.DataFrame:
        """
        Computes success rate for the whole recourse method.

        Returns
        -------
        pd.Dataframe
        """
        return SuccessRate(self._mlmodel).get_evaluation(
            counterfactuals=self._counterfactuals, factuals=self._factuals
        )

    def run_benchmark(self) -> pd.DataFrame:
        """
        Runs every measurement and returns every value as dict.

        Returns
        -------
        pd.DataFrame
        """
        pipeline = [
            self.compute_distances(),
            self.compute_constraint_violation(),
            self.compute_redundancy(),
            self.compute_ynn(),
            self.compute_success_rate(),
            self.compute_average_time(),
        ]

        output = pd.concat(pipeline, axis=1)

        return output
