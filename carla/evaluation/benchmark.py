import timeit
from typing import List

import pandas as pd

from carla.evaluation.api import Evaluation
from carla.models.api import MLModel
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
        Black Box model we want to explain.
    recourse_method: carla.recourse_methods.RecourseMethod
        Recourse method we want to benchmark.
    factuals: pd.DataFrame
        Instances we want to find counterfactuals.

    Methods
    -------
    compute_ynn:
        Computes y-Nearest-Neighbours for generated counterfactuals
    compute_average_time:
        Computes average time for generated counterfactual
    compute_distances:
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
        mlmodel: MLModel,
        recourse_method: RecourseMethod,
        factuals: pd.DataFrame,
    ) -> None:

        self.mlmodel = mlmodel
        self._recourse_method = recourse_method
        self._factuals = self.mlmodel.get_ordered_features(factuals.copy())

        start = timeit.default_timer()
        self._counterfactuals = recourse_method.get_counterfactuals(factuals)
        stop = timeit.default_timer()
        self.timer = stop - start

    def run_benchmark(self, measures: List[Evaluation]) -> pd.DataFrame:
        """
        Runs every measurement and returns every value as dict.

        Returns
        -------
        pd.DataFrame
        """
        pipeline = [
            measure.get_evaluation(
                counterfactuals=self._counterfactuals, factuals=self._factuals
            )
            for measure in measures
        ]

        output = pd.concat(pipeline, axis=1)

        return output
