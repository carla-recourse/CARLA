import pandas as pd

from carla.evaluation.distances import get_distances
from carla.evaluation.violations import constraint_violation
from carla.models.api import MLModel
from carla.models.pipelining import encode, scale
from carla.recourse_methods.api import RecourseMethod


class Benchmark:
    def __init__(
        self, mlmodel: MLModel, recourse_method: RecourseMethod, factuals: pd.DataFrame
    ) -> None:
        """
        Constructor for benchmarking class

        Parameters
        ----------
        mlmodel: MLModel
            Black Box model we want to explain
        recmodel: RecourseMethod
            Recourse method we want to benchmark
        factuals: pd.DataFrame
            Instances we want to find counterfactuals
        """
        self._mlmodel = mlmodel
        self._counterfactuals = recourse_method.get_counterfactuals(factuals)
        self._factuals = factuals.copy()

        # Normalizing and encoding factual for later use
        self._enc_norm_factuals = scale(
            mlmodel.scaler, mlmodel.data.continous, factuals
        )
        self._enc_norm_factuals = encode(
            mlmodel.encoder, mlmodel.data.categoricals, self._enc_norm_factuals
        )
        self._enc_norm_factuals = self._enc_norm_factuals[
            mlmodel.feature_input_order + [mlmodel.data.target]
        ]

    def compute_distances(self) -> pd.DataFrame:
        """
        Calculates the distance measure and returns it as dataframe

        Returns
        -------
        pd.DataFrame
        """
        arr_f = self._enc_norm_factuals.to_numpy()
        arr_cf = self._counterfactuals.to_numpy()

        distances = get_distances(arr_f, arr_cf)
        columns = ["Distance_1", "Distance_2", "Distance_3", "Distance_4"]

        output = pd.DataFrame(distances, columns=columns)

        return output

    def compute_constraint_violation(self) -> pd.DataFrame:
        """
        Computes the constraint violation per factual as dataframe

        Returns
        -------
        pd.Dataframe
        """
        violations = constraint_violation(
            self._mlmodel, self._counterfactuals, self._factuals
        )
        columns = ["Constraint_Violation"]

        return pd.DataFrame(violations, columns=columns)

    def run_benchmark(self) -> pd.DataFrame:
        """
        Runs every measurement and returns every value as dict.

        Returns
        -------
        pd.DataFrame
        """
        # TODO: Extend with implementation of further measurements
        pipeline = [self.compute_distances(), self.compute_constraint_violation()]

        output = pd.concat(pipeline, axis=1)

        return output
