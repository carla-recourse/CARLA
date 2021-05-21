from typing import Union

import pandas as pd

from carla.evaluation.distances import get_distances
from carla.evaluation.process_nans import remove_nans
from carla.evaluation.redundancy import redundancy
from carla.evaluation.success_rate import success_rate
from carla.evaluation.violations import constraint_violation
from carla.models.api import MLModel
from carla.models.catalog import MLModelCatalog
from carla.models.pipelining import encode, scale
from carla.recourse_methods.api import RecourseMethod


class Benchmark:
    def __init__(
        self,
        mlmodel: Union[MLModel, MLModelCatalog],
        recourse_method: RecourseMethod,
        factuals: pd.DataFrame,
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

        # Avoid using scaling and normalizing more than once
        if isinstance(mlmodel, MLModelCatalog):
            self._mlmodel.use_pipeline = False  # type: ignore

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
        factual_without_nans, counterfactuals_without_nans = remove_nans(
            self._enc_norm_factuals, self._counterfactuals
        )
        arr_f = factual_without_nans.to_numpy()
        arr_cf = counterfactuals_without_nans.to_numpy()

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
        factual_without_nans, counterfactuals_without_nans = remove_nans(
            self._factuals, self._counterfactuals
        )

        violations = constraint_violation(
            self._mlmodel, counterfactuals_without_nans, factual_without_nans
        )
        columns = ["Constraint_Violation"]

        return pd.DataFrame(violations, columns=columns)

    def compute_redundancy(self) -> pd.DataFrame:
        """
        Computes redundancy for each counterfactual

        Returns
        -------
        pd.Dataframe
        """
        factual_without_nans, counterfactuals_without_nans = remove_nans(
            self._enc_norm_factuals, self._counterfactuals
        )

        redundancies = redundancy(
            factual_without_nans, counterfactuals_without_nans, self._mlmodel
        )
        columns = ["Redundancy"]

        return pd.DataFrame(redundancies, columns=columns)

    def compute_success_rate(self) -> pd.DataFrame:
        """
        Computes success rate for the whole recourse method.

        Returns
        -------
        pd.Dataframe
        """

        rate = success_rate(self._counterfactuals)
        columns = ["Success_Rate"]

        return pd.DataFrame([[rate]], columns=columns)

    def run_benchmark(self) -> pd.DataFrame:
        """
        Runs every measurement and returns every value as dict.

        Returns
        -------
        pd.DataFrame
        """
        # TODO: Extend with implementation of further measurements
        pipeline = [
            self.compute_distances(),
            self.compute_constraint_violation(),
            self.compute_redundancy(),
            self.compute_success_rate(),
        ]

        output = pd.concat(pipeline, axis=1)

        return output
