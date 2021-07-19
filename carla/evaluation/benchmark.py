import timeit
from typing import Union

import numpy as np
import pandas as pd

from carla.evaluation.distances import get_distances
from carla.evaluation.nearest_neighbours import yNN
from carla.evaluation.process_nans import remove_nans
from carla.evaluation.redundancy import redundancy
from carla.evaluation.success_rate import success_rate
from carla.evaluation.violations import constraint_violation
from carla.models.api import MLModel
from carla.models.catalog import MLModelCatalog
from carla.recourse_methods.api import RecourseMethod
from carla.recourse_methods.processing import get_drop_columns_binary


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
        start = timeit.default_timer()
        self._counterfactuals = recourse_method.get_counterfactuals(factuals)
        stop = timeit.default_timer()
        self._timer = stop - start

        # Avoid using scaling and normalizing more than once
        if isinstance(mlmodel, MLModelCatalog):
            self._mlmodel.use_pipeline = False  # type: ignore

        self._factuals = factuals.copy()

        # Normalizing and encoding factual for later use
        self._enc_norm_factuals = recourse_method.encode_normalize_order_factuals(
            factuals, with_target=True
        )

    def compute_ynn(self) -> pd.DataFrame:
        """
        Computes y-Nearest-Neighbours for generated counterfactuals

        Returns
        -------
        pd.DataFrame
        """
        _, counterfactuals_without_nans = remove_nans(
            self._factuals, self._counterfactuals
        )

        if counterfactuals_without_nans.empty:
            ynn = np.nan
        else:
            ynn = yNN(
                counterfactuals_without_nans, self._recourse_method, self._mlmodel, 5
            )

        columns = ["y-Nearest-Neighbours"]

        return pd.DataFrame([[ynn]], columns=columns)

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
        factual_without_nans, counterfactuals_without_nans = remove_nans(
            self._enc_norm_factuals, self._counterfactuals
        )

        columns = ["Distance_1", "Distance_2", "Distance_3", "Distance_4"]

        if counterfactuals_without_nans.empty:
            return pd.DataFrame(columns=columns)

        if self._mlmodel.encoder.drop is None:
            # To prevent double count of encoded features without drop if_binary
            binary_columns_to_drop = get_drop_columns_binary(
                self._mlmodel.data.categoricals,
                counterfactuals_without_nans.columns.tolist(),
            )
            counterfactuals_without_nans = counterfactuals_without_nans.drop(
                binary_columns_to_drop, axis=1
            )
            factual_without_nans = factual_without_nans.drop(
                binary_columns_to_drop, axis=1
            )

        arr_f = factual_without_nans.to_numpy()
        arr_cf = counterfactuals_without_nans.to_numpy()

        distances = get_distances(arr_f, arr_cf)

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

        if counterfactuals_without_nans.empty:
            violations = []
        else:
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

        if counterfactuals_without_nans.empty:
            redundancies = []
        else:
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
