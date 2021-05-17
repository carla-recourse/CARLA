import pandas as pd

from carla.evaluation.distances import get_distances
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
        self._recmodel = recourse_method
        self._counterfactuals = self._recmodel.get_counterfactuals(factuals)

        # Normalizing and encoding factual for later use
        self._factuals = scale(mlmodel.scaler, mlmodel.data.continous, factuals)
        self._factuals = encode(
            mlmodel.encoder, mlmodel.data.categoricals, self._factuals
        )
        self._factuals = self._factuals[
            mlmodel.feature_input_order + [mlmodel.data.target]
        ]

    def compute_distances(self) -> pd.DataFrame:
        """
        Calculates the distance measure and returns it as dataframe

        Returns
        -------
        pd.DataFrame
        """
        arr_f = self._factuals.to_numpy()
        arr_cf = self._counterfactuals.to_numpy()

        distances = get_distances(arr_f, arr_cf)
        columns = ["Distance_1", "Distance_2", "Distance_3", "Distance_4"]

        output = pd.DataFrame(distances, columns=columns)

        return output

    def run_benchmark(self) -> pd.DataFrame:
        """
        Runs every measurement and returns every value as dict.

        Returns
        -------
        pd.DataFrame
        """
        # TODO: Extend with implementation of further measurements
        pipeline = [self.compute_distances()]

        output = pd.concat(pipeline, axis=1)

        return output
