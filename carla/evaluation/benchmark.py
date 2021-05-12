import csv
from typing import Dict

import pandas as pd

from carla.evaluation.distances import get_distances
from carla.models.api import MLModel
from carla.models.pipelining import encode, scale
from carla.recourse_methods.api import RecourseMethod


class Benchmark:
    def __init__(
        self, mlmodel: MLModel, recmodel: RecourseMethod, factuals: pd.DataFrame
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
        self._recmodel = recmodel
        self._counterfactuals = self._recmodel.get_counterfactuals(factuals)

        # Normalizing and encoding factual for later use
        self._factuals = scale(mlmodel.scaler, mlmodel.data.continous, factuals)
        self._factuals = encode(
            mlmodel.encoder, mlmodel.data.categoricals, self._factuals
        )
        self._factuals = self._factuals[
            mlmodel.feature_input_order + [mlmodel.data.target]
        ]

    def compute_distances(self) -> Dict:
        """
        Calculates the distance measure and returns it as dict

        Returns
        -------
        Dict
        """
        key = "Distances"
        output: Dict = {key: []}
        arr_f = self._factuals.to_numpy()
        arr_cf = self._counterfactuals.to_numpy()

        distances = get_distances(arr_f, arr_cf)
        sub_keys = ["d1", "d2", "d3", "d4"]
        for i in range(len(distances)):
            output[key].append(dict(zip(sub_keys, distances[i])))

        return output

    def run_benchmark(self) -> Dict:
        """
        Runs every measurement and returns every value as dict.

        Returns
        -------
        Dict
        """
        output: Dict = dict()

        # TODO: Extend with implementation of further measurements
        pipeline = [self.compute_distances()]

        for measurement in pipeline:
            output = {**output, **measurement}

        return output

    def to_csv(self, eval: Dict[str, Dict], path: str) -> None:
        """
        Write dictionary with evaluation data into csv

        Parameters
        ----------
        eval: Dict[str, Dict]
            Evaluation data
        path: str
            Path and name of savefile

        Returns
        -------
        None
        """
        samples = []
        num_samples = len(eval[list(eval.keys())[0]])

        # bring data into correct format
        for i in range(num_samples):
            tocsv: Dict = dict()
            for key, val in eval.items():
                sample = val[i]
                tocsv = {**tocsv, **sample}
            samples.append(tocsv)

        csv_cols = list(samples[0].keys())

        # save data as csv
        try:
            with open(path, "w") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=csv_cols)
                writer.writeheader()
                for data in samples:
                    writer.writerow(data)
        except IOError:
            print("I/O error")
