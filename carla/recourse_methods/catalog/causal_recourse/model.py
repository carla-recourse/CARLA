import itertools
from typing import Dict

import numpy as np
import pandas as pd
from sklearn import preprocessing

from carla.models.catalog import MLModelCatalog
from carla.recourse_methods.api import RecourseMethod

from .action_set import get_discretized_action_sets
from .cost import action_set_cost


def _series_plus_dict(x: pd.Series, y: dict):
    """Helper function to implemention addition for a Series object and a dict with overlapping keys

    Parameters
    ----------
    x: pd.Series
    y: dict

    Returns
    -------
    pd.Series analogous to x + y
    """
    y = pd.Series(y)

    result = x + y
    nan_cols = result.index[result.isna()].tolist()
    result = result.drop(index=nan_cols)
    result = pd.concat([result, x[nan_cols]])

    return result


# https://stackoverflow.com/a/1482316/2759976
def powerset(iterable):
    """powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"""
    s = list(iterable)
    return itertools.chain.from_iterable(
        itertools.combinations(s, r) for r in range(len(s) + 1)
    )


class CausalRecourse(RecourseMethod):
    def __init__(self, mlmodel: MLModelCatalog, hyperparams: Dict):

        self._mlmodel = mlmodel
        self._dataset = mlmodel.data

        self._optimization_approach = hyperparams["optimization_approach"]
        self._num_samples = hyperparams["num_samples"]
        self._scm = hyperparams["scm"]

        self._constraint_handle = hyperparams["constraint_handle"]
        self._sampler_handle = hyperparams["sampler_handle"]

    def get_intervenable_nodes(self) -> dict:
        intervenable_nodes = {
            "continuous": np.setdiff1d(
                self._dataset.continous, self._dataset.immutables
            ),
            "categorical": np.setdiff1d(
                self._dataset.categoricals, self._dataset.immutables
            ),
        }
        return intervenable_nodes

    def _get_original_df(self):
        normalized = self._mlmodel.use_pipeline
        if normalized:
            data_df = self.encode_normalize_order_factuals(self._dataset.raw)
        else:
            data_df = self._dataset.raw
        return data_df

    def _get_range_values(self):
        data_df = self._get_original_df()
        min_values = data_df.min()
        max_values = data_df.max()
        return min_values, max_values

    def _get_mean_values(self):
        data_df = self._get_original_df()
        mean_values = data_df.mean()
        return mean_values

    def compute_optimal_action_set(
        self, factual_instance, constraint_handle, sampling_handle
    ):

        intervenables_nodes = self.get_intervenable_nodes()
        min_values, max_values = self._get_range_values()
        mean_values = self._get_mean_values()

        min_cost = np.infty
        min_action_set = {}
        if self._optimization_approach == "brute_force":
            valid_action_sets = get_discretized_action_sets(
                intervenables_nodes, min_values, max_values, mean_values
            )

            # we need to make sure that actions don't go out of bounds [0, 1]
            if self._mlmodel.use_pipeline and isinstance(
                self._mlmodel.scaler, preprocessing.MinMaxScaler
            ):
                out_of_bounds_idx = []
                for i, action_set in enumerate(valid_action_sets):
                    instance = _series_plus_dict(factual_instance, action_set)
                    if not np.all((1 > instance.values) & (instance.values > 0)):
                        out_of_bounds_idx.append(i)
                valid_action_sets = [
                    action_set
                    for i, action_set in enumerate(valid_action_sets)
                    if i not in set(out_of_bounds_idx)
                ]

            for action_set in valid_action_sets:
                if constraint_handle(
                    self._scm,
                    factual_instance,
                    action_set,
                    sampling_handle,
                    self._mlmodel,
                ):
                    cost = action_set_cost(
                        factual_instance, action_set, max_values - min_values
                    )
                    if cost < min_cost:
                        min_cost = cost
                        min_action_set = action_set

        elif self._optimization_approach == "gradient_descent":
            raise NotImplementedError
        else:
            raise ValueError("optimization approach not recognized")

        # print("MIN COST", min_cost, "ACTION SET", min_action_set)

        return min_action_set, min_cost

    def get_counterfactuals(self, factuals: pd.DataFrame):

        normalized = self._mlmodel.use_pipeline
        if normalized:
            factual_df = self.encode_normalize_order_factuals(factuals)
        else:
            factual_df = factuals.drop(columns=self._dataset.target)

        cfs = []
        # actions = []
        for index, factual_instance in factual_df.iterrows():
            min_action_set, _ = self.compute_optimal_action_set(
                factual_instance, self._constraint_handle, self._sampler_handle
            )
            cf = _series_plus_dict(factual_instance, min_action_set)
            # min_action_set["cost"] = min_cost
            # actions.append(min_action_set)
            cfs.append(cf)

        # convert to dataframe
        cfs = pd.DataFrame(cfs)
        # action_df = pd.DataFrame(actions)
        return cfs
