import itertools
from typing import Dict

import numpy as np
import pandas as pd
from tqdm import tqdm

from carla.models.catalog import MLModelCatalog
from carla.recourse_methods.api import RecourseMethod

from .action_set import get_discretized_action_sets
from .cost import action_set_cost


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
        # self._recourse_type = hyperparams["recourse_type"]
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

    def _get_ranges(self):
        normalized = self._mlmodel.use_pipeline
        if normalized:
            data_df = self.encode_normalize_order_factuals(self._dataset.raw)
        else:
            data_df = self._dataset.raw

        min_values = data_df.min()
        max_values = data_df.max()

        # ranges = pd.concat([min_values, max_values], axis=1)
        ranges = max_values - min_values
        return ranges

    def compute_optimal_action_set(
        self, factual_instance, constraint_handle, sampling_handle
    ):

        intervenables_nodes = self.get_intervenable_nodes()
        ranges = self._get_ranges()

        min_cost = np.infty
        min_action_set = {}
        if self._optimization_approach == "brute_force":
            valid_action_sets = get_discretized_action_sets(
                intervenables_nodes, self._mlmodel.scaler
            )
            for action_set in tqdm(valid_action_sets):
                if constraint_handle(
                    self._scm,
                    factual_instance,
                    action_set,
                    sampling_handle,
                    self._mlmodel,
                ):
                    cost = action_set_cost(factual_instance, action_set, ranges)
                    if cost < min_cost:
                        min_cost = cost
                        min_action_set = action_set

        elif self._optimization_approach == "gradient_descent":
            raise NotImplementedError
            #
            # for i, intervention_set in enumerate(valid_intervention_sets):
            #     action_set, cost = self.perform_gradient_descent(
            #         factual_instance, intervention_set
            #     )
            #
            #     if self.constraint(action_set, factual_instance):
            #         if cost < min_cost:
            #             min_cost = cost
            #             min_action_set = action_set
        else:
            raise ValueError("optimization approach not recognized")

        return min_action_set

    def get_counterfactuals(self, factuals: pd.DataFrame):

        cfs = []
        for index, factual_instance in factuals.iterrows():
            min_action_set = self.compute_optimal_action_set(
                factual_instance, self._constraint_handle, self._sampler_handle
            )
            cfs.append(min_action_set)
        return cfs
