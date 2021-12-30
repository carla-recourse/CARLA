import numpy as np
import pandas as pd

from carla.data.causal_model import CausalModel


class Sampler:
    def __init__(self, scm: CausalModel):
        self._scm = scm
        self._input_attributes = list(scm.structural_equations_np.keys())

    @property
    def input_attributes(self):
        return self._input_attributes

    def _create_counterfactual_template(self, action_set: dict, factual_instance: dict):

        counterfactual_template = dict.fromkeys(self.input_attributes, np.NaN)

        # get intervention and conditioning sets
        intervention_set = set(action_set.keys())

        # intersection of non-descendents of intervened upon variables
        conditioning_set = set.intersection(
            *[self._scm.get_non_descendents(node) for node in intervention_set]
        )

        # check there is no intersection
        if not set.intersection(intervention_set, conditioning_set) == set():
            raise ValueError

        # set values in intervention and conditioning sets
        for node in conditioning_set:
            counterfactual_template[node] = factual_instance[node]

        for node in intervention_set:
            counterfactual_template[node] = action_set[node]

        return counterfactual_template

    def _create_factual_df(self, num_samples, factual_instance):
        factual_df = pd.DataFrame(
            dict(
                zip(
                    self.input_attributes,
                    [
                        num_samples * [factual_instance.dict()[node]]
                        for node in self.input_attributes
                    ],
                )
            )
        )
        return factual_df

    def _create_samples_df(self, num_samples, factual_instance, action_set):

        counterfactual_template = self._create_counterfactual_template(
            action_set, factual_instance
        )

        # this dataframe has populated columns set to intervention or conditioning values
        # and has NaN columns that will be set accordingly.
        samples_df = pd.DataFrame(
            dict(
                zip(
                    self.input_attributes,
                    [
                        num_samples * [counterfactual_template[node]]
                        for node in self.input_attributes
                    ],
                )
            )
        )
        return samples_df

    def sample(
        self,
        num_samples: int,
        factual_instance: pd.Series,
        action_set: dict,
        sampling_handle,
    ):
        """
        Get sample based on the factual instance and it's perturbation.

        Parameters
        ----------
        num_samples: int
            Number of samples to return.
        factual_instance: pd.Series
            Contains a single factual instance, where each element corresponds to a feature.
        action_set: dict
            Contains perturbation of features.
        sampling_handle: function
            Function that control the sampling.

        Returns
        -------
        pd.DataFrame
        """

        samples_df = self._create_samples_df(num_samples, factual_instance, action_set)

        # Simply traverse the graph in order, and populate nodes as we go!
        for node in self._scm.get_topological_ordering():
            # set variable if value not yet set through either intervention or conditioning
            if samples_df[node].isnull().values.any():

                parents = self._scm.get_parents(node)
                assert len(parents) > 0
                assert not samples_df.loc[:, list(parents)].isnull().values.any()

                samples_df[node] = sampling_handle(
                    node, self._scm, samples_df, factual_instance
                )

        if not np.all(list(samples_df.columns) == self.input_attributes):
            raise ValueError("ordering of column names has changed unexpectedly")

        return samples_df
