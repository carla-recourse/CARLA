from typing import List

import numpy as np
import pandas as pd

from carla.evaluation import remove_nans
from carla.evaluation.api import Evaluation


class Redundancy(Evaluation):
    """
    Computes redundancy for each counterfactual
    """

    def __init__(self, mlmodel, hyperparameters):
        super().__init__(mlmodel, hyperparameters)
        self.cf_label = self.hyperparameters["cf_label"]
        self.columns = ["Redundancy"]

    def _compute_redundancy(
        self, factual: np.ndarray, counterfactual: np.ndarray
    ) -> int:
        redundancy = 0
        for col_idx in range(len(counterfactual)):
            # if feature is changed
            if factual[col_idx] != counterfactual[col_idx]:
                temp_cf = np.copy(counterfactual)
                temp_cf[col_idx] = factual[col_idx]
                # see if change is needed to flip the label
                temp_pred = np.argmax(
                    self.mlmodel.predict_proba(temp_cf.reshape((1, -1)))
                )
                if temp_pred == self.cf_label:
                    redundancy += 1
        return redundancy

    def _redundancy(
        self,
        factuals: pd.DataFrame,
        counterfactuals: pd.DataFrame,
    ) -> List[List[int]]:
        """
        Computes Redundancy measure for every counterfactual.

        Parameters
        ----------
        factuals:
            Encoded and normalized factual samples.
        counterfactuals:
            Encoded and normalized counterfactual samples.

        Returns
        -------
        List with redundancy values per counterfactual sample
        """
        df_enc_norm_fact = factuals.reset_index(drop=True)
        df_cfs = counterfactuals.reset_index(drop=True)

        df_cfs["redundancy"] = df_cfs.apply(
            lambda x: self._compute_redundancy(
                df_enc_norm_fact.iloc[x.name].values,
                x.values,
            ),
            axis=1,
        )
        return df_cfs["redundancy"].values.reshape((-1, 1)).tolist()

    def get_evaluation(self, counterfactuals, factuals):
        counterfactuals_without_nans, factual_without_nans = remove_nans(
            counterfactuals, factuals
        )

        if counterfactuals_without_nans.empty:
            redundancies = []
        else:
            redundancies = self._redundancy(
                factual_without_nans,
                counterfactuals_without_nans,
            )

        return pd.DataFrame(redundancies, columns=self.columns)
