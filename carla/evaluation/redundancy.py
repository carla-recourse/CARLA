from typing import List

import numpy as np
import pandas as pd

from carla.evaluation.evaluation import Evaluation, remove_nans
from carla.models.api import MLModel


class Redundancy(Evaluation):
    def __init__(self, mlmodel, hyperparameters):
        super().__init__(mlmodel, hyperparameters)
        self.cf_label = self.hyperparameters["cf_label"]
        self.columns = ["Redundancy"]

    def compute_redundancy(
        self, fact: np.ndarray, cf: np.ndarray, mlmodel: MLModel, label_value: int
    ) -> int:
        red = 0
        for col_idx in range(cf.shape[0]):  # input array has one-dimensional shape
            if fact[col_idx] != cf[col_idx]:
                temp_cf = np.copy(cf)

                temp_cf[col_idx] = fact[col_idx]

                temp_pred = np.argmax(mlmodel.predict_proba(temp_cf.reshape((1, -1))))

                if temp_pred == label_value:
                    red += 1

        return red

    def redundancy(
        self,
        factuals: pd.DataFrame,
        counterfactuals: pd.DataFrame,
    ) -> List[List[int]]:
        """
        Computes Redundancy measure for every counterfactual

        Parameters
        ----------
        factuals: Encoded and normalized factual samples
        counterfactuals: Encoded and normalized counterfactual samples
        mlmodel: Black-box-model we want to discover
        cf_label: The target counterfactual label

        Returns
        -------
        List with redundancy values per counterfactual sample
        """

        df_enc_norm_fact = factuals.reset_index(drop=True)
        df_cfs = counterfactuals.reset_index(drop=True)

        df_cfs["redundancy"] = df_cfs.apply(
            lambda x: self.compute_redundancy(
                df_enc_norm_fact.iloc[x.name].values,
                x.values,
                self.mlmodel,
                self.cf_label,
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
            redundancies = self.redundancy(
                factual_without_nans,
                counterfactuals_without_nans,
            )

        return pd.DataFrame(redundancies, columns=self.columns)
