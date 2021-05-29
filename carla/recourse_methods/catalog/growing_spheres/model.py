import pandas as pd

from carla.models.api import MLModel
from carla.recourse_methods.api import RecourseMethod
from carla.recourse_methods.catalog.growing_spheres.library import (
    growing_spheres_search,
)
from carla.recourse_methods.processing import (
    check_counterfactuals,
    encode_feature_names,
)


class GrowingSpheres(RecourseMethod):
    def __init__(self, mlmodel: MLModel, hyperparams=None) -> None:
        """
        Implementation follows the Random Point Picking over a sphere
        The algorithm's implementation follows: Pawelczyk, Broelemann & Kascneci (2020);

        Restrictions
        ------------
        - Growing Spheres works at the moment only for data with dropped first column of binary categorical features.

        Parameters
        ----------
        mlmodel: Black-box-model we want to discover

        """
        super().__init__(mlmodel)

        self._immutables = encode_feature_names(
            self._mlmodel.data.immutables, self._mlmodel.feature_input_order
        )
        self._mutables = [
            feature
            for feature in self._mlmodel.feature_input_order
            if feature not in self._immutables
        ]
        self._continuous = self._mlmodel.data.continous
        self._categoricals_enc = encode_feature_names(
            self._mlmodel.data.categoricals, self._mlmodel.feature_input_order
        )

    def get_counterfactuals(self, factuals: pd.DataFrame) -> pd.DataFrame:
        # Normalize and encode data
        df_enc_norm_fact = self.encode_normalize_order_factuals(factuals)

        list_cfs = []
        for index, row in df_enc_norm_fact.iterrows():
            counterfactual = growing_spheres_search(
                row,
                self._mutables,
                self._immutables,
                self._continuous,
                self._categoricals_enc,
                self._mlmodel.feature_input_order,
                self._mlmodel,
            )
            list_cfs.append(counterfactual)

        df_cfs = check_counterfactuals(self._mlmodel, list_cfs)

        return df_cfs
