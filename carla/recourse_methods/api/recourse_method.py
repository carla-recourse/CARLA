from abc import ABC, abstractmethod

import pandas as pd

from carla.models.pipelining import encode, scale


class RecourseMethod(ABC):
    def __init__(self, mlmodel):
        self._mlmodel = mlmodel

    @abstractmethod
    def get_counterfactuals(self, factuals: pd.DataFrame):
        pass

    def encode_normalize_order_factuals(
        self, factuals: pd.DataFrame, with_target: bool = False
    ):
        # Prepare factuals
        querry_instances = factuals.copy()

        # check if querry_instances are not empty
        if querry_instances.shape[0] == 0:
            raise ValueError("Factuals should not be empty")

        factuals_enc_norm = scale(
            self._mlmodel.scaler, self._mlmodel.data.continous, querry_instances
        )
        factuals_enc_norm = encode(
            self._mlmodel.encoder, self._mlmodel.data.categoricals, factuals_enc_norm
        )

        label = [self._mlmodel.data.target] if with_target else []

        return factuals_enc_norm[self._mlmodel.feature_input_order + label]
