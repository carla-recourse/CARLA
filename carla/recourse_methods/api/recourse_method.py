from abc import ABC, abstractmethod

import pandas as pd

from carla.models.pipelining import encode, scale


class RecourseMethod(ABC):
    """
    Abstract class to implement custom recourse methods for a given black-box-model.

    Parameters
    ----------
    mlmodel: carla.models.MLModel
        Black-box-classifier we want to discover.

    Methods
    -------
    get_counterfactuals:
        Generate counterfactual examples for given factuals.
    encode_normalize_order_factuals:
        Uses encoder and scaler from black-box-model to preprocess data as needed.

    Returns
    -------
    None
    """

    def __init__(self, mlmodel):
        self._mlmodel = mlmodel

    @abstractmethod
    def get_counterfactuals(self, factuals: pd.DataFrame):
        """
        Generate counterfactual examples for given factuals.

        Parameters
        ----------
        factuals: pd.DataFrame
            Not encoded and not normalised factual examples in two-dimensional shape (m, n).

        Returns
        -------
        pd.DataFrame
            Encoded and normalised counterfactual examples.
        """
        pass

    def encode_normalize_order_factuals(
        self, factuals: pd.DataFrame, with_target: bool = False
    ):
        """
        Uses encoder and scaler from black-box-model to preprocess data as needed.

        Parameters
        ----------
        factuals: pd.DataFrame
            Not encoded and not normalised factual examples in two-dimensional shape (m, n).
        with_target: bool, default: False
            If True, the output DataFrame contains label information

        Returns
        -------
        pd.DataFrame
            Encoded, normalized and ordered factual examples
        """
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
