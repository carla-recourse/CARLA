from typing import Any, Dict

import dice_ml
import pandas as pd

from carla.models.api import MLModel

from ...api import RecourseMethod


class Dice(RecourseMethod):
    """
    Implementation of CLUE from Mothilal et.al. [1]_.

    Parameters
    ----------
    mlmodel : carla.model.MLModel
        Black-Box-Model
    hyperparams : dict
        Dictionary containing hyperparameters. See notes below for its contents.

    Methods
    -------
    get_counterfactuals:
        Generate counterfactual examples for given factuals.
    encode_normalize_order_factuals:
        Uses encoder and scaler from black-box-model to preprocess data as needed.

    Notes
    -----
    - Hyperparams
        Hyperparameter contains important information for the recourse method to initialize.
        Please make sure to pass all values as dict with the following keys.

        * "num": int,
            Number of counterfactuals per factual to generate
        * "desired_class": int
            Given a binary class label, the desired class a counterfactual should have (e.g., 0 or 1)
    - Restrictions:
        *   Only the model agnostic approach (backend: sklearn) is used in our implementation.
        *   ML model needs to have a transformation pipeline for normalization, encoding and feature order.
            See pipelining at carla/models/catalog/catalog.py for an example ML model class implementation

    .. [1] R. K. Mothilal, Amit Sharma, and Chenhao Tan. 2020. Explaining machine learning classifiers
            through diverse counterfactual explanations
    """

    def __init__(self, mlmodel: MLModel, hyperparams: Dict[str, Any]) -> None:
        super().__init__(mlmodel)
        self._continous = mlmodel.data.continous
        self._categoricals = mlmodel.data.categoricals
        self._target = mlmodel.data.target
        # Prepare data for dice data structure
        self._dice_data = dice_ml.Data(
            dataframe=mlmodel.data.raw,
            continuous_features=self._continous,
            outcome_name=self._target,
        )

        self._dice_model = dice_ml.Model(model=mlmodel, backend="sklearn")

        self._dice = dice_ml.Dice(self._dice_data, self._dice_model, method="random")
        self._num = hyperparams["num"]
        self._desired_class = hyperparams["desired_class"]

        # Need scaler and encoder for get_counterfactual output
        self._scaler = mlmodel.scaler
        self._encoder = mlmodel.encoder
        self._feature_order = mlmodel.feature_input_order

    def get_counterfactuals(self, factuals: pd.DataFrame) -> pd.DataFrame:
        # Prepare factuals
        querry_instances = factuals.copy()

        # check if querry_instances are not empty
        if not querry_instances.shape[0] > 0:
            raise ValueError("Factuals should not be empty")

        # Generate counterfactuals
        dice_exp = self._dice.generate_counterfactuals(
            querry_instances, total_CFs=self._num, desired_class=self._desired_class
        )

        list_cfs = dice_exp.cf_examples_list
        df_cfs = pd.concat([cf.final_cfs_df for cf in list_cfs], ignore_index=True)
        df_cfs[self._continous] = self._scaler.transform(df_cfs[self._continous])
        encoded_features = self._encoder.get_feature_names(self._categoricals)
        df_cfs[encoded_features] = self._encoder.transform(df_cfs[self._categoricals])
        df_cfs = df_cfs[self._feature_order + [self._target]]
        # TODO: Expandable for further functionality

        return df_cfs
