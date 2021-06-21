from typing import Any, Dict

import dice_ml
import pandas as pd

from carla.models.api import MLModel

from ...api import RecourseMethod
from ...processing import merge_default_parameters


class Dice(RecourseMethod):
    _DEFAULT_HYPERPARAMS = {"num": 1, "desired_class": 1, "posthoc_sparsity_param": 0.1}

    def __init__(self, mlmodel: MLModel, hyperparams: Dict[str, Any]) -> None:
        """
        Constructor for Dice model
        Implementation can be seen at https://github.com/interpretml/DiCE

        Restrictions:
        ------------
        -   Only the model agnostic approach (backend: sklearn) is used in our implementation.
        -   ML model needs to have a transformation pipeline for normalization, encoding and feature order.
            See pipelining at carla/models/catalog/catalog.py for an example ML model class implementation

        Parameters
        ----------
        mlmodel : models.api.MLModel
            ML model to build counterfactuals for.
        hyperparams : dict
            Hyperparameter which are needed for DICE to generate counterfactuals.
            Structure: {
                "num": int,
                "desired_class": int,
                "posthoc_sparsity_param": float, default: 0.1
                    Parameter for the post-hoc operation on continuous features to enhance sparsity.
            }
        """
        super().__init__(mlmodel)
        self._continous = mlmodel.data.continous
        self._categoricals = mlmodel.data.categoricals
        self._target = mlmodel.data.target

        checked_hyperparams = merge_default_parameters(
            hyperparams, self._DEFAULT_HYPERPARAMS
        )
        # Prepare data for dice data structure
        self._dice_data = dice_ml.Data(
            dataframe=mlmodel.data.raw,
            continuous_features=self._continous,
            outcome_name=self._target,
        )

        self._dice_model = dice_ml.Model(model=mlmodel, backend="sklearn")

        self._dice = dice_ml.Dice(self._dice_data, self._dice_model, method="random")
        self._num = checked_hyperparams["num"]
        self._desired_class = checked_hyperparams["desired_class"]
        self._post_hoc_sparsity_param = checked_hyperparams["posthoc_sparsity_param"]

        # Need scaler and encoder for get_counterfactual output
        self._scaler = mlmodel.scaler
        self._encoder = mlmodel.encoder
        self._feature_order = mlmodel.feature_input_order

    @property
    def dice_model(self):
        return self._dice

    def get_counterfactuals(self, factuals: pd.DataFrame) -> pd.DataFrame:
        """
        Compute a certain number of counterfactuals per factual example.


        Parameters
        ----------
        factuals : pd.DataFrame
            DataFrame containing all samples for which we want to generate counterfactual examples.
            All instances should belong to the same class.

        Returns
        -------
        df_cfs : pd.DataFrame
            Encoded and normalized counterfactuals

        """

        # Prepare factuals
        querry_instances = factuals.copy()

        # check if querry_instances are not empty
        if not querry_instances.shape[0] > 0:
            raise ValueError("Factuals should not be empty")

        # Generate counterfactuals
        dice_exp = self._dice.generate_counterfactuals(
            querry_instances,
            total_CFs=self._num,
            desired_class=self._desired_class,
            posthoc_sparsity_param=self._post_hoc_sparsity_param,
        )

        list_cfs = dice_exp.cf_examples_list
        df_cfs = pd.concat([cf.final_cfs_df for cf in list_cfs], ignore_index=True)
        df_cfs[self._continous] = self._scaler.transform(df_cfs[self._continous])
        encoded_features = self._encoder.get_feature_names(self._categoricals)
        df_cfs[encoded_features] = self._encoder.transform(df_cfs[self._categoricals])
        df_cfs = df_cfs[self._feature_order + [self._target]]
        # TODO: Expandable for further functionality

        return df_cfs
