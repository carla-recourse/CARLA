from typing import Dict, Optional

import dice_ml
import pandas as pd

from carla.models.api import MLModel
from carla.recourse_methods.processing import check_counterfactuals

from ...api import RecourseMethod
from ...processing import merge_default_parameters


class Dice(RecourseMethod):
    """
    Implementation of Dice from Mothilal et.al. [1]_.

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

    Notes
    -----
    - Hyperparams
        Hyperparameter contains important information for the recourse method to initialize.
        Please make sure to pass all values as dict with the following keys.

        * "num": int, default: 1
            Number of counterfactuals per factual to generate
        * "desired_class": int, default: 1
            Given a binary class label, the desired class a counterfactual should have (e.g., 0 or 1)
        * "posthoc_sparsity_param": float, default: 0.1
            Fraction of post-hoc preprocessing steps.
    - Restrictions:
        *   Only the model agnostic approach (backend: sklearn) is used in our implementation.
        *   ML model needs to have a transformation pipeline for normalization, encoding and feature order.
            See pipelining at carla/models/catalog/catalog.py for an example ML model class implementation

    .. [1] R. K. Mothilal, Amit Sharma, and Chenhao Tan. 2020. Explaining machine learning classifiers
            through diverse counterfactual explanations
    """

    _DEFAULT_HYPERPARAMS = {"num": 1, "desired_class": 1, "posthoc_sparsity_param": 0.1}

    def __init__(self, mlmodel: MLModel, hyperparams: Optional[Dict] = None) -> None:

        supported_backends = ["tensorflow", "pytorch"]
        if mlmodel.backend not in supported_backends:
            raise ValueError(
                f"{mlmodel.backend} is not in supported backends {supported_backends}"
            )

        super().__init__(mlmodel)
        self._continuous = mlmodel.data.continuous
        self._categorical = mlmodel.data.categorical
        self._target = mlmodel.data.target
        self._model = mlmodel

        checked_hyperparams = merge_default_parameters(
            hyperparams, self._DEFAULT_HYPERPARAMS
        )
        # Prepare data for dice data structure
        self._dice_data = dice_ml.Data(
            dataframe=mlmodel.data.df,
            continuous_features=self._continuous,
            outcome_name=self._target,
        )

        self._dice_model = dice_ml.Model(model=mlmodel, backend="sklearn")

        self._dice = dice_ml.Dice(self._dice_data, self._dice_model, method="random")
        self._num = checked_hyperparams["num"]
        self._desired_class = checked_hyperparams["desired_class"]
        self._post_hoc_sparsity_param = checked_hyperparams["posthoc_sparsity_param"]

    def get_counterfactuals(self, factuals: pd.DataFrame) -> pd.DataFrame:
        # Prepare factuals
        querry_instances = factuals.copy()
        querry_instances = self._model.get_ordered_features(querry_instances)

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
        df_cfs = check_counterfactuals(self._mlmodel, df_cfs, factuals.index)
        df_cfs = self._mlmodel.get_ordered_features(df_cfs)
        return df_cfs
