import pandas as pd
from julia import Main

# have to download the pyjulia in order to using GeCo https://pyjulia.readthedocs.io/en/latest/installation.html
from julia.api import Julia

from carla.models.api import MLModel
from carla.recourse_methods.api import RecourseMethod
from carla.recourse_methods.processing import (
    check_counterfactuals,
    encode_feature_names,
)
from carla.recourse_methods.processing.counterfactuals import merge_default_parameters

jl = Julia(compiled_modules=False)
jl.eval('include("./carla/recourse_methods/catalog/geco/library/geco_carla.jl")')


class GeCo(RecourseMethod):
    """
    GeCo from https://github.com/mjschleich/GeCo.jl [1]

    Parameters
    ----------
    mlmodel : carla.model.MLModel
        Black-Box-Model
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

        * "desired_class": int, optional
            The desired class from the classifier
        * "max_num_generations": int, optional
            The maximum number of generations for geco to run
        * "min_num_generations": int, optional
            The minimum number of generations for geco to run
        * "max_num_samples": int, optional
            The maximum number of samples
        * "norm_ratio": list, optional
            The norm ratio for the defined distance between instances

    .. [1] Maximilian Schleich, Zixuan Geng, Yihong Zhang, and Dan Suciu. 2021.
            GeCo: quality counterfactual explanations in real time. Proc. VLDB
            Endow. 14, 9 (May 2021), 1681-1693.
    """

    _DEFAULT_HYPERPARAMS = {
        "desired_class": 1,
        "max_num_generations": 100,
        "min_num_generations": 3,
        "max_num_samples": 5,
        "norm_ratio": [0.25, 0.25, 0.25, 0.25],
    }

    def __init__(self, mlmodel: MLModel, hyperparams=None) -> None:
        super().__init__(mlmodel)

        # set the parameterss
        self._immutables = encode_feature_names(
            self._mlmodel.data.immutables, self._mlmodel.feature_input_order
        )

        checked_hyperparams = merge_default_parameters(
            hyperparams, self._DEFAULT_HYPERPARAMS
        )

        Main.encoded_data = self._mlmodel.get_ordered_features(self._mlmodel.data.df)
        self.geco_input = jl.eval("pd_to_df(encoded_data)")
        Main.model = self._mlmodel
        Main.X = self.geco_input
        Main.immutables = self._immutables
        Main.desired_class = checked_hyperparams["desired_class"]
        Main.max_num_generations = checked_hyperparams["max_num_generations"]
        Main.min_num_generations = checked_hyperparams["min_num_generations"]
        Main.max_num_samples = checked_hyperparams["max_num_samples"]
        Main.norm_ratio = checked_hyperparams["norm_ratio"]

    def get_counterfactuals(self, factuals: pd.DataFrame) -> pd.DataFrame:
        self.df_enc_norm_fact = self._mlmodel.get_ordered_features(factuals)
        Main.factuals = self.df_enc_norm_fact
        cfs = jl.eval(
            "get_explanations(factuals, X, model, immutables, desired_class, max_num_generations, min_num_generations, max_num_samples, norm_ratio)"
        )

        return self._mlmodel.get_ordered_features(
            check_counterfactuals(self._mlmodel, cfs)
        )
