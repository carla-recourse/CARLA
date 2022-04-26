import pandas as pd

from carla.models.api import MLModel
from carla.recourse_methods.api import RecourseMethod
from carla.recourse_methods.processing import (
    check_counterfactuals,
    encode_feature_names,
)

# have to download the pyjulia in order to using GeCo https://pyjulia.readthedocs.io/en/latest/installation.html
from julia.api import Julia
jl = Julia(compiled_modules=False)
from julia import Main
jl.eval('include("./carla/recourse_methods/catalog/geco/library/geco_carla.jl")')


class GeCo(RecourseMethod):
    """
    Implementation of Geco from Poyiadzi et.al. [1]_.

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

    """

    def __init__(self, mlmodel: MLModel, hyperparams=None) -> None:
        super().__init__(mlmodel)
        
        # set the parameters
        self._immutables = encode_feature_names(
            self._mlmodel.data.immutables, self._mlmodel.feature_input_order
        )

        Main.encoded_data = self._mlmodel.get_ordered_features(self._mlmodel.data.df)
        self.geco_input = jl.eval('pd_to_df(encoded_data)')
        Main.model = self._mlmodel
        Main.X = self.geco_input
        Main.immutables = self._immutables

    def get_counterfactuals(self, factuals: pd.DataFrame) -> pd.DataFrame:
        # Normalize and encode data so that we can use the classifier
        self.df_enc_norm_fact = self._mlmodel.get_ordered_features(factuals)
        Main.factuals = self.df_enc_norm_fact
        cfs = jl.eval('get_explanations(factuals, X, model, immutables)')
        
        return self._mlmodel.get_ordered_features(check_counterfactuals(self._mlmodel, cfs))
