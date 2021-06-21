import pandas as pd

from carla.recourse_methods.api import RecourseMethod
from carla.recourse_methods.catalog.wachter.library import wachter_recourse
from carla.recourse_methods.processing import (
    check_counterfactuals,
    merge_default_parameters,
)


class Wachter(RecourseMethod):
    __DEFAULT_HYPERPARAMS = {
        "feature_cost": "_optional_",
        "lr": 0.01,
        "lambda_": 0.01,
        "n_iter": 1000,
        "t_max_min": 0.5,
        "norm": 1,
        "clamp": True,
        "loss_type": "MSE",
        "y_target": [0, 1],
        "binary_cat_features": True,
    }

    def __init__(self, mlmodel, hyperparams):
        """
        Initialisation of the Wachter recourse method.

        Restrictions
        ------------
        - Works currently only on Pytorch models
        - Only binary categorical features with and without one-hot-encoding

        Parameters
        ----------
        mlmodel: black-box-model we want to explore
        hyperparams: Parameter for Wachter method, with following possibilites
            {
                feature_cost: Optional[str]   List with costs per feature
                lr: float default: 0.01   learning rate for gradient descent
                lambda_: float default: 0.01  weight factor for feature_cost
                n_iter: int defaul: 1000 maximum number of iteration
                t_max_min: float default: 0.5   maximum time of search
                norm: int default: 2  L-norm to calculate cost
                clamp: bool defaul: True  If true, feature values will be clamped to (0, 1)
                loss_type: str default: 'MSE'  String for loss function (MSE or BCE)
                y_target List default: [0, 1]  List of one-hot-encoded target class
                binary_cat_features: bool default: True If true, the encoding of x is done by drop_if_binary
            }
        """
        super().__init__(mlmodel)

        checked_hyperparams = merge_default_parameters(
            hyperparams, self.__DEFAULT_HYPERPARAMS
        )
        self._feature_costs = checked_hyperparams["feature_cost"]
        self._lr = checked_hyperparams["lr"]
        self._lambda = checked_hyperparams["lambda_"]
        self._n_iter = checked_hyperparams["n_iter"]
        self._t_max_min = checked_hyperparams["t_max_min"]
        self._norm = checked_hyperparams["norm"]
        self._clamp = checked_hyperparams["clamp"]
        self._loss_type = checked_hyperparams["loss_type"]
        self._y_target = checked_hyperparams["y_target"]
        self._binary_cat_features = checked_hyperparams["binary_cat_features"]

    def get_counterfactuals(self, factuals: pd.DataFrame) -> pd.DataFrame:
        # Normalize and encode data
        df_enc_norm_fact = self.encode_normalize_order_factuals(factuals)

        encoded_feature_names = self._mlmodel.encoder.get_feature_names(
            self._mlmodel.data.categoricals
        )
        cat_features_indices = [
            df_enc_norm_fact.columns.get_loc(feature)
            for feature in encoded_feature_names
        ]

        df_cfs = df_enc_norm_fact.apply(
            lambda x: wachter_recourse(
                self._mlmodel.raw_model,
                x.reshape((1, -1)),
                cat_features_indices,
                binary_cat_features=self._binary_cat_features,
                feature_costs=self._feature_costs,
                lr=self._lr,
                lambda_=self._lambda,
                n_iter=self._n_iter,
                t_max_min=self._t_max_min,
                norm=self._norm,
                clamp=self._clamp,
                loss_type=self._loss_type,
            ),
            raw=True,
            axis=1,
        )

        df_cfs = check_counterfactuals(self._mlmodel, df_cfs)

        return df_cfs
