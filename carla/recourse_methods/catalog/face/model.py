from typing import Any, Dict

import pandas as pd

from carla.models.api import MLModel
from carla.recourse_methods.api import RecourseMethod
from carla.recourse_methods.catalog.face.library import graph_search
from carla.recourse_methods.processing import (
    check_counterfactuals,
    encode_feature_names,
    merge_default_parameters,
)


class Face(RecourseMethod):
    """
    Implementation of FACE from Poyiadzi et.al. [1]_.

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

        * "mode": {"knn", "epsilon"},
            Decides over type of FACE
        * "fraction": float [0 < x < 1]
            determines fraction of data set to be used to construct neighbourhood graph

    .. [1] Rafael Poyiadzi, Kacper Sokol, Raul Santos-Rodriguez, Tijl De Bie, and Peter Flach. 2020. In Proceedings
            of the AAAI/ACM Conference on AI, Ethics, and Society (AIES)
    """

    _DEFAULT_HYPERPARAMS = {"mode": None, "fraction": 0.1}

    def __init__(self, mlmodel: MLModel, hyperparams: Dict[str, Any]) -> None:
        super().__init__(mlmodel)

        checked_hyperparams = merge_default_parameters(
            hyperparams, self._DEFAULT_HYPERPARAMS
        )
        self.mode = checked_hyperparams["mode"]
        self.fraction = checked_hyperparams["fraction"]

        self._immutables = encode_feature_names(
            self._mlmodel.data.immutables, self._mlmodel.feature_input_order
        )

    @property
    def fraction(self) -> float:
        """
        Controls the fraction of the used dataset to build graph on.

        Returns
        -------
        float
        """
        return self._fraction

    @fraction.setter
    def fraction(self, x: float) -> None:
        if 0 < x < 1:
            self._fraction = x
        else:
            raise ValueError("Fraction has to be between 0 and 1")

    @property
    def mode(self) -> str:
        """
        Sets and changes the type of FACE. {"knn", "epsilon"}

        Returns
        -------
        str
        """
        return self._mode

    @mode.setter
    def mode(self, mode: str) -> None:
        if mode in ["knn", "epsilon"]:
            self._mode = mode
        else:
            raise ValueError("Mode has to be either knn or epsilon")

    def get_counterfactuals(self, factuals: pd.DataFrame) -> pd.DataFrame:
        # >drop< factuals from dataset to prevent duplicates,
        # >reorder< and >add< factuals to top; necessary in order to use the index
        df = self._mlmodel.data.df.copy()
        cond = df.isin(factuals).values
        df = df.drop(df[cond].index)
        df = pd.concat([factuals, df], ignore_index=True)

        df = self._mlmodel.get_ordered_features(df)
        factuals = self._mlmodel.get_ordered_features(factuals)

        list_cfs = []
        for i in range(factuals.shape[0]):
            cf = graph_search(
                df,
                i,
                self._immutables,
                self._mlmodel,
                mode=self._mode,
                frac=self._fraction,
            )
            list_cfs.append(cf)

        df_cfs = check_counterfactuals(self._mlmodel, list_cfs)
        df_cfs = self._mlmodel.get_ordered_features(df_cfs)
        return df_cfs
