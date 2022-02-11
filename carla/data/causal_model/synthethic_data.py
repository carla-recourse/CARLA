from typing import Callable, List

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split

from carla.data.api import Data


def _get_noise_string(node):
    if not node[0] == "x":
        raise ValueError
    return "u" + node[1:]


def _create_synthetic_data(scm, num_samples):
    """
    Generate synthetic data.

    Parameters
    ----------
    scm : CausalModel
        Structural causal model
    num_samples : int
        Number of samples in the dataset

    """

    exogenous_variables = np.concatenate(
        [
            np.array(scm.noise_distributions[node].sample(num_samples)).reshape(-1, 1)
            for node in scm.get_topological_ordering("exogenous")
        ],
        axis=1,
    )
    exogenous_variables = pd.DataFrame(
        exogenous_variables, columns=scm.get_topological_ordering("exogenous")
    )

    endogenous_variables = exogenous_variables.copy()
    endogenous_variables = endogenous_variables.rename(
        columns=dict(
            zip(
                scm.get_topological_ordering("exogenous"),
                scm.get_topological_ordering("endogenous"),
            )
        )
    )
    # used later to make sure parents are populated when computing children
    endogenous_variables.loc[:] = np.nan
    for node in scm.get_topological_ordering("endogenous"):
        parents = scm.get_parents(node)
        if endogenous_variables.loc[:, list(parents)].isnull().values.any():
            raise ValueError(
                "parents in endogenous_variables should already be occupied"
            )
        endogenous_variables[node] = scm.structural_equations_np[node](
            exogenous_variables[_get_noise_string(node)],
            *[endogenous_variables[p] for p in parents],
        )

    # fix a hyperplane
    w = np.ones((endogenous_variables.shape[1], 1))
    # get the average scale of (w^T)*X, this depends on the scale of the data
    scale = 2.5 / np.mean(np.abs(np.dot(endogenous_variables, w)))
    predictions = 1 / (1 + np.exp(-scale * np.dot(endogenous_variables, w)))

    if not 0.20 < np.std(predictions) < 0.42:
        raise ValueError(f"std of labels is strange: {np.std(predictions)}")

    # sample labels from class probabilities in predictions
    uniform_rv = np.random.rand(endogenous_variables.shape[0], 1)
    labels = uniform_rv < predictions
    labels = pd.DataFrame(data=labels, columns={"label"})

    df_endogenous = pd.concat([labels, endogenous_variables], axis=1).astype("float64")
    df_exogenous = pd.concat([exogenous_variables], axis=1).astype("float64")
    return df_endogenous, df_exogenous


class ScmDataset(Data):
    """
    Generate a dataset from structural equations

    Parameters
    ----------
    scm : CausalModel
        Structural causal model
    size : int
        Number of samples in the dataset
    """

    def __init__(
        self,
        scm,
        size: int,
        scaling_method: str = "MinMax",
        encoding_method: str = "OneHot",
    ):
        self.name = scm.scm_class
        self.scm = scm
        self._raw, self._noise = _create_synthetic_data(scm, num_samples=size)
        self._train_raw, self._test_raw = train_test_split(self._raw)

        # Fit scaler and encoder
        self.scaler: BaseEstimator = self.__fit_scaler(scaling_method)
        self.encoder: BaseEstimator = self.__fit_encoder(encoding_method)

        # Preparing pipeline components
        self._pipeline = self.__init_pipeline()
        self._inverse_pipeline = self.__init_inverse_pipeline()

        # Process the data
        self._processed: pd.DataFrame = self.perform_pipeline(self.raw)
        self._train_processed: pd.DataFrame = self.perform_pipeline(self.train_raw)
        self._test_processed: pd.DataFrame = self.perform_pipeline(self.test_raw)

    @property
    def categorical(self) -> List[str]:
        """
        Provides the column names of the categorical data.

        Returns
        -------
        List[str]
        """
        return self.scm._categorical

    @property
    def continuous(self) -> List[str]:
        """
        Provides the column names of the continuous data.

        Returns
        -------
        List[str]
        """
        return self.scm._continuous

    @property
    def categorical_noise(self) -> List[str]:
        """
        Provides the column names of the categorical data.

        Returns
        -------
        List[str]
        """
        return self.scm._categorical_noise

    @property
    def continuous_noise(self) -> List[str]:
        """
        Provides the column names of the continuous data.

        Returns
        -------
        List[str]
        """
        return self.scm._continuous_noise

    @property
    def immutables(self) -> List[str]:
        """
        Provides the column names of the immutable data.

        Returns
        -------
        List[str]
        """
        return self.scm._immutables

    @property
    def target(self) -> str:
        """
        Provies the name of the label column.

        Returns
        -------
        str
        """
        return "label"

    @property
    def raw(self) -> pd.DataFrame:
        return self._raw.copy()

    @property
    def noise(self) -> pd.DataFrame:
        return self._noise.copy()

    @property
    def train_raw(self) -> pd.DataFrame:
        return self._train_raw.copy()

    @property
    def test_raw(self) -> pd.DataFrame:
        return self._test_raw.copy()

    def processed(self, with_target=True) -> pd.DataFrame:
        df = self._processed.copy()
        if with_target:
            return df
        else:
            df = df[list(set(df.columns) - {self.target})]
            return df

    def train_processed(self, with_target=True) -> pd.DataFrame:
        df = self._train_processed.copy()
        if with_target:
            return df
        else:
            df = df[list(set(df.columns) - {self.target})]
            return df

    def test_processed(self, with_target=True) -> pd.DataFrame:
        df = self._test_processed.copy()
        if with_target:
            return df
        else:
            df = df[list(set(df.columns) - {self.target})]
            return df

    @property
    def scaler(self) -> BaseEstimator:
        """
        Contains a fitted sklearn scaler.

        Returns
        -------
        sklearn.preprocessing.BaseEstimator
        """
        return self._scaler

    @scaler.setter
    def scaler(self, scaler: BaseEstimator):
        """
        Sets a new fitted sklearn scaler.

        Parameters
        ----------
        scaler : sklearn.preprocessing.Scaler
            Fitted scaler for ML model.

        Returns
        -------
        sklearn.preprocessing.BaseEstimator
        """
        self._scaler = scaler

    @property
    def encoder(self) -> BaseEstimator:
        """
        Contains a fitted sklearn encoder:

        Returns
        -------
        sklearn.preprocessing.BaseEstimator
        """
        return self._encoder

    @encoder.setter
    def encoder(self, encoder: BaseEstimator):
        """
        Sets a new fitted sklearn encoder.

        Parameters
        ----------
        encoder: sklearn.preprocessing.Encoder
            Fitted encoder for ML model.
        """
        self._encoder = encoder

    def get_pipeline_element(self, key: str) -> Callable:
        """
        Returns a specific element of the pipeline

        Parameters
        ----------
        key : str
            Element of the pipeline we want to return

        Returns
        -------
        Pipeline element
        """
        key_idx = list(zip(*self._pipeline))[0].index(key)  # find key in pipeline
        return self._pipeline[key_idx][1]

    def perform_pipeline(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms input for prediction into correct form.
        Only possible for DataFrames without preprocessing steps.

        Recommended to use to keep correct encodings and normalization

        Parameters
        ----------
        df : pd.DataFrame
            Contains raw (unnormalized and not encoded) data.

        Returns
        -------
        output : pd.DataFrame
            Prediction input normalized and encoded

        """
        output = df.copy()

        for trans_name, trans_function in self._pipeline:
            output = trans_function(output)

        return output

    def perform_inverse_pipeline(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms output after prediction back into original form.
        Only possible for DataFrames with preprocessing steps.

        Parameters
        ----------
        df : pd.DataFrame
            Contains normalized and encoded data.

        Returns
        -------
        output : pd.DataFrame
            Prediction output denormalized and decoded

        """
        output = df.copy()

        for trans_name, trans_function in self._inverse_pipeline:
            output = trans_function(output)

        return output
