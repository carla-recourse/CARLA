"""
code adapted from:
https://github.com/upura/featureTweakPy
"""

import copy
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import sklearn
import xgboost
import xgboost.core

from carla.models.catalog import MLModelCatalog
from carla.models.catalog.parse_xgboost import parse_booster
from carla.recourse_methods.api import RecourseMethod
from carla.recourse_methods.processing import (
    check_counterfactuals,
    merge_default_parameters,
)


def _L1_cost_func(a, b):
    """The 1-norm ||a-b||_1"""
    return np.linalg.norm(a - b, ord=1)


def _L2_cost_func(a, b):
    """The 2-norm ||a-b||_2"""
    return np.linalg.norm(a - b, ord=2)


def search_path(tree, class_labels):
    """
    return path index list containing [{leaf node id, inequality symbol, threshold, feature index}].

    Parameters
    ----------
    tree: sklearn.tree.DecisionTreeClassifier or xgboost.core.Booster
        The classification tree.
    class_labels:
        All the possible class labels.

    Returns
    -------
    path_info
    """

    def parse_tree(tree):
        """

        Parameters
        ----------
        tree: sklearn.tree.DecisionTreeClassifier or xgboost.core.Booster
            The classification tree we want to parse.

        Returns
        -------
        children_left: array of int
            children_left[i] holds the node id of the left child of node i.
            For leaves, children_left[i] == TREE_LEAF.

        children_right: array of int
            children_right[i] holds the node id of the right child of node i.
            For leaves, children_right[i] == TREE_LEAF.

        threshold: array of double
            threshold[i] holds the threshold for the internal node i.

        feature: array of int
            feature[i] holds the feature to split on, for the internal node i.

        leaf_nodes: array of int
            leaf nodes with outcome aim label

        """
        if isinstance(tree, sklearn.tree.DecisionTreeClassifier):
            children_left = tree.tree_.children_left
            children_right = tree.tree_.children_right
            feature = tree.tree_.feature
            threshold = tree.tree_.threshold
            values = tree.tree_.value

            # leaf nodes ID
            leaf_nodes = np.where(children_left == -1)[0]

            # outcomes of leaf nodes
            leaf_values = values[leaf_nodes].reshape(len(leaf_nodes), len(class_labels))
            leaf_classes = np.argmax(leaf_values, axis=-1)

            """
            We want to find the leaf_nodes where the class is equal to the counterfactual label 1.
            In the original code the line was as follows:

            leaf_nodes = np.where(leaf_values[:, cf_label] != 0)[0]

            However this seems wrong as we want to index the leaf_nodes with the above expression.
            This also caused that sometimes 0 would be in the leaf_nodes, but as 0 is the root node this
            should not happen.
            """
            # select the leaf nodes whose outcome is aim_label
            leaf_nodes = leaf_nodes[np.where(leaf_classes != 0)[0]]

            return children_left, children_right, feature, threshold, leaf_nodes
        elif isinstance(tree, xgboost.core.Booster):
            children_left, children_right, threshold, feature, scores = parse_booster(
                tree
            )
            # leaf nodes ID
            leaf_nodes = np.where(children_left == -1)[0]

            # TODO threshold of 0.5 because of logistic function, doesn't work for other xgboost objective functions
            # outcome of leaf nodes
            leaf_classes = scores[leaf_nodes] > 0.5
            leaf_nodes = leaf_nodes[np.where(leaf_classes != 0)[0]]

            return children_left, children_right, feature, threshold, leaf_nodes
        else:
            raise ValueError("tree is not of a supported Class")

    """ select leaf nodes whose outcome is aim_label """
    children_left, children_right, feature, threshold, leaf_nodes = parse_tree(tree)

    """ search the path to the selected leaf node """
    paths = {}
    for leaf_node in leaf_nodes:
        """correspond leaf node to left and right parents"""
        child_node = leaf_node
        parent_node = -100  # initialize
        parents_left = []
        parents_right = []
        while parent_node != 0:
            if np.where(children_left == child_node)[0].shape == (0,):
                parent_left = -1
                parent_right = np.where(children_right == child_node)[0][0]
                parent_node = parent_right
            elif np.where(children_right == child_node)[0].shape == (0,):
                parent_right = -1
                parent_left = np.where(children_left == child_node)[0][0]
                parent_node = parent_left
            parents_left.append(parent_left)
            parents_right.append(parent_right)
            """ for next step """
            child_node = parent_node

        # nodes dictionary containing left parents and right parents
        paths[leaf_node] = (parents_left, parents_right)

    path_info = get_path_info(paths, threshold, feature)
    return path_info


def get_path_info(paths, threshold, feature):
    """
    Extract the path info from the parameters

    Parameters
    ----------
    paths:
        Paths trough the tree from root to leaves.

    threshold: array of double
        threshold[i] holds the threshold for the internal node i.

    feature: array of int
        feature[i] holds the feature to split on, for the internal node i.

    Returns
    -------
    dictionary where dict[i] contains node_id, inequality_symbol, threshold, and feature
    """
    path_info = {}
    for i in paths:
        node_ids = []  # node ids used in the current node
        inequality_symbols = []  # inequality symbols used in the current node
        thresholds = []  # thresholds used in the current node
        features = []  # features used in the current node
        parents_left, parents_right = paths[i]

        for idx in range(len(parents_left)):

            def do_appends(node_id):
                """helper function to reduce duplicate code"""
                node_ids.append(node_id)
                thresholds.append(threshold[node_id])
                features.append(feature[node_id])

            if parents_left[idx] != -1:
                """the child node is the left child of the parent"""
                node_id = parents_left[idx]  # node id
                inequality_symbols.append(0)
                do_appends(node_id)
            elif parents_right[idx] != -1:
                """the child node is the right child of the parent"""
                node_id = parents_right[idx]
                inequality_symbols.append(1)
                do_appends(node_id)

            path_info[i] = {
                "node_id": node_ids,
                "inequality_symbol": inequality_symbols,
                "threshold": thresholds,
                "feature": features,
            }
    return path_info


class FeatureTweak(RecourseMethod):
    """
    Implementation of FeatureTweak [1]_.

    Parameters
    ----------
    mlmodel: MLModelCatalog
        Black-Box-Model
    hyperparams : dict
        Dictionary containing hyperparameters. See notes below for its contents.

    Methods
    -------
    get_counterfactuals:
        Generate counterfactual examples for given factuals.
    esatisfactory_instance:
        Return the epsilon satisfactory instance of x.
    feature_tweaking:
        Generate a single counterfactual by FeatureTweaking.

    Notes
    -----
    - Hyperparams
        Hyperparameter contains important information for the recourse method to initialize.
        Please make sure to pass all values as dict with the following keys.

        * "eps": float

    .. [1] Tolomei, G., Silvestri, F., Haines, A., & Lalmas, M. (2017, August). Interpretable predictions of tree-based
            ensembles via actionable feature tweaking. In Proceedings of the 23rd ACM SIGKDD international conference on
            knowledge discovery and data mining (pp. 465-474).
    """

    _DEFAULT_HYPERPARAMS = {"eps": 0.1}

    def __init__(
        self,
        mlmodel: MLModelCatalog,
        hyperparams: Optional[Dict] = None,
        cost_func=_L2_cost_func,
    ):
        supported_backends = ["sklearn", "xgboost"]
        if mlmodel.backend not in supported_backends:
            raise ValueError(
                f"{mlmodel.backend} is not in supported backends {supported_backends}"
            )

        super().__init__(mlmodel)

        checked_hyperparams = merge_default_parameters(
            hyperparams, self._DEFAULT_HYPERPARAMS
        )

        self.model = mlmodel
        self.data = mlmodel.data
        self.eps = checked_hyperparams["eps"]
        self.target_col = self.data.target
        self.cost_func = cost_func

    def esatisfactory_instance(self, x: np.ndarray, path_info):
        """
        return the epsilon satisfactory instance of x.

        Parameters
        ----------
        x: np.ndarray
            A single factual example.
        path_info:
            One path from the result of search_path(tree, class_labels, cf_label)

        Returns
        -------
        epsilon satisfactory instance
        """
        esatisfactory = copy.deepcopy(x)
        for i in range(len(path_info["feature"])):
            feature_idx = path_info["feature"][i]  # feature index

            if isinstance(feature_idx, str):
                feature_idx = np.where(
                    np.array(self.model.feature_input_order) == feature_idx
                )

            threshold_value = path_info["threshold"][i]  # threshold in current node
            inequality_symbol = path_info["inequality_symbol"][i]  # inequality symbol
            if inequality_symbol == 0:
                esatisfactory[feature_idx] = threshold_value - self.eps
            elif inequality_symbol == 1:
                esatisfactory[feature_idx] = threshold_value + self.eps
            else:
                print("something wrong")
        return esatisfactory

    def feature_tweaking(self, x: np.ndarray, class_labels: List[int], cf_label: int):
        """
        Perform feature tweaking on a single factual example.

        Parameters
        ----------
        x: np.ndarray
            A single factual example.
        class_labels:  List[int]
            List of possible class labels.
        cf_label: int
            What label the counterfactual should have.

        Returns
        -------
        counterfactual example
        """

        def predict(classifier, x):
            if isinstance(
                classifier,
                (sklearn.tree.DecisionTreeClassifier, MLModelCatalog),
            ):
                # need to reshape x as it's not a batch
                return classifier.predict(x.reshape(1, -1))
            elif isinstance(classifier, xgboost.core.Booster):
                # TODO is this threshold always correct? E.g. does it depend on num_classes?
                threshold = 0.5
                # need to convert Numpy array to DMatrix
                return (
                    classifier.predict(
                        xgboost.DMatrix(
                            x.reshape(1, -1),
                            feature_names=self.model.feature_input_order,
                        )
                    )
                    > threshold
                )
            raise ValueError("tree is not of a supported Class")

        x_out = copy.deepcopy(x)  # initialize output
        delta_mini = 10**3  # initialize cost

        for tree in self.model.tree_iterator:  # loop over individual trees

            estimator_prediction = predict(tree, x)
            if (
                predict(self.model, x) == estimator_prediction
                and estimator_prediction != cf_label
            ):
                paths_info = search_path(tree, class_labels)
                for key in paths_info:
                    """generate epsilon-satisfactory instance"""
                    path_info = paths_info[key]
                    es_instance = self.esatisfactory_instance(x, path_info)
                    if (
                        predict(tree, es_instance) == cf_label
                        and self.cost_func(x, es_instance) < delta_mini
                    ):
                        x_out = es_instance
                        delta_mini = self.cost_func(x, es_instance)
                else:
                    continue
        return x_out

    def get_counterfactuals(self, factuals: pd.DataFrame):

        # drop targets
        instances = factuals.copy()
        instances = instances.reset_index(drop=True)

        # only works for continuous data
        instances = self.model.get_ordered_features(instances)

        class_labels = [0, 1]

        counterfactuals = []
        for i, row in instances.iterrows():
            cf_label = 1  # flipped target label
            counterfactual = self.feature_tweaking(
                row.to_numpy(), class_labels, cf_label
            )
            counterfactuals.append(counterfactual)

        counterfactuals_df = check_counterfactuals(
            self._mlmodel, counterfactuals, factuals.index
        )
        counterfactuals_df = self._mlmodel.get_ordered_features(counterfactuals_df)
        return counterfactuals_df
