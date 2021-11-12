"""
Tolomei, G., Silvestri, F., Haines, A., & Lalmas, M. (2017, August).
Interpretable predictions of tree-based ensembles via actionable feature tweaking.
In Proceedings of the 23rd ACM SIGKDD international conference on knowledge discovery and data mining (pp. 465-474).

code from:
https://github.com/upura/featureTweakPy
"""

import copy
from typing import Dict, List, Union

import numpy as np
import pandas as pd
import sklearn
import xgboost
import xgboost.core

from carla.data.api import Data
from carla.recourse_methods.api import RecourseMethod
from carla.recourse_methods.catalog.focus.parse_xgboost import parse_booster
from carla.recourse_methods.catalog.focus.tree_model import ForestModel, XGBoostModel
from carla.recourse_methods.processing import check_counterfactuals


def L1_cost_func(a, b):
    """ ||a-b||_1 """
    return np.linalg.norm(a - b, ord=1)


def L2_cost_func(a, b):
    """ ||a-b||_2 """
    return np.linalg.norm(a - b, ord=2)


def search_path(tree, class_labels, cf_label):
    """
    return path index list containing [{leaf node id, inequality symbol, threshold, feature index}].
    estimator: decision tree
    maxj: the number of selected leaf nodes
    """
    """ select leaf nodes whose outcome is aim_label """

    def parse_tree(tree):
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
            # select the leaf nodes whose outcome is aim_label

            """
            In the original code the line was as follows:

            leaf_nodes = np.where(leaf_values[:, cf_label] != 0)[0]

            However this seems wrong as we want to index the leaf_nodes with the above expression.
            This also caused that sometimes 0 would be in the leaf_nodes, but as 0 is the root node this
            should not happen.
            """
            leaf_nodes = leaf_nodes[np.where(leaf_classes != 0)[0]]

            return children_left, children_right, feature, threshold, leaf_nodes
        elif isinstance(tree, xgboost.core.Booster):
            children_left, children_right, threshold, feature, classes = parse_booster(
                tree
            )
            # leaf nodes ID
            leaf_nodes = np.where(children_left == -1)[0]
            leaf_nodes = np.where(classes[leaf_nodes] != 0)[0]

            return children_left, children_right, feature, threshold, leaf_nodes

        raise ValueError("tree is not of a supported Class")

    children_left, children_right, feature, threshold, leaf_nodes = parse_tree(tree)

    """ search the path to the selected leaf node """
    paths = {}
    for leaf_node in leaf_nodes:
        """ correspond leaf node to left and right parents """
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
    path_info = {}
    for i in paths:
        node_ids = []  # node ids used in the current node
        inequality_symbols = []  # inequality symbols used in the current node
        thresholds = []  # thresholds used in the current node
        features = []  # features used in the current node
        parents_left, parents_right = paths[i]

        for idx in range(len(parents_left)):

            def do_appends(node_id):
                """ helper function to reduce duplicate code"""
                node_ids.append(node_id)
                thresholds.append(threshold[node_id])
                features.append(feature[node_id])

            if parents_left[idx] != -1:
                """ the child node is the left child of the parent """
                node_id = parents_left[idx]  # node id
                inequality_symbols.append(0)
                do_appends(node_id)
            elif parents_right[idx] != -1:
                """ the child node is the right child of the parent """
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
    # TODO can this method actually work for a single TreeModel?
    def __init__(
        self,
        mlmodel: Union[ForestModel, XGBoostModel],
        data: Data,
        hyperparams: Dict,
        cost_func=L2_cost_func,
    ):

        super().__init__(mlmodel)

        self.model = mlmodel
        self.data = data
        self.eps = hyperparams["eps"]
        self.target_col = data.target
        self.cost_func = cost_func

    def esatisfactory_instance(self, x, path_info):
        """
        return the epsilon satisfactory instance of x.
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

        Parameters
        ----------
        x: a single factual example
        class_labels: list of possible class labels
        cf_label: what label the counterfactual should have

        Returns
        -------
        counterfactual example
        """

        def predict(classifier, x):
            if isinstance(
                classifier,
                (sklearn.tree.DecisionTreeClassifier, ForestModel, XGBoostModel),
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
        delta_mini = 10 ** 3  # initialize cost

        for tree in self.model.tree_iterator:  # loop over individual trees

            estimator_prediction = predict(tree, x)
            if (
                predict(self.model, x) == estimator_prediction
                and estimator_prediction != cf_label
            ):
                paths_info = search_path(tree, class_labels, cf_label)
                for key in paths_info:
                    """ generate epsilon-satisfactory instance """
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

        # normalize and one-hot-encoding
        instances = self.encode_normalize_order_factuals(instances, with_target=False)
        instances = instances[self.data.continous]

        # y = factuals[self.target_col]
        # y = self.model.predict(instances)
        class_labels = [0, 1]

        counterfactuals = []
        for i, row in instances.iterrows():
            cf_label = 1  # flipped target label
            counterfactual = self.feature_tweaking(
                row.to_numpy(), class_labels, cf_label
            )
            counterfactuals.append(counterfactual)

        counterfactuals_df = check_counterfactuals(self._mlmodel, counterfactuals)
        return counterfactuals_df
