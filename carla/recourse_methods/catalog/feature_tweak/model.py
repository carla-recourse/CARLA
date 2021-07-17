import copy
from typing import Dict

import numpy as np
import pandas as pd

from carla.data.api import Data
from carla.models.api import MLModel
from carla.recourse_methods.api import RecourseMethod
from carla.recourse_methods.processing import check_counterfactuals


def cost_func(a, b):
    return np.linalg.norm(a - b)


def search_path(estimator, class_labels, cf_label):
    """
    return path index list containing [{leaf node id, inequality symbol, threshold, feature index}].
    estimator: decision tree
    maxj: the number of selected leaf nodes
    """
    """ select leaf nodes whose outcome is aim_label """
    children_left = estimator.tree_.children_left  # information of left child node
    children_right = estimator.tree_.children_right
    feature = estimator.tree_.feature
    threshold = estimator.tree_.threshold

    # leaf nodes ID
    leaf_nodes = np.where(children_left == -1)[0]
    # outcomes of leaf nodes
    leaf_values = estimator.tree_.value[leaf_nodes].reshape(
        len(leaf_nodes), len(class_labels)
    )
    # select the leaf nodes whose outcome is aim_label
    leaf_nodes = np.where(leaf_values[:, cf_label] != 0)[0]

    # if the following is not done it creates an error. Also 0 should not be a leaf node, it is not part of the
    # original leaf_nodes ID.
    # TODO figure out why there even is a zero in there in the first place, it's not supposed to be there
    if 0 in leaf_nodes:
        leaf_nodes = np.delete(leaf_nodes, 0)

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
    def __init__(self, mlmodel: MLModel, data: Data, hyperparams: Dict):

        super().__init__(mlmodel)

        self.model = mlmodel
        self.data = data
        self.eps = hyperparams["eps"]
        self.target_col = data.target

    def esatisfactory_instance(self, x, path_info):
        """
        return the epsilon satisfactory instance of x.
        """
        esatisfactory = copy.deepcopy(x)
        for i in range(len(path_info["feature"])):
            feature_idx = path_info["feature"][i]  # feature index
            threshold_value = path_info["threshold"][i]  # threshold in current node
            inequality_symbol = path_info["inequality_symbol"][i]  # inequality symbol
            if inequality_symbol == 0:
                esatisfactory[feature_idx] = threshold_value - self.eps
            elif inequality_symbol == 1:
                esatisfactory[feature_idx] = threshold_value + self.eps
            else:
                print("something wrong")
        return esatisfactory

    def feature_tweaking(self, x, class_labels, cf_label):
        def predict(model, x):
            # need to reshape x as it's not a batch
            return model.predict(x.reshape(1, -1))

        x_out = copy.deepcopy(x)  # initialize output
        delta_mini = 10 ** 3  # initialize cost
        for estimator in self.model.raw_model:  # loop over individual trees

            estimator_prediction = predict(estimator, x)
            if (
                predict(self.model, x) == estimator_prediction
                and estimator_prediction != cf_label
            ):
                paths_info = search_path(estimator, class_labels, cf_label)
                for key in paths_info:
                    """ generate epsilon-satisfactory instance """
                    path_info = paths_info[key]
                    es_instance = self.esatisfactory_instance(x, path_info)
                    if (
                        predict(estimator, es_instance) == cf_label
                        and cost_func(x, es_instance) < delta_mini
                    ):
                        x_out = es_instance
                        delta_mini = cost_func(x, es_instance)
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
