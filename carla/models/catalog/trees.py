import numpy as np
import tensorflow as tf
import xgboost.core
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier

from carla.models.catalog.parse_xgboost import parse_booster


def _split_approx(node, feat_input, feat_index, threshold, sigma):
    """
    Approximate the decision tree split using a sigmoid function.
    """

    def _approx_activation_by_index(feat_input, feat_index, threshold, sigma):
        x = feat_input[:, feat_index] - threshold
        activation = tf.math.sigmoid(x * sigma)
        return 1.0 - activation, activation

    if node is None:
        node = 1.0

    l_n, r_n = _approx_activation_by_index(feat_input, feat_index, threshold, sigma)
    return node * l_n, node * r_n


def _parse_class_tree(tree, feat_columns, feat_input, split_function):
    if isinstance(tree, xgboost.core.Booster):  # XGBoost
        children_left, children_right, threshold, feature, scores = parse_booster(tree)

        # feature needs to be list of ints, not string
        # -2 is the indicator of a leaf node
        feature = [tree.feature_names.index(f) if not f == "-2" else f for f in feature]

        # TODO make n_classes not hardcoded?
        n_classes = 2
        n_nodes = len(scores)
    else:  # Sklearn
        # Code is adapted from https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html
        children_left = tree.tree_.children_left
        children_right = tree.tree_.children_right
        feature = tree.tree_.feature
        threshold = tree.tree_.threshold
        n_classes = len(tree.classes_)
        n_nodes = tree.tree_.node_count
        values = tree.tree_.value

    nodes = [None] * n_nodes
    leaf_nodes = [[] for _ in range(n_classes)]
    for i in range(n_nodes):
        cur_node = nodes[i]
        if children_left[i] != children_right[i]:  # split node
            l_n, r_n = split_function(cur_node, feat_input, feature[i], threshold[i])
            nodes[children_left[i]] = l_n
            nodes[children_right[i]] = r_n
        else:  # leaf node
            if isinstance(tree, xgboost.core.Booster):
                # TODO score to class depends on the objective function of the XGBoost model
                max_class = scores[i] > 0.5
            else:
                max_class = np.argmax(values[i])
            leaf_nodes[max_class].append(cur_node)

    return leaf_nodes, n_nodes, n_classes


def get_prob_classification_tree(tree, feat_columns, feat_input, sigma):
    """
    class probability for input
    """

    def split_function(node, feat_input, feat_index, threshold):
        return _split_approx(node, feat_input, feat_index, threshold, sigma)

    # leaf nodes has a value for each feature
    leaf_nodes, n_nodes, n_classes = _parse_class_tree(
        tree, feat_columns, feat_input, split_function
    )
    if n_nodes > 1:  # tree has multiple nodes
        if [] in leaf_nodes:  # if no leaf predicts (majority vote) this class
            # TODO only works for 2 classes
            # class which is empty
            idx = leaf_nodes.index([])
            out_l = [None] * 2
            # compute the sum of the other non-empty class
            out_l[1 - idx] = sum(leaf_nodes[1 - idx])
            out_l[idx] = 1 - out_l[1 - idx]
        else:
            out_l = [sum(leaf_nodes[c_i]) for c_i in range(n_classes)]
        stacked = tf.stack(out_l, axis=-1)

    else:  # sometimes tree only has one node
        only_class = tree.predict(
            tf.reshape(feat_input[0, :], shape=(1, -1))
        )  # can differ depending on particular samples used to train each tree

        correct_class = tf.constant(
            1, shape=(len(feat_input)), dtype=tf.float64
        )  # prob(belong to correct class) = 100 since there's only one node
        incorrect_class = tf.constant(
            0, shape=(len(feat_input)), dtype=tf.float64
        )  # prob(wrong class) = 0
        if only_class == 1.0:
            class_0 = incorrect_class
            class_1 = correct_class
        elif only_class == 0.0:
            class_0 = correct_class
            class_1 = incorrect_class
        else:
            raise ValueError
        class_labels = [class_0, class_1]
        stacked = tf.stack(class_labels, axis=1)
    return stacked


def get_prob_classification_forest(
    model, feat_columns, feat_input, number_trees=100, sigma=10.0, temperature=1.0
):
    def tree_parser(tree):
        """parse and individual tree"""
        return get_prob_classification_tree(tree, feat_columns, feat_input, sigma)

    if model.backend == "sklearn":
        tree_l = [
            tree_parser(estimator) for estimator in model.tree_iterator.estimators_
        ][:number_trees]
    elif model.backend == "xgboost":
        tree_l = [tree_parser(estimator) for estimator in model.tree_iterator][
            :number_trees
        ]

    if isinstance(model.tree_iterator, AdaBoostClassifier):
        weights = model.tree_iterator.estimator_weights_
    elif isinstance(model.tree_iterator, RandomForestClassifier) or isinstance(
        model.tree_iterator[0], xgboost.core.Booster
    ):
        weights = np.full(
            len(model.tree_iterator),
            1 / len(model.tree_iterator),
        )
    else:
        raise Exception("model not supported")

    logits = sum(w * tree for w, tree in zip(weights, tree_l))

    if type(temperature) in [float, int]:
        expits = tf.exp(temperature * logits)
    else:
        expits = tf.exp(temperature[:, None] * logits)

    softmax = expits / tf.reduce_sum(expits, axis=1)[:, None]

    return softmax
