import re

import numpy as np
import xgboost

# CONSTANTS from
# https://github.com/scikit-learn/scikit-learn/blob/4907029b1ddff16b111c501ad010d5207e0bd177/sklearn/tree/_tree.pyx
TREE_LEAF = -1
TREE_UNDEFINED = -2

# https://stackoverflow.com/questions/45001775/find-all-floats-or-ints-in-a-given-string/45001796
re_numbers = re.compile(r"[+-]? *(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?")
re_feature = re.compile(r"\[(.*?)(?=[<,=,>,<=,>=])")


def logistic_function(x):
    # returns value between 0 and 1
    return 1 / (1 + np.exp(-x))


def _get_tree_from_booster(booster: xgboost.core.Booster):
    """get string from the booster object"""
    tree = booster.get_dump()[0]
    tree = tree.replace("\t", "")
    tree = tree.split("\n")
    tree = tree[:-1]  # last element is empty
    return tree


def _parse_node(node):
    """
    node has either format:
    'node_id:[split] yes=left_child_id,no=right_child_id,missing=?'
    or
    'node_id:leaf=value'
    """

    is_leaf = "leaf" in node
    if is_leaf:
        # find all numbers in a string
        numbers = [float(x) if "." in x else int(x) for x in re_numbers.findall(node)]
        node_id, value = numbers
        left_child = TREE_LEAF
        right_child = TREE_LEAF
        threshold = TREE_UNDEFINED
        feature = TREE_UNDEFINED
        # TODO this depends on the loss function used in the defined XGBoost model
        score = logistic_function(x=value)
    else:
        # find all numbers in a string
        numbers = [float(x) if "." in x else int(x) for x in re_numbers.findall(node)]
        node_id, threshold, left_child, right_child, _ = numbers
        # find which feature we split on
        feature = re_feature.findall(node)[0]
        score = None

    return node_id, threshold, feature, left_child, right_child, score


def parse_booster(booster: xgboost.core.Booster):

    tree = _get_tree_from_booster(booster)
    # xgboost.plot_tree(booster)
    # plt.show()
    n_nodes = len(tree)

    children_left = [None] * n_nodes
    children_right = [None] * n_nodes
    features = [None] * n_nodes
    thresholds = [None] * n_nodes
    scores = [None] * n_nodes

    for node in tree:
        (
            node_id,
            threshold,
            feature,
            left_child,
            right_child,
            score,
        ) = _parse_node(node)

        children_right[node_id] = right_child
        children_left[node_id] = left_child
        thresholds[node_id] = threshold
        features[node_id] = feature
        scores[node_id] = score

    children_left = np.array(children_left)
    children_right = np.array(children_right)
    thresholds = np.array(thresholds)
    features = np.array(features)
    scores = np.array(scores, dtype=np.float)

    return children_left, children_right, thresholds, features, scores
