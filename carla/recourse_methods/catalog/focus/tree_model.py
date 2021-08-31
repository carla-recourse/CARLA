import re

import numpy as np
import pandas as pd
import xgboost
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from carla import MLModel


# Custom black-box models need to inherit from
# the MLModel interface
class TreeModel(MLModel):
    def __init__(self, data):
        super().__init__(data)
        # The constructor can be used to load or build an
        # arbitrary black-box-model
        self._mymodel = DecisionTreeClassifier(max_depth=4)

        # add support for methods that can also use categorical data
        data_transformed = self.scaler.transform(data.raw[data.continous])
        target = data.raw[data.target]

        X_train, X_test, y_train, y_test = train_test_split(
            data_transformed, target, test_size=0.20
        )
        self._mymodel.fit(X=X_train, y=y_train)
        train_score = self._mymodel.score(X=X_train, y=y_train)
        test_score = self._mymodel.score(X=X_test, y=y_test)
        print(
            "model fitted with training score {} and test score {}".format(
                train_score, test_score
            )
        )

        self._feature_input_order = data.continous

    @property
    def feature_input_order(self):
        # List of the feature order the ml model was trained on
        return self._feature_input_order

    @property
    def backend(self):
        # The ML framework the model was trained on
        return "sklearn"

    @property
    def raw_model(self):
        # The black-box model object
        return self._mymodel

    # The predict function outputs
    # the continous prediction of the model
    def predict(self, x):
        return self._mymodel.predict(x)

    # The predict_proba method outputs
    # the prediction as class probabilities
    def predict_proba(self, x):
        return self._mymodel.predict_proba(x)


# Custom black-box models need to inherit from
# the MLModel interface
class ForestModel(MLModel):
    def __init__(self, data):
        super().__init__(data)
        # The constructor can be used to load or build an
        # arbitrary black-box-model
        self._mymodel = RandomForestClassifier(
            n_estimators=5,
            max_depth=2,
        )
        data_transformed = self.scaler.transform(data.raw[data.continous])
        target = data.raw[data.target]

        X_train, X_test, y_train, y_test = train_test_split(
            data_transformed, target, test_size=0.20
        )
        self._mymodel.fit(X=X_train, y=y_train)
        train_score = self._mymodel.score(X=X_train, y=y_train)
        test_score = self._mymodel.score(X=X_test, y=y_test)
        print(
            "model fitted with training score {} and test score {}".format(
                train_score, test_score
            )
        )

        self._feature_input_order = data.continous

    @property
    def feature_input_order(self):
        # List of the feature order the ml model was trained on
        return self._feature_input_order

    @property
    def backend(self):
        # The ML framework the model was trained on
        return "sklearn"

    @property
    def raw_model(self):
        # The black-box model object
        return self._mymodel

    @property
    def tree_iterator(self):
        return self.raw_model

    # The predict function outputs
    # the continous prediction of the model
    def predict(self, x):
        return self._mymodel.predict(x)

    # The predict_proba method outputs
    # the prediction as class probabilities
    def predict_proba(self, x):
        return self._mymodel.predict_proba(x)


def parse_booster(booster: xgboost.core.Booster):
    # CONSTANTS from
    # https://github.com/scikit-learn/scikit-learn/blob/4907029b1ddff16b111c501ad010d5207e0bd177/sklearn/tree/_tree.pyx
    TREE_LEAF = -1
    TREE_UNDEFINED = -2

    # https://stackoverflow.com/questions/45001775/find-all-floats-or-ints-in-a-given-string/45001796
    re_numbers = re.compile(r"[+-]? *(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?")
    re_feature = re.compile(r"\[(.*?)(?=[<,=,>,<=,>=])")

    def parse_node(node):
        """
        node has either format:
        'node_id:[split] yes=left_child_id,no=right_child_id,missing=?'
        or
        'node_id:leaf=value'
        """

        def logistic_function(x):
            # returns value between 0 and 1
            return 1 / (1 + np.exp(-x))

        is_leaf = "leaf" in node
        if is_leaf:
            # find all numbers in a string
            numbers = [
                float(x) if "." in x else int(x) for x in re_numbers.findall(node)
            ]
            node_id, value = numbers
            left_child = TREE_LEAF
            right_child = TREE_LEAF
            threshold = TREE_UNDEFINED
            feature = TREE_UNDEFINED
            # TODO this depends on the loss function used in the defined XGBoost model
            score = logistic_function(x=value)
        else:
            # find all numbers in a string
            numbers = [
                float(x) if "." in x else int(x) for x in re_numbers.findall(node)
            ]
            node_id, threshold, left_child, right_child, _ = numbers
            # find which feature we split on
            feature = re_feature.findall(node)[0]
            score = None

        return node_id, threshold, feature, left_child, right_child, score

    def get_tree_from_booster(booster):
        """get string from the booster object"""
        tree = booster.get_dump()[0]
        tree = tree.replace("\t", "")
        tree = tree.split("\n")
        tree = tree[:-1]  # last element is empty
        return tree

    tree = get_tree_from_booster(booster)
    # xgboost.plot_tree(booster)
    # plt.show()
    n_nodes = len(tree)

    # TODO sorting the tree by node_id rather then indexing by node_id could be faster
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
        ) = parse_node(node)

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


class XGBoostModel(MLModel):
    """The default way of implementing XGBoost
    https://xgboost.readthedocs.io/en/latest/python/python_intro.html"""

    def __init__(self, data):
        super().__init__(data)
        self._feature_input_order = data.continous

        data_transformed = self.scaler.transform(data.raw[data.continous])
        data_transformed = pd.DataFrame(data_transformed, columns=data.continous)
        target = data.raw[data.target]

        X_train, X_test, y_train, y_test = train_test_split(
            data_transformed, target, test_size=0.20
        )

        self.X = X_train

        param = {
            "max_depth": 2,  # determines how deep the tree can go
            "objective": "binary:logistic",  # determines the loss function
            "n_estimators": 5,
        }
        self._mymodel = xgboost.XGBClassifier(**param)
        self._mymodel.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            eval_metric="logloss",
            verbose=True,
        )

    @property
    def feature_input_order(self):
        # List of the feature order the ml model was trained on
        return self._feature_input_order

    @property
    def backend(self):
        # The ML framework the model was trained on
        return "xgboost"

    @property
    def raw_model(self):
        # The black-box model object
        return self._mymodel

    @property
    def tree_iterator(self):
        # make a copy of the trees, else feature names are not saved
        booster_it = [booster for booster in self.raw_model.get_booster()]
        # set the feature names
        for booster in booster_it:
            booster.feature_names = self.feature_input_order
        return booster_it

    # The predict function outputs
    # the continuous prediction of the model
    def predict(self, x):
        return self._mymodel.predict(x)

    # The predict_proba method outputs
    # the prediction as class probabilities
    def predict_proba(self, x):
        return self._mymodel.predict_proba(x)
