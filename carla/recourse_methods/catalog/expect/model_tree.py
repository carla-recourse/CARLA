from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree._tree import TREE_UNDEFINED
from torch.distributions import Normal

from carla.models.api import MLModel
from carla.recourse_methods.api import RecourseMethod
from carla.recourse_methods.processing import (
    check_counterfactuals,
    merge_default_parameters,
)

from .hypercube import get_classes_for_hypercubes, get_hypercubes

INF = torch.tensor(float("inf"))


def random_uniform_grid(n_samples, min_vals, max_vals):
    # generate independent uniform samples for each dimension
    grid = [
        np.random.uniform(low, high, n_samples) for low, high in zip(min_vals, max_vals)
    ]
    # combine independent 1-d samples into multi-d samples
    grid = np.vstack(grid).T
    return grid


def get_distilled_model(model_rf, plot_step=0.1, include_training_data=True):
    data_df = model_rf.data.df
    X = data_df[model_rf.feature_input_order].to_numpy()
    y = data_df[model_rf.data.target].to_numpy()

    # TODO adapt for different normalization
    min_vals = np.min(X, axis=0)  # min values per feature
    max_vals = np.max(X, axis=0)  # max values per feature
    grid = random_uniform_grid(len(X), min_vals, max_vals)

    # Get labels
    if include_training_data:
        # augmented data + training data
        X = np.concatenate((X, grid))
        y = model_rf.predict(X)
    else:
        # only augmented data
        X = grid
        y = model_rf.predict(X)

    parameters = {
        "max_depth": [None],
        "min_samples_leaf": [5, 10, 15],
        "min_samples_split": [3, 5, 10, 15],
    }
    model = GridSearchCV(DecisionTreeClassifier(), parameters, n_jobs=4)
    model.fit(X=X, y=y)
    print(model.best_score_, model.best_params_)
    model_distilled = model.best_estimator_

    return model_distilled


# TODO move this to trees.py?
def get_rules(tree, feature_names: list, class_names: list, verbose=False) -> list:
    """
    solution from:
    https://stackoverflow.com/questions/20224526/how-to-extract-the-decision-rules-from-scikit-learn-decision-tree

    Parameters
    ----------
    tree
    feature_names
    class_names
    verbose

    Returns
    -------

    """

    def recurse(node, path: List, paths: List):
        if tree_.feature[node] != TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            p1, p2 = list(path), list(path)

            p1 += [f"|{name} <= {np.round(threshold, 3)}|"]
            recurse(tree_.children_left[node], p1, paths)

            p2 += [f"|{name} > {np.round(threshold, 3)}|"]
            recurse(tree_.children_right[node], p2, paths)
        else:
            path += [(tree_.value[node], tree_.n_node_samples[node])]
            paths += [path]

    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != TREE_UNDEFINED else "undefined!" for i in tree_.feature
    ]

    paths: List = []
    path: List = []
    recurse(0, path, paths)

    # sort by samples count
    samples_count = [p[-1][1] for p in paths]
    ii = list(np.argsort(samples_count))
    paths = [paths[i] for i in reversed(ii)]

    rules = []
    for path in paths:
        rule = "if "
        for p in path[:-1]:
            if rule != "if ":
                rule += " and "
            rule += str(p)
        rule += " then "

        if class_names is None:
            rule += "response: " + str(np.round(path[-1][0][0][0], 3))
        else:
            classes = path[-1][0][0]
            i = np.argmax(classes)
            rule += f"class: {class_names[i]}"
            if verbose:
                rule += f" | (proba: {np.round(100.0 * classes[i] / np.sum(classes), 2)}% of class {class_names[i]})"

        if verbose:
            rule += f" | based on {path[-1][1]:,} samples"

        rules += [rule]

    return rules


def cdf_difference(upper, lower, value, sigma_sq, verbose=False):
    normal = Normal(loc=value, scale=torch.sqrt(sigma_sq))
    upper_cdf = normal.cdf(upper)
    lower_cdf = normal.cdf(lower)
    cdf_diff = upper_cdf - lower_cdf

    if verbose:
        print(f"upper cdf {upper_cdf}")
        print(f"lower cdf {lower_cdf}")
        print(f"cdf diff {cdf_diff}")
        print("--------------------------------")

    if not cdf_diff >= 0:
        raise ValueError(
            "Something went wrong -> Hints: Confused upper with lower?; Are the Hypercubes correct?"
        )

    return cdf_diff


class EXPECT_tree(RecourseMethod):
    _DEFAULT_HYPERPARAMS = {
        "optimizer": "adam",
        "learning_rate": 0.05,
        "lambda": 0.2,
        "invalidation_target": 0.4,
        "upper_bound": INF,
        "lower_bound": -INF,
        "var": 0.25,
        "target_names": [1, 0],
    }

    def __init__(self, mlmodel: MLModel, hyperparams: Dict = None):
        super().__init__(mlmodel)
        self.hyperparams = merge_default_parameters(
            hyperparams, self._DEFAULT_HYPERPARAMS
        )

        self.lr = self.hyperparams["learning_rate"]
        self.lam = self.hyperparams["lambda"]
        self.invalidation_target = self.hyperparams["invalidation_target"]
        self.upper_bound = self.hyperparams["upper_bound"]
        self.lower_bound = self.hyperparams["lower_bound"]

        # Get Hypercubes
        model = get_distilled_model(mlmodel)
        self.feature_names = mlmodel.feature_input_order
        rules = get_rules(model, self.feature_names, self.hyperparams["target_names"])
        self.all_intervals = get_hypercubes(
            self.feature_names, rules, self.lower_bound, self.upper_bound
        )
        self.classes = get_classes_for_hypercubes(rules)

    def invalidation_loss(self, x, verbose=False):
        """Compute the invalidation loss.

        Parameters
        ----------
        x:
            Input for which we want to find invalidation loss.
        verbose:
            Print or not print.

        Returns
        -------

        """
        validation_rate = 0
        for i, interval in enumerate(self.all_intervals):
            predicted_label = torch.tensor(self.classes[i]).float()

            if verbose:
                print(f"it: {i}")
                print(f"predicted label: {predicted_label}")

            # only compute loss for positive class labels
            product_cdf_diff = 1
            for j, feature in enumerate(self.feature_names):
                if verbose:
                    print(f"it_f: {j}")

                interval_range = torch.tensor(interval[j][0][1])

                # replace upper and lower bounds due to data constrains
                # TODO should this depend on data normalization (e.g. min max vs scaled)
                if INF in interval_range:
                    interval_range = [
                        y if y != np.inf else self.upper_bound for y in interval_range
                    ]
                if -INF in interval_range:
                    interval_range = [
                        y if y != -np.inf else self.lower_bound for y in interval_range
                    ]

                # form updated intervals
                input_value = x[j].float()
                interval_lower, interval_upper = interval_range[0], interval_range[1]

                d_cdf = cdf_difference(
                    interval_upper,
                    interval_lower,
                    input_value,
                    sigma_sq=torch.tensor(self.hyperparams["var"]),
                )

                product_cdf_diff *= d_cdf

            validation_rate += predicted_label * product_cdf_diff

        invalidation_rate = torch.tensor(1).float() - validation_rate

        if verbose:
            print(f"validation rate: {validation_rate}")
            print(f"invalidation rate: {invalidation_rate}")

        return invalidation_rate

    def optimization(self, x: torch.tensor, n_iter=50):
        x_check = torch.tensor(x, requires_grad=True, dtype=torch.float)

        if self.hyperparams["optimizer"] == "adam":
            optim = torch.optim.Adam([x_check], self.lr)
        else:
            optim = torch.optim.RMSprop([x_check], self.lr)

        for i in range(n_iter):
            x_check.requires_grad = True

            invalidation_rate = self.invalidation_loss(x_check)

            # Compute & update losses
            loss_invalidation = invalidation_rate - self.invalidation_target

            # Hinge loss
            loss_invalidation[loss_invalidation < 0] = 0

            # Distance loss
            loss_distance = torch.norm(x_check - x, 1)

            # Combined loss
            loss = loss_invalidation + self.lam * loss_distance

            # Update loss
            optim.zero_grad()
            loss.backward()
            optim.step()

            print(f"invalidation rate iteration {i}: {invalidation_rate.item()}")
            print(f"loss at iteration {i}: {loss.item()}")
            print(f"CE suggestion at iteration {i}: {x_check}")

        return x_check.detach().numpy()

    def get_counterfactuals(self, factuals: pd.DataFrame):
        factuals = self._mlmodel.get_ordered_features(factuals)

        df_cfs = factuals.apply(
            lambda x: self.optimization(torch.from_numpy(x)),
            raw=True,
            axis=1,
        )

        df_cfs = check_counterfactuals(self._mlmodel, df_cfs, factuals.index)
        df_cfs = self._mlmodel.get_ordered_features(df_cfs)
        return df_cfs
