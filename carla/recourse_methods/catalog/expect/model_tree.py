from typing import Dict

import numpy as np
import pandas as pd
import torch
from torch.distributions import Normal
from sklearn.tree._tree import TREE_UNDEFINED

from carla.models.api import MLModel
from carla.recourse_methods.api import RecourseMethod
from carla.recourse_methods.processing import (
    merge_default_parameters,
    check_counterfactuals,
)

from .hypercube import get_classes_for_hypercubes, HypercubeInfo

INF = torch.tensor(float("inf"))


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

    def recurse(node, path, paths):
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

    paths = []
    path = []
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
            l = np.argmax(classes)
            rule += f"class: {class_names[l]}"
            if verbose:
                rule += f" | (proba: {np.round(100.0 * classes[l] / np.sum(classes), 2)}% of class {class_names[l]})"

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
        rules = get_rules(mlmodel, mlmodel.feature_input_order, mlmodel.data.target)
        self.all_intervals = HypercubeInfo(
            mlmodel, mlmodel.feature_input_order, mlmodel.data.target
        ).get_hypercubes(rules)
        self.classes = get_classes_for_hypercubes(rules)

    def invalidation_loss(
        self, x, verbose: bool = False,
    ):
        """

        Parameters
        ----------
        x:
            Input for which we want to find invalidation.
        verbose:
            Print or not.

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
                    sigma_sq=self.hyperparams["var"],
                )

                product_cdf_diff *= d_cdf

            validation_rate += predicted_label * product_cdf_diff

        invalidation_rate = torch.tensor(1).float() - validation_rate

        if verbose:
            print(f"validation rate: {validation_rate}")
            print(f"invalidation rate: {invalidation_rate}")

        return invalidation_rate

    def optimization(
        self, x: torch.tensor, all_intervals: list, classes: list,
    ):
        x_check = torch.tensor(x, requires_grad=True, dtype=torch.float)

        if self.hyperparams["optimizer"] == "adam":
            optim = torch.optim.Adam([x_check], self.lr)
        else:
            optim = torch.optim.RMSprop([x_check], self.lr)

        for i in range(50):
            x_check.requires_grad = True

            invalidation_rate = self.invalidation_loss(x_check, all_intervals, classes)

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
            lambda x: self.optimization(torch.from_numpy(x)), raw=True, axis=1
        )

        df_cfs = check_counterfactuals(self._mlmodel, df_cfs, factuals.index)
        df_cfs = self._mlmodel.get_ordered_features(df_cfs)
        return df_cfs
