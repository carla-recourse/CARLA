import re

import numpy as np


def get_classes_from_rules(rules: list) -> list:
    """
    Extract class for every rule in rules.

    Parameters
    ----------
    rules:
        List of classification rules.

    Returns
    -------

    """
    classes = []
    for rule in rules:
        # last element in string is class information
        c = rule.split(": ")[-1]
        c = int(c)
        classes.append(c)
    return classes


def get_plain_rules(rules: list) -> list:
    """
    An example rule looks like:
        'if |x1 <= 0.4| and |x2 <= 0.7| and |x3 > 0.2| then class: 1'
    The corresponding plain rule would look like:
        ['x1 <= 0.4', 'x2 <= 0.7', 'x3 > 0.2']

    Parameters
    ----------
    rules:
        List of classification rules.

    Returns
    -------

    """
    re_plain_rule = re.compile(r"\|(.*?)\|")
    plain_rules = [re_plain_rule.findall(rule) for rule in rules]
    return plain_rules


def get_threshold_grouped_rules(rule_subset: list) -> list:
    """Group rules based on their threshold direction.

    An example input would look like:
    ['x1 <= -0.064', 'x1 > -0.652', 'x1 <= -0.312']

    The corresponding output would be like:
    [['x1 <= -0.064', 'x1 <= -0.312'], ['x1 > -0.652']]

    Parameters
    ----------
    rule_subset:
        List of classification rules.

    """
    list_leq = [r for r in rule_subset if "<=" in r]
    list_g = [r for r in rule_subset if ">" in r]
    threshold_grouped_rules = [list_leq, list_g]
    threshold_grouped_rules = list(
        filter(None, threshold_grouped_rules)
    )  # get rid of empty lists
    return threshold_grouped_rules


def get_feature_grouped_rules(feature_names: list, plain_rules: list) -> list:
    """Group rules based on their feature.

    An example input could be:
    [['x1 > -0.064', 'x1 > 0.263', 'x2 > -1.453'],
     ['x1 <= -0.064', 'x1 <= -0.652', 'x1 <= -1.028']]

    The corresponding output would be.
    [['x1 > -0.064', 'x1 > 0.263'], ['x2 > -1.453']],
     [['x1 <= -0.064', 'x1 <= -0.652', 'x1 <= -1.028']]

    Parameters
    ----------
    feature_names:
        Names of the features, e.g. 'x1' and 'x2'.
    plain_rules:
        List of classification rules.

    Returns
    -------

    """

    def get_feature_group(rule):
        # group rule into multiple rules, one for each feature.
        feature_grouped_rules = []
        for feature in feature_names:
            feature_group = [r for r in rule if feature in r]
            feature_grouped_rules.append(feature_group)
        # feature_grouped_rules = list(
        #     filter(None, feature_grouped_rules)
        # )  # get rid of empty lists
        return feature_grouped_rules

    grouped_rules = []
    for rule in plain_rules:
        feature_grouped_rules = get_feature_group(rule)
        grouped_rules.append(feature_grouped_rules)
    return grouped_rules


def form_interval(rule_subset: list, lower_bound, upper_bound):
    """
    input example: threshold_grouped_rules:
    [['x1 <= -0.064', 'x1 <= -0.312'], ['x1 > -0.652']]
    """

    def get_interval(comparison):
        # only the last split matters
        comparison_split = re.split(" > | <= ", comparison[-1])
        threshold = float(comparison_split[-1])

        # determine endpoints
        if ">" in comparison[-1]:
            upper = upper_bound
            lower = threshold
        else:
            upper = threshold
            lower = lower_bound

        return [lower, upper]

    n_elements = len(rule_subset)
    if (
        n_elements == 1
    ):  # covers the case of (a) lower bound =< x or (b) upper bound >= x
        return get_interval(rule_subset[0])
    elif n_elements == 2:  # covers the case of lower bound <= x <= upper bound
        # obtain two intervals since we have two kind of rules: <= and >
        interval_1 = get_interval(rule_subset[0])
        interval_2 = get_interval(rule_subset[1])
        intervals = [interval_1, interval_2]

        # combine the two intervals
        left_interval = [inter for inter in intervals if lower_bound in inter]
        right_interval = [inter for inter in intervals if upper_bound in inter]

        # obtain final interval
        left = np.max(left_interval)
        right = np.min(right_interval)

        return sorted([left, right], reverse=False)
    else:
        raise ValueError(
            f"n_elements is {n_elements}: Something went wrong in the construction of the threshold_grouped_rules"
        )


def get_hypercubes(feature_names, rules: list, lower_bound, upper_bound):
    plain_rules = get_plain_rules(rules)
    grouped_rules_complete = get_feature_grouped_rules(feature_names, plain_rules)

    all_intervals = []
    for rule_set in grouped_rules_complete:
        feature_intervals = []
        for feature, rule_subset in zip(feature_names, rule_set):
            if rule_subset:
                grouped_rule_subset = get_threshold_grouped_rules(rule_subset)
                interval = form_interval(grouped_rule_subset, lower_bound, upper_bound)
            else:  # if rule_subset == []
                interval = [lower_bound, upper_bound]
            zipped_interval = list(zip([feature], [interval]))
            feature_intervals.append(zipped_interval)
        all_intervals.append(feature_intervals)
    return all_intervals
