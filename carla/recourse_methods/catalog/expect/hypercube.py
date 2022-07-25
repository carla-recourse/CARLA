import re

import numpy as np


def get_classes_for_hypercubes(rules: list) -> list:
    classes_for_hypercubes = []
    for rule in rules:
        c = rule.split(": ")[-1]  # last element in string is class information
        c = int(c)
        classes_for_hypercubes.append(c)
    return classes_for_hypercubes


def get_plain_rules(rules: list) -> list:
    plain_rules = []
    for rule in rules:
        r = rule.split("|")
        r.pop(-1)  # remove class information
        r.remove("if ")  # remove 'if'
        while " and " in r:
            r.remove(" and ")  # remove all 'and's
        plain_rules.append(r)
    return plain_rules


def get_threshold_grouped_rules(rule_subset: list) -> list:
    """
    potential input: potential rule_subset:
    ['x1 <= -0.064', 'x1 > -0.652', 'x1 <= -0.312']

    potential output: threshold_grouped_rules:
    [['x1 <= -0.064', 'x1 <= -0.312'], ['x1 > -0.652']]
    """

    list_leq = []  # list for 'less or equal'-rules
    list_g = []  # list for 'greater than'-rules
    for rule in rule_subset:
        if "<=" in rule:
            list_leq.append(rule)
        else:
            list_g.append(rule)
    threshold_grouped_rules = [list_leq, list_g]
    threshold_grouped_rules = list(
        filter(None, threshold_grouped_rules)
    )  # get rid of empty lists
    return threshold_grouped_rules


class HypercubeInfo:
    def __init__(
        self,
        tree_classifier,
        feature_names,
        target_names,
        lower_bound=-np.inf,
        upper_bound=+np.inf,
    ):

        self.tree_classifier = tree_classifier
        self.feature_names = feature_names
        self.target_names = target_names
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def _get_grouped_rules(self, plain_rules: list):
        """
        input example: plain_rules , if feature names are 'x1' and 'x2'
        [['x1 > -0.064', 'x1 > 0.263', 'x2 > -1.453'],
         ['x1 <= -0.064', 'x1 <= -0.652', 'x1 <= -1.028']]

        output example: grouped_rules , if feature names are 'x1' and 'x2'
        [['x1 > -0.064', 'x1 > 0.263'], ['x2 > -1.453']],
         [['x1 <= -0.064', 'x1 <= -0.652', 'x1 <= -1.028']]

        We call 'x1 > -0.064' a comparison
        """
        all_grouped_rules_cleaned = []
        all_grouped_rules_complete = []
        for rules in plain_rules:
            grouped_rules = []
            grouped_rules_cleaned = []
            for feature_name in self.feature_names:
                fine_rules = []
                for comparison in rules:
                    if feature_name in comparison:
                        fine_rules.append(comparison)
                grouped_rules.append(fine_rules)
                grouped_rules_cleaned = list(
                    filter(None, grouped_rules)
                )  # get rid of empty lists
            all_grouped_rules_cleaned.append(grouped_rules_cleaned)
            all_grouped_rules_complete.append(grouped_rules)
        return all_grouped_rules_cleaned, all_grouped_rules_complete

    def _form_interval(self, rule_subset: list) -> list:
        """
        input example: threshold_grouped_rules:
        [['x1 <= -0.064', 'x1 <= -0.312'], ['x1 > -0.652']]
        """
        n_elements = len(rule_subset)
        assert (n_elements > 0) and (
            n_elements < 3
        ), f"n_elements is {n_elements}: Something went wrong in the construction of the threshold_grouped_rules"

        intervals = []
        # covers the case of lower bound =< x =< upper bound
        if n_elements == 2:
            # obtain two intervals since we have two kind of rules: <= and >
            for i in range(2):
                comparisons = rule_subset[i]

                # only last split matters
                comparison_split = re.split(" > | <= ", comparisons[-1])
                threshold = float(comparison_split[-1])

                # determine endpoints
                if ">" in comparisons[-1]:
                    lower = threshold
                    upper = self.upper_bound
                else:
                    upper = threshold
                    lower = self.lower_bound
                interval = [lower, upper]
                intervals.append(interval)

            # combine the two intervals
            left_intervals = []
            right_intervals = []
            for inter in intervals:
                if -np.inf in inter:
                    left_intervals.append(inter)
                else:
                    right_intervals.append(inter)

            # obtain final interval
            lower = np.max(left_intervals)
            upper = np.min(right_intervals)
            intervals = sorted([lower, upper], reverse=False)
        # covers the case of (a) lower bound =< x or (b) upper bound >= x
        else:
            comparisons = rule_subset[0]
            thresholds = []
            comparison_split = re.split(" > | <= ", comparisons[-1])
            thresholds.append(float(comparison_split[-1]))

            # determine endpoints
            if ">" in comparisons[-1]:
                lower = np.min(thresholds)
                upper = self.upper_bound
            else:
                upper = np.max(thresholds)
                lower = self.lower_bound
            intervals = [lower, upper]

        return intervals

    def get_hypercubes(self, rules: list) -> list:
        plain_rules = get_plain_rules(rules)
        grouped_rules_cleaned, grouped_rules_complete = self._get_grouped_rules(
            plain_rules
        )

        all_intervals = []
        for rule_set in grouped_rules_complete:
            intervals = []
            for index, rule_subset in enumerate(rule_set):
                feature = self.feature_names[index]
                if rule_subset:
                    grouped_rule_subset = get_threshold_grouped_rules(rule_subset)
                    interval = self._form_interval(grouped_rule_subset)
                else:
                    interval = [self.lower_bound, self.upper_bound]
                zipped_interval = list(zip([feature], [interval]))
                intervals.append(zipped_interval)
            all_intervals.append(intervals)
        return all_intervals
