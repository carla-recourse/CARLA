import itertools

import numpy as np
from sklearn import preprocessing

from .sampler import Sampler


def initialize_non_saturated_action_set(
    scm,
    dataset,
    sampling_handle,
    classifier,
    factual_instance,
    intervention_set,
    num_samples=1,
    epsilon=5e-2,
):
    # default action_set
    action_set = dict(
        zip(
            intervention_set,
            [factual_instance.dict()[node] for node in intervention_set],
        )
    )

    for noise_multiplier in np.arange(0, 10.1, 0.1):
        # create an action set from the factual instance, and possibly some noise
        action_set = {
            k: v + noise_multiplier * np.random.randn() for k, v in action_set.items()
        }

        # sample values
        sampler = Sampler(scm)
        samples_df = sampler.sample(
            num_samples,
            factual_instance,
            action_set,
            sampling_handle,
        )

        # return action set if average predictive probability of samples >= eps (non-saturated region of classifier)
        predict_proba_list = classifier.predict_proba(samples_df)[:, 1]
        if (
            np.mean(predict_proba_list) >= epsilon and np.mean(predict_proba_list) - 0.5
        ):  # don't want to start on the other side
            return action_set

    return action_set


def get_discretized_action_sets(
    intervenable_nodes,
    scaler,
    decimals=5,
    grid_search_bins=10,
    max_intervention_cardinality=100,
):
    """
    Get possible action sets by finding valid actions on a grid.

    Parameters
    ----------
    intervenable_nodes: dict
        Contains nodes that are not immutable {"continous": [continous nodes], "categorical": [categical nodes].
    scaler: either sklearn.preprocessing.MinMaxScaler or sklearn.preprocessing.StandardScaler
        Needed to find what values to search over.
    decimals: int
        Determines the precision of the values to search over, in the case of continuous variables.
    grid_search_bins: int
        Determines the number of values to search over.
    max_intervention_cardinality: int
        Determines the maximum size of an action set.

    Returns
    -------
    dict containing the valid action sets.
    """

    # TODO are we using normalized or non-normalized data? (assuming normalized now)
    possible_actions_per_node = []

    for i, _ in enumerate(intervenable_nodes["continuous"]):
        if isinstance(scaler, preprocessing.MinMaxScaler):
            # search over [0, 1]
            # TODO is this index corresponding to same variable as in ["continous"]
            # min_value = scaler.data_min_[i]
            # max_value = scaler.data_max_[i]
            min_value = 0
            max_value = 1
        elif isinstance(scaler, preprocessing.StandardScaler):
            # search over mean +- 2 * std
            # mean_value = scaler.mean_[i]
            # std_value = scaler.scale_[i]
            mean_value = 0
            std_value = 1
            min_value = mean_value - 2 * std_value
            max_value = mean_value + 2 * std_value
        else:
            raise ValueError(f"scaler: {scaler} not recognized!")
        grid = list(
            np.around(np.linspace(min_value, max_value, grid_search_bins), decimals)
        )
        grid.append(None)
        grid = list(dict.fromkeys(grid))
        possible_actions_per_node.append(grid)

    for _ in intervenable_nodes["categorical"]:
        # TODO only binary categories supported right now
        grid = list(np.around(np.linspace(0, 1, grid_search_bins), decimals=0))
        grid.append(None)
        grid = list(dict.fromkeys(grid))
        possible_actions_per_node.append(grid)

    all_action_tuples = list(itertools.product(*possible_actions_per_node))
    all_action_tuples = [
        _tuple
        for _tuple in all_action_tuples
        if len([element for element in _tuple if element is not None])
        < max_intervention_cardinality
    ]

    # get all node names
    nodes = np.concatenate(
        [intervenable_nodes["continuous"], intervenable_nodes["categorical"]]
    )
    # create from list and tuple a dict: {nodes[0]: tuple[0], nodes[1]: tuple[1], etc.}
    all_action_sets = [dict(zip(nodes, _tuple)) for _tuple in all_action_tuples]

    valid_action_sets = []
    for action_set in all_action_sets:
        valid_action_set = {k: v for k, v in action_set.items() if v is not None}
        valid_action_sets.append(valid_action_set)

    return valid_action_sets
