import hashlib

import numpy as np


def cost_1(
    factual: np.array, counterfactual: np.array, norm_cdf: np.array, bin_edges: np.array
):
    """
    Cost function as absolute difference
    :param factual: np.array with original instance
    :param counterfactual: np.array with counterfactual instance
    :param norm_cdf: np.array matrix with normed cdfs
    :param bin_edges: np.array matrix withedges of each bin
    :return: scalar with cost
    """
    norm_cdf_factual = cdf_of_instance(norm_cdf, bin_edges, factual)[:-1]
    norm_cdf_counterfactual = cdf_of_instance(norm_cdf, bin_edges, counterfactual)[:-1]

    delta = np.abs(norm_cdf_counterfactual - norm_cdf_factual)
    cost_1 = np.sum(delta)

    return cost_1


def cost_2(
    factual: np.array, counterfactual: np.array, norm_cdf: np.array, bin_edges: np.array
):
    """
    Cost function as maximum difference
    :param factual: np.array with original instance
    :param counterfactual: np.array with counterfactual instance
    :param norm_cdf: np.array matrix with normed cdfs
    :param bin_edges: np.array matrix withedges of each bin
    :return: scalar with cost
    """
    norm_cdf_factual = cdf_of_instance(norm_cdf, bin_edges, factual)[:-1]
    norm_cdf_counterfactual = cdf_of_instance(norm_cdf, bin_edges, counterfactual)[:-1]

    delta = np.abs(norm_cdf_counterfactual - norm_cdf_factual)
    cost_2 = np.max(delta)

    return cost_2


def cdf_of_instance(norm_cdf: np.array, bin_edges: np.array, instance: np.array):
    """
    Compute the Q_j(instance)
    :param norm_cdf: np.array with normed cdfs
    :param bin_edges: np.array with edges of the bins
    :param instance: np.array with the instance we want to compute the cdf
    :return:
    """
    density_instance = np.zeros(np.shape(instance))
    binned_instance = get_bin_number(bin_edges, instance)

    for i in range(np.shape(binned_instance)[1]):
        bin = int(binned_instance[:, i][0])
        density_value = norm_cdf[bin][i]
        density_instance[i] = density_value

    return density_instance


def get_bin_number(bin_edges: np.array, instance: np.array) -> np.array:
    """
    Find the correct bin for each feature of an instance
    :param bin_edges: np.array with bin edges
    :param instance: np.array with instance features
    :return: np.array with bin for each feature
    """
    n, p = np.shape(bin_edges)
    x = np.zeros((1, p))
    for i in range(p):
        value = instance[i]

        if type(value) == str:
            # hash string feature
            value = int(transform_feature_to_int(np.array([value]), n)[0])

        # Find the correct bin for a specific feature
        for j in range(n):
            if j == 0:
                if value <= bin_edges[j, i]:
                    x[:, i] = j
                    break
            else:
                if bin_edges[j, i] >= value > bin_edges[j - 1, i]:
                    x[:, i] = j
                    break

    return x


def transform_feature_to_int(column, n):
    """
    Transform Column with String features
    :param column:
    :return:
    """
    digits = int(np.log10(n)) + 1
    for i in range(column.size):
        column[i] = (
            int(hashlib.sha256(column[i].encode("utf-8")).hexdigest(), 16)
            % 10 ** digits
        )

    return column
