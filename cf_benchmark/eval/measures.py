import hashlib

import library.data_processing as processing
import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import NearestNeighbors


def get_delta(instance, cf):
    """
    Compute difference between original instance and counterfactual
    :param instance: List of features of original instance
    :param cf: List of features of counterfactual
    :return: List of differences between cf and original instance
    """
    delta = []
    for i, original in enumerate(instance):
        counterfactual = cf[i]

        if type(original) == str:
            if original == counterfactual:
                delta.append(0)
            else:
                delta.append(1)
        else:
            delta.append(counterfactual - original)

    return delta


def get_max_list(data):
    """
    get max element for every column.
    Max for string elements is 1
    :param data: numpy array
    :return: list of max elements
    """
    max = []
    for i in range(data.shape[-1] - 1):
        column = data[:, i]

        if type(column[0]) == str:
            max.append(1)
        else:
            max.append(np.max(column))

    return max


def get_min_list(data):
    """
    get min element for every column.
    Min for string elements is 0
    :param data: numpy array
    :return: list of min elements
    """
    min = []
    for i in range(data.shape[-1] - 1):
        column = data[:, i]

        if type(column[0]) == str:
            min.append(0)
        else:
            min.append(np.min(column))

    return min


def get_range(df):
    """
    Get range max - min of every feature
    :param df: dataframe object of dataset
    :return: list of ranges for every feature
    """
    data = df.values
    max = get_max_list(data)
    min = get_min_list(data)

    range = [x[0] - x[1] for x in zip(max, min)]

    return range


def distance_d1(instance, cf):
    """
    Compute d1-distance
    :param instance: List of original feature
    :param cf: List of counterfactual feature
    :return: Scalar number
    """
    # get difference between original and counterfactual
    delta = get_delta(instance, cf)

    # compute elements which are greater than 0
    delta_bin = [i != 0 for i in delta]
    delta_bin = delta_bin[:-1]  # loose label column

    d1 = sum(delta_bin)

    return d1


def distance_d2(instance, cf, df):
    """
    Compute d2 distance
    :param instance: List of original feature
    :param cf: List of counterfactual feature
    :param df: Dataframe object of dataset
    :return: Scalar number
    """
    # get difference between original and counterfactual
    delta = get_delta(instance, cf)
    delta = delta[:-1]  # loose label column

    # get range of every feature
    range = get_range(df)

    d2 = [np.abs(x[0] / x[1]) for x in zip(delta, range)]
    d2 = sum(d2)

    return d2


def distance_d3(instance, cf, df):
    """
    Compute d3 distance
    :param instance: List of original feature
    :param cf: List of counterfactual feature
    :param df: Dataframe object of dataset
    :return: Scalar number
    """
    # get difference between original and counterfactual
    delta = get_delta(instance, cf)
    delta = delta[:-1]  # loose label column

    # get range of every feature
    range = get_range(df)

    d3 = [(x[0] / x[1]) ** 2 for x in zip(delta, range)]
    d3 = sum(d3)

    return d3


def distance_d4(instance, cf):
    """
    Compute d4 distance
    :param instance: List of original feature
    :param cf: List of counterfactual feature
    :return: Scalar number
    """
    # get difference between original and counterfactual
    delta = get_delta(instance, cf)
    delta = delta[:-1]  # loose label column

    d4 = [np.abs(x) for x in delta]
    d4 = np.max(d4)

    return d4


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


def compute_cdf(data):
    # per free feature
    # relies on computing histogram first
    # num_bins: # bins in histogram
    # you can use bin_edges & norm_cdf to plot cdf

    n, p = np.shape(data)
    # num_bins = n
    norm_cdf = np.zeros((n, p))
    bins = np.zeros((n, p))

    for j in range(p):
        column = data[:, j]
        # Check if feature type is string
        if type(column[0]) == str:
            # transform string feature into int
            column = transform_feature_to_int(column, n)
        counts, bin_edges = np.histogram(column, bins=n, density=True)
        cdf = np.cumsum(counts)
        bins[:, j] = bin_edges[1:]
        norm_cdf[:, j] = cdf / cdf[-1]

    return bins, norm_cdf


def get_bin_number(bin_edges, instance):
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


def cdf_of_instance(norm_cdf, bin_edges, instance):
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


def cost_1(factual, counterfactual, norm_cdf, bin_edges):
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


def cost_2(factual, counterfactual, norm_cdf, bin_edges):
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


def redundancy(factual, counterfactual, model):
    """
    Redundency metric which looks for unnecessary changes
    :param factual: np.array with one-hot-encoded original instance
    :param counterfactual: np.array with one-hot-encoded counterfactual instance
    :param model: pytorch model
    :return: scalar, number of unnecessary changes
    """
    red = 0

    # get model prediction and cast it from tensor to float
    pred_f = model(torch.from_numpy(factual).float()).detach().numpy().reshape(1)[0]
    pred_cf = (
        model(torch.from_numpy(counterfactual).float()).detach().numpy().reshape(1)[0]
    )

    if pred_f != pred_cf:
        for i in range(factual.shape[1]):
            if factual[0][i] != counterfactual[0][i]:
                temp_cf = np.copy(counterfactual)

                # reverse change in counterfactual and predict new label
                temp_cf[0][i] = factual[0][i]
                pred_temp_cf = (
                    model(torch.from_numpy(temp_cf).float())
                    .detach()
                    .numpy()
                    .reshape(1)[0]
                )

                # if new prediction has the same label as the old prediction for cf, increase redundancy
                if pred_temp_cf == pred_cf:
                    red += 1

    else:
        print("Factual and counterfactual are in the same class")
        return red

    return red


def yNN(counterfactuals, data, label, k, cat_features, cont_features, model):
    """
    Compute yNN measure
    :param counterfactuals: List wit dataframes of counterfactual instances
    :param data: dataframe with whole dataset
    :param label: string with target class
    :param k: number of nearest neighbours
    :param cat_features: list with all categorical features
    :param cat_features: list with all continuous features
    :param model: pytorch model
    :return: scalar
    """
    N = len(counterfactuals)
    number_of_diff_labels = 0
    norm_data = processing.normalize_instance(data, data, cont_features)
    enc_data = pd.get_dummies(norm_data, columns=cat_features)

    nbrs = NearestNeighbors(n_neighbors=k).fit(enc_data.values)

    for cf_df in counterfactuals:
        norm_cf = processing.normalize_instance(data, cf_df, cont_features)
        enc_cf = processing.one_hot_encode_instance(norm_data, norm_cf, cat_features)

        knn = nbrs.kneighbors(enc_cf.values, k, return_distance=False)[0]

        cf_label = round(cf_df[label].values[0])
        for idx in knn:
            inst = enc_data.iloc[idx]
            inst = inst.drop(index=label)
            inst = inst.values
            pred_inst = round(
                model(torch.from_numpy(inst).float()).detach().numpy().reshape(1)[0]
            )

            number_of_diff_labels += np.abs(cf_label - pred_inst)

    number_of_diff_labels = 1 - (1 / (N * k)) * number_of_diff_labels

    return number_of_diff_labels
