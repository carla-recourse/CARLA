import numpy as np
import pandas as pd
import torch

from ..models.catalog.catalog import MLModelCatalog


def repeated_elements(x):
    _size = len(x)
    repeated = []
    for i in range(_size):
        k = i + 1
        for j in range(k, _size):
            if x[i] == x[j] and x[i] not in repeated:
                repeated.append(x[i])
    return repeated


def constraint_violation(
    counterfactual: pd.DataFrame, instance: pd.DataFrame, immutable_list, separator="_"
):
    """
    :param counterfactual:
    :param instance:
    :param immutable_list:
    :param separator: String, determines the separator for one-hot-encoded columns
    :return: # number of constraint violations
    """
    names = counterfactual.columns.tolist()
    clean_names = []
    for i in names:
        clean_names.append(i.split(separator)[0])

    doubled_elements = repeated_elements(
        clean_names
    )  # if features are one-hot, names appear twice & we need to account
    accounting_factor = list(set(immutable_list) & set(doubled_elements))

    counterfactual.columns = clean_names
    instance.columns = clean_names

    logical = np.round(counterfactual[immutable_list].values, decimals=2) != np.round(
        instance[immutable_list].values, decimals=2
    )
    violations = (1 * logical).sum()

    if violations == 0:
        return violations
    else:
        if doubled_elements is None:
            return violations
        else:
            return violations - len(accounting_factor)


def success_rate_and_indices(counterfactuals: pd.DataFrame):
    """
    Used to indicate which counterfactuals should be dropped (due to lack of success indicated by NaN).
    Also computes percent of successfully found counterfactuals
    :param counterfactuals: pd df, where NaNs indicate 'no counterfactual found' [df should contain no object values)
    :return: success_rate, indices
    """

    # Success rate & drop not successful counterfactuals & process remainder
    success_rate = (counterfactuals.dropna().shape[0]) / counterfactuals.shape[0]
    counterfactual_indices = np.where(
        # np.any(np.isnan(counterfactuals.values) == True, axis=1) == False
        not np.any(np.isnan(counterfactuals.values), axis=1)
    )[0]

    return success_rate, counterfactual_indices


def redundancy(factual: np.array, counterfactual: np.array, ml_model: MLModelCatalog):
    """
    Redundency metric which looks for 'none-essential' changes
    :param factual: np.array with original instance
    :param counterfactual: np.array with counterfactual instance
    :param model: pytorch model
    :return: scalar, number of unnecessary changes
    """
    red = 0

    # get model prediction and cast it from tensor to float
    # TODO replace backend with boolean, to avoid errors in string which would fail the condition
    if ml_model.backend == "pytorch":
        pred_f = round(
            ml_model.predict(torch.from_numpy(factual).float())
            .detach()
            .numpy()
            .squeeze()
            .reshape(1)[0]
        )
        pred_cf = round(
            ml_model.predict(torch.from_numpy(counterfactual).float())
            .detach()
            .numpy()
            .squeeze()
            .reshape(1)[0]
        )
    elif ml_model.backend == "tensorflow":
        pred_f = ml_model.predict(factual)
        pred_f = np.argmax(pred_f, axis=1)
        pred_cf = ml_model.predict(counterfactual)
        pred_cf = np.argmax(pred_cf, axis=1)
    else:
        raise NotImplementedError()

    if pred_f != pred_cf:
        for i in range(factual.shape[1]):
            if factual[0][i] != counterfactual[0][i]:
                temp_cf = np.copy(counterfactual)

                # reverse change in counterfactual and predict new label
                temp_cf[0][i] = factual[0][i]
                if ml_model.backend == "pytorch":
                    pred_temp_cf = round(
                        ml_model.predict(torch.from_numpy(temp_cf).float())
                        .detach()
                        .numpy()
                        .squeeze()
                        .reshape(1)[0]
                    )
                elif ml_model.backend == "tensorflow":
                    pred_temp_cf = ml_model.predict(temp_cf)
                    pred_temp_cf = np.argmax(pred_temp_cf, axis=1)
                else:
                    raise NotImplementedError()

                # if new prediction has the same label as the old prediction for cf, increase redundancy
                if pred_temp_cf == pred_cf:
                    red += 1

    else:
        print("Factual and counterfactual are in the same class")
        return np.nan

    return red
