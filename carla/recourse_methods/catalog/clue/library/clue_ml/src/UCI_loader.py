from __future__ import division

import os
import urllib
import urllib.request
import zipfile

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from src.utils import mkdir


def load_UCI(dset_name, splits=10, seed=0, separate_targets=True, save_dir="data/"):
    mkdir(save_dir)

    if dset_name == "boston":
        if not os.path.isfile(save_dir + "housing.data"):
            urllib.request.urlretrieve(
                "https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data",
                filename=save_dir + "housing.data",
            )
        data = pd.read_csv(save_dir + "housing.data", header=0, delimiter="\s+").values
        y_idx = [-1]

    elif dset_name == "concrete":
        if not os.path.isfile(save_dir + "Concrete_Data.xls"):
            urllib.urlretrieve(
                "https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls",
                filename=save_dir + "Concrete_Data.xls",
            )
        data = pd.read_excel(
            save_dir + "Concrete_Data.xls", header=0, delimiter="\s+"
        ).values
        y_idx = [-1]

    elif dset_name == "energy":
        if not os.path.isfile(save_dir + "ENB2012_data.xlsx"):
            urllib.urlretrieve(
                "http://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx",
                filename=save_dir + "ENB2012_data.xlsx",
            )
        data = pd.read_excel(
            save_dir + "ENB2012_data.xlsx", header=0, delimiter="\s+"
        ).values
        y_idx = [-2, -1]

    elif dset_name == "power":
        if not os.path.isfile(save_dir + "CCPP.zip"):
            urllib.urlretrieve(
                "https://archive.ics.uci.edu/ml/machine-learning-databases/00294/CCPP.zip",
                filename=save_dir + "CCPP.zip",
            )
        zipped = zipfile.ZipFile(save_dir + "CCPP.zip")
        data = pd.read_excel(
            zipped.open("CCPP/Folds5x2_pp.xlsx"), header=0, delimiter="\t"
        ).values
        y_idx = [-1]

    elif dset_name == "wine":
        if not os.path.isfile(save_dir + "winequality-red.csv"):
            urllib.request.urlretrieve(
                "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",
                filename=save_dir + "winequality-red.csv",
            )
        data = pd.read_csv(
            save_dir + "winequality-red.csv", header=1, delimiter=";"
        ).values
        y_idx = [-1]

    elif dset_name == "yatch":
        if not os.path.isfile(save_dir + "yacht_hydrodynamics.data"):
            urllib.urlretrieve(
                "http://archive.ics.uci.edu/ml/machine-learning-databases/00243/yacht_hydrodynamics.data",
                filename=save_dir + "yacht_hydrodynamics.data",
            )
        data = pd.read_csv(
            save_dir + "yacht_hydrodynamics.data", header=1, delimiter="\s+"
        ).values
        y_idx = [-1]

    elif dset_name == "kin8nm":
        if not os.path.isfile(save_dir + "dataset_2175_kin8nm.csv"):
            urllib.urlretrieve(
                "https://www.openml.org/data/get_csv/3626/dataset_2175_kin8nm.csv",
                filename=save_dir + "dataset_2175_kin8nm.csv",
            )
        data = pd.read_csv(
            save_dir + "dataset_2175_kin8nm.csv", header=1, delimiter=","
        ).values
        y_idx = [-1]

    elif dset_name == "kin8nm":
        if not os.path.isfile(save_dir + "dataset_2175_kin8nm.csv"):
            urllib.urlretrieve(
                "https://www.openml.org/data/get_csv/3626/dataset_2175_kin8nm.csv",
                filename=save_dir + "dataset_2175_kin8nm.csv",
            )
        data = pd.read_csv(
            save_dir + "dataset_2175_kin8nm.csv", header=1, delimiter=","
        ).values
        y_idx = [-1]

    elif dset_name == "naval":
        if not os.path.isfile(save_dir + "UCI%20CBM%20Dataset.zip"):
            urllib.urlretrieve(
                "http://archive.ics.uci.edu/ml/machine-learning-databases/00316/UCI%20CBM%20Dataset.zip",
                filename=save_dir + "UCI%20CBM%20Dataset.zip",
            )
        zipped = zipfile.ZipFile(save_dir + "UCI%20CBM%20Dataset.zip")
        data = pd.read_csv(
            zipped.open("UCI CBM Dataset/data.txt"), header="infer", delimiter="\s+"
        ).values
        y_idx = [-2, -1]

    elif dset_name == "protein":
        if not os.path.isfile(save_dir + "CASP.csv"):
            urllib.urlretrieve(
                "https://archive.ics.uci.edu/ml/machine-learning-databases/00265/CASP.csv",
                filename=save_dir + "CASP.csv",
            )
        data = pd.read_csv(save_dir + "CASP.csv", header=1, delimiter=",").values
        y_idx = [0]

    elif dset_name == "default_credit":
        if not os.path.isfile(save_dir + "default of credit card clients.xls"):
            urllib.request.urlretrieve(
                "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls",
                filename=save_dir + "default of credit card clients.xls",
            )
        data = pd.read_excel(
            save_dir + "default of credit card clients.xls",
            header=[0, 1],
            index_col=0,
            delimiter="\s+",
        ).values
        y_idx = [-1]  # OK

    else:
        raise Exception("Dataset name doesnt match any known datasets.")

    np.random.seed(seed)
    data = data[np.random.permutation(np.arange(len(data)))]

    kf = KFold(n_splits=splits)
    for j, (train_index, test_index) in enumerate(kf.split(data)):

        if separate_targets:
            x_idx = list(range(data.shape[1]))
            for e in y_idx:
                x_idx.remove(x_idx[e])

            x_idx = np.array(x_idx)
            y_idx = np.array(y_idx)
            x_train, y_train = data[train_index, :], data[train_index, :]
            x_train, y_train = x_train[:, x_idx], y_train[:, y_idx]
            x_test, y_test = data[test_index, :], data[test_index, :]
            x_test, y_test = x_test[:, x_idx], y_test[:, y_idx]

            x_means, x_stds = x_train.mean(axis=0), x_train.std(axis=0)
            y_means, y_stds = y_train.mean(axis=0), y_train.std(axis=0)

            y_stds[y_stds < 1e-10] = 1
            x_stds[x_stds < 1e-10] = 1

            x_train = ((x_train - x_means) / x_stds).astype(np.float32)
            y_train = ((y_train - y_means) / y_stds).astype(np.float32)

            x_test = ((x_test - x_means) / x_stds).astype(np.float32)
            y_test = ((y_test - y_means) / y_stds).astype(np.float32)

            return x_train, x_test, x_means, x_stds, y_train, y_test, y_means, y_stds

        else:
            x_train, x_test = data[train_index, :], data[test_index, :]
            x_means, x_stds = x_train.mean(axis=0), x_train.std(axis=0)

            x_stds[x_stds < 1e-10] = 1

            x_train = ((x_train - x_means) / x_stds).astype(np.float32)
            x_test = ((x_test - x_means) / x_stds).astype(np.float32)

            return x_train, x_test, x_means, x_stds


def unnormalise_cat_vars(x, x_means, x_stds, input_dim_vec):
    input_dim_vec = np.array(input_dim_vec)
    unnorm_x = np.multiply(x, x_stds) + x_means

    fixed_unnorm = unnorm_x.round()
    fixed_unnorm -= fixed_unnorm.min(axis=0).reshape(
        [1, fixed_unnorm.shape[1]]
    )  # this sets all mins to 0
    for idx, dims in enumerate(input_dim_vec):
        if dims > 1:
            vec = fixed_unnorm[:, idx]
            vec[vec > dims - 1] = dims - 1
            fixed_unnorm[:, idx] = vec

    x[:, input_dim_vec > 1] = fixed_unnorm[:, input_dim_vec > 1]
    return x
