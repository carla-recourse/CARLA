from __future__ import division

import os
import urllib.request

import numpy as np
import pandas as pd

# try:
#    import urllib2
# except:
#    import urllib3


def check_data_file(fdir):
    fname = os.path.join(fdir, "law_school_cf_test.csv")

    if not os.path.isfile(fname):
        print
        "'%s' not found! Downloading from GitHub..." % fname
        addr = "https://raw.githubusercontent.com/throwaway20190523/MonotonicFairness/master/data/law_school_cf_test.csv"

        # PYTHON 2 version
        # try:
        #    response = urllib2.urlopen(addr)
        # except:
        #    response = urllib3.urlopen(addr)

        # PTYHON 3 version
        response = urllib.request.urlopen(addr)
        data = response.read()
        data = data.decode()

        fileOut = open(fname, "w")
        fileOut.write(data)
        fileOut.close()
        print
        "'%s' download and saved locally.." % fname
    else:
        print
        "File found in current directory.."

    fname = os.path.join(fdir, "law_school_cf_train.csv")

    if not os.path.isfile(fname):
        print
        "'%s' not found! Downloading from GitHub..." % fname
        addr = "https://raw.githubusercontent.com/throwaway20190523/MonotonicFairness/master/data/law_school_cf_train.csv"

        # PYTHON 2 version
        # try:
        #    response = urllib2.urlopen(addr)
        # except:
        #    response = urllib3.urlopen(addr)

        # PYTHON 3 version
        response = urllib.request.urlopen(addr)
        data = response.read()
        data = data.decode()

        fileOut = open(fname, "w")
        fileOut.write(data)
        fileOut.close()
        print
        "'%s' download and saved locally.." % fname
    else:
        print
        "File found in current directory.."

    return None


def input_dim_vec_to_X_dims(input_dim_vec):
    """This is for our cat_Gauss VAE model"""
    X_dims = []
    for i in input_dim_vec:
        for ii in range(i):
            X_dims.append(i)
    return np.array(X_dims)


def get_my_LSAT(save_dir="../data/"):
    # The loaded files have a test set size of 0.2
    check_data_file(save_dir)

    train_file = save_dir + "law_school_cf_train.csv"
    test_file = save_dir + "law_school_cf_test.csv"

    df = pd.read_csv(train_file)
    data_train = df.to_dict("list")

    df = pd.read_csv(test_file)
    data_test = df.to_dict("list")

    keys = data_train.keys()
    print(keys)

    # I reorder the keys so they make a little bit more sense

    target_key = "ZFYA"
    # we are going to exclude
    my_data_keys = [
        "LSAT",
        "UGPA",
        "amerind",
        "mexican",
        "other",
        "black",
        "asian",
        "puerto",
        "hisp",
        "white",
        "female",
        "male",
    ]
    input_dim_vec = [1, 1, 8, 2]
    X_dims = input_dim_vec_to_X_dims(input_dim_vec)

    # I added list to keys (Python 2 version)
    """
    X_train = np.empty((len(data_train[keys[0]]), len(my_data_keys)))
    y_train = np.array(data_train[target_key]).reshape(-1, 1)
    X_test = np.empty((len(data_test[keys[0]]), len(my_data_keys)))
    y_test = np.array(data_test[target_key]).reshape(-1, 1)
    """

    # I added list to keys (Python 3 version)
    X_train = np.empty((len(data_train[list(keys)[0]]), len(my_data_keys)))
    y_train = np.array(data_train[target_key]).reshape(-1, 1)
    X_test = np.empty((len(data_test[list(keys)[0]]), len(my_data_keys)))
    y_test = np.array(data_test[target_key]).reshape(-1, 1)

    for k_idx, k in enumerate(my_data_keys):
        X_train[:, k_idx] = np.array(data_train[k])
        X_test[:, k_idx] = np.array(data_test[k])

    x_means, x_stds = X_train.mean(axis=0), X_train.std(axis=0)
    x_means[X_dims > 1] = 0
    x_stds[X_dims > 1] = 1
    x_stds[x_stds < 1e-10] = 1

    x_train = ((X_train - x_means) / x_stds).astype(np.float32)
    x_test = ((X_test - x_means) / x_stds).astype(np.float32)

    y_means, y_stds = y_train.mean(axis=0), y_train.std(axis=0)
    y_stds[y_stds < 1e-10] = 1

    y_train = ((y_train - y_means) / y_stds).astype(np.float32)
    y_test = ((y_test - y_means) / y_stds).astype(np.float32)

    return (
        x_train,
        x_test,
        x_means,
        x_stds,
        y_train,
        y_test,
        y_means,
        y_stds,
        my_data_keys,
        input_dim_vec,
    )


def join_LSAT_targets(x_train, x_test, y_train, y_test, input_dim_vec):
    input_dim_vec = np.append(input_dim_vec, 1)

    x_train = np.concatenate([x_train, y_train], axis=1)
    x_test = np.concatenate([x_test, y_test], axis=1)
    return x_train, x_test, input_dim_vec
