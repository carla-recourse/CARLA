import numpy as np
import pandas as pd

from carla.data.catalog import DataCatalog
from carla.evaluation import Benchmark, remove_nans
from carla.models.catalog import MLModelCatalog
from carla.models.negative_instances import predict_negative_instances
from carla.recourse_methods.catalog.dice import Dice
from carla.recourse_methods.processing import get_drop_columns_binary


def test_benchmarks():
    # Build data and mlmodel
    data_name = "adult"
    data = DataCatalog(data_name)

    model_tf = MLModelCatalog(data, "ann")
    # get factuals
    factuals = predict_negative_instances(model_tf, data)

    hyperparams = {"num": 1, "desired_class": 1}
    # Pipeline needed for dice, but not for predicting negative instances
    model_tf.use_pipeline = True
    test_factual = factuals.iloc[:5]

    dice = Dice(model_tf, hyperparams)

    benchmark = Benchmark(model_tf, dice, test_factual)
    df_benchmark = benchmark.run_benchmark()

    expected = (5, 9)
    actual = df_benchmark.shape
    assert expected == actual


def test_ynn():
    # Build data and mlmodel
    data_name = "adult"
    data = DataCatalog(data_name)

    model_tf = MLModelCatalog(data, "ann")
    # get factuals
    factuals = predict_negative_instances(model_tf, data)

    hyperparams = {"num": 1, "desired_class": 1}
    # Pipeline needed for dice, but not for predicting negative instances
    model_tf.use_pipeline = True
    test_factual = factuals.iloc[:5]

    dice = Dice(model_tf, hyperparams)

    benchmark = Benchmark(model_tf, dice, test_factual)
    yNN = benchmark.compute_ynn()

    expected = (1, 1)
    actual = yNN.shape
    assert expected == actual


def test_time():
    # Build data and mlmodel
    data_name = "adult"
    data = DataCatalog(data_name)

    model_tf = MLModelCatalog(data, "ann")
    # get factuals
    factuals = predict_negative_instances(model_tf, data)

    hyperparams = {"num": 1, "desired_class": 1}
    # Pipeline needed for dice, but not for predicting negative instances
    model_tf.use_pipeline = True
    test_factual = factuals.iloc[:5]

    dice = Dice(model_tf, hyperparams)

    benchmark = Benchmark(model_tf, dice, test_factual)
    df_time = benchmark.compute_average_time()

    expected = (1, 1)
    actual = df_time.shape
    assert expected == actual


def test_distances():
    # Build data and mlmodel
    data_name = "adult"
    data = DataCatalog(data_name)

    model_tf = MLModelCatalog(data, "ann")
    # get factuals
    factuals = predict_negative_instances(model_tf, data)

    hyperparams = {"num": 1, "desired_class": 1}
    # Pipeline needed for dice, but not for predicting negative instances
    model_tf.use_pipeline = True
    test_factual = factuals.iloc[:5]

    dice = Dice(model_tf, hyperparams)

    benchmark = Benchmark(model_tf, dice, test_factual)
    df_distances = benchmark.compute_distances()

    expected = (5, 4)
    actual = df_distances.shape
    assert expected == actual


def test_drop_binary():
    test_columns = [
        "workclass_Non-Private",
        "workclass_Private",
        "marital-status_Married",
        "marital-status_Non-Married",
        "occupation_Managerial-Specialist",
        "occupation_Other",
        "relationship_Husband",
        "relationship_Non-Husband",
        "race_Non-White",
        "race_White",
        "sex_Female",
        "sex_Male",
        "native-country_Non-US",
        "native-country_US",
    ]
    test_categoricals = [
        "workclass",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    expected = [
        "workclass_Non-Private",
        "marital-status_Married",
        "occupation_Managerial-Specialist",
        "relationship_Husband",
        "race_Non-White",
        "sex_Female",
        "native-country_Non-US",
    ]

    actual = get_drop_columns_binary(test_categoricals, test_columns)

    assert actual == expected


def test_success_rate():
    # Build data and mlmodel
    data_name = "adult"
    data = DataCatalog(data_name)

    model_tf = MLModelCatalog(data, "ann")
    # get factuals
    factuals = predict_negative_instances(model_tf, data)

    hyperparams = {"num": 1, "desired_class": 1}
    # Pipeline needed for dice, but not for predicting negative instances
    model_tf.use_pipeline = True
    test_factual = factuals.iloc[:5]

    dice = Dice(model_tf, hyperparams)

    benchmark = Benchmark(model_tf, dice, test_factual)
    rate = benchmark.compute_success_rate()

    expected = (1, 1)
    actual = rate.shape
    assert expected == actual


def test_redundancy():
    # Build data and mlmodel
    data_name = "adult"
    data = DataCatalog(data_name)

    model_tf = MLModelCatalog(data, "ann")
    # get factuals
    factuals = predict_negative_instances(model_tf, data)

    hyperparams = {"num": 1, "desired_class": 1}
    # Pipeline needed for dice, but not for predicting negative instances
    model_tf.use_pipeline = True
    test_factual = factuals.iloc[:5]

    dice = Dice(model_tf, hyperparams)

    benchmark = Benchmark(model_tf, dice, test_factual)
    df_redundancy = benchmark.compute_redundancy()

    expected = (5, 1)
    actual = df_redundancy.shape

    assert expected == actual


def test_violation():
    # Build data and mlmodel
    data_name = "adult"
    data = DataCatalog(data_name)

    model_tf = MLModelCatalog(data, "ann")
    # get factuals
    factuals = predict_negative_instances(model_tf, data)

    hyperparams = {"num": 1, "desired_class": 1}
    # Pipeline needed for dice, but not for predicting negative instances
    model_tf.use_pipeline = True
    test_factual = factuals.iloc[:5]

    dice = Dice(model_tf, hyperparams)

    benchmark = Benchmark(model_tf, dice, test_factual)
    df_violation = benchmark.compute_constraint_violation()

    expected = (5, 1)
    actual = df_violation.shape

    assert expected == actual


def test_removing_nans():
    columns = [
        "age",
        "workclass",
        "fnlwgt",
        "education-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country",
        "income",
    ]
    test_factual = [
        [
            45,
            "Non-Private",
            77516,
            13,
            "Non-Married",
            "Managerial-Specialist",
            "Non-Husband",
            "White",
            "Female",
            2174,
            0,
            40,
            "US",
            0,
        ],
        [
            50,
            "Non-Private",
            83311,
            13,
            "Married",
            "Managerial-Specialist",
            "Husband",
            "White",
            "Male",
            0,
            0,
            13,
            "US",
            0,
        ],
        [
            18,
            "Private",
            215646,
            9,
            "Non-Married",
            "Other",
            "Non-Husband",
            "White",
            "Male",
            0,
            0,
            40,
            "US",
            0,
        ],
    ]
    test_counterfactual = [
        [
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
        ],
        [
            50,
            "Non-Private",
            83311,
            13,
            "Married",
            "Managerial-Specialist",
            "Husband",
            "White",
            "Male",
            0,
            0,
            13,
            "US",
            0,
        ],
        [
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
        ],
    ]
    test_factual = pd.DataFrame(
        test_factual,
        columns=columns,
    )
    test_counterfactual = pd.DataFrame(
        test_counterfactual,
        columns=columns,
    )
    actual_factual, actual_counterfactual = remove_nans(
        test_factual, test_counterfactual
    )

    expected = [
        [
            50,
            "Non-Private",
            83311,
            13,
            "Married",
            "Managerial-Specialist",
            "Husband",
            "White",
            "Male",
            0,
            0,
            13,
            "US",
            0,
        ],
    ]

    assert actual_factual.values.tolist() == expected
    assert actual_counterfactual.values.tolist() == expected
