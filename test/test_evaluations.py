import numpy as np
import pandas as pd

from carla.data.catalog import OnlineCatalog
from carla.evaluation import remove_nans
from carla.evaluation.catalog import distance
from carla.evaluation.catalog.success_rate import _success_rate
from carla.evaluation.catalog.violations import constraint_violation
from carla.recourse_methods.processing import get_drop_columns_binary


def test_l0():
    test_input_1 = np.array([[0, 1]])
    test_input_2 = np.array([[1, 0]])
    test_input_3 = np.concatenate((test_input_1, test_input_2), axis=0)

    delta = distance._get_delta(test_input_1, test_input_1)
    actual = distance.l0_distance(delta)
    expected = [0.0]

    assert actual == expected

    delta = distance._get_delta(test_input_1, test_input_2)
    actual = distance.l0_distance(delta)
    expected = [2.0]

    assert actual == expected

    expected = [1.0, 1.0]
    actual = distance.l0_distance(test_input_3)
    assert actual == expected


def test_d2():
    test_input_1 = np.array([[0, 0]])
    test_input_2 = np.array([[1, -1]])
    test_input_3 = np.concatenate((test_input_1, test_input_2), axis=0)

    expected = [0.0]
    actual = distance.l1_distance(test_input_1)
    assert actual == expected

    expected = [2.0]
    actual = distance.l1_distance(test_input_2)
    assert actual == expected

    expected = [0.0, 2.0]
    actual = distance.l1_distance(test_input_3)
    assert actual == expected


def test_d3():
    test_input_1 = np.array([[0, 0]])
    test_input_2 = np.array([[1, -1]])
    test_input_3 = np.concatenate((test_input_1, test_input_2), axis=0)

    expected = [0.0]
    actual = distance.l2_distance(test_input_1)
    assert actual == expected

    expected = [2.0]
    actual = distance.l2_distance(test_input_2)
    assert actual == expected

    expected = [0.0, 2.0]
    actual = distance.l2_distance(test_input_3)
    assert actual == expected


def test_d4():
    test_input_1 = np.array([[0, 0]])
    test_input_2 = np.array([[1, -4]])
    test_input_3 = np.concatenate((test_input_1, test_input_2), axis=0)

    expected = [0.0]
    actual = distance.linf_distance(test_input_1)
    assert actual == expected

    expected = [4.0]
    actual = distance.linf_distance(test_input_2)
    assert actual == expected

    expected = [0.0, 4.0]
    actual = distance.linf_distance(test_input_3)
    assert actual == expected


def test_distances():
    test_input_1 = np.array([[1, 0, 1, 0, 0, 1, 1, 0], [1, 0, 1, 0, 0, 1, 1, 0]])
    test_input_2 = np.array([[1, 0, 0, 0, 0, 1, 1, 0], [1, 0, 1, 0, 0, 1, 1, 0]])

    expected = [[1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]]
    actual = distance._get_distances(test_input_1, test_input_2)
    assert actual == expected


def test_success_rate():
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
    test_counterfactual = pd.DataFrame(
        test_counterfactual,
        columns=columns,
    )

    actual = _success_rate(test_counterfactual)
    expected = 1 / 3

    assert actual == expected


def test_constraint_violations():
    # Build data and mlmodel
    data_name = "adult"
    data = OnlineCatalog(data_name)

    # get factuals
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
            39,
            "Non-Private",
            77516,
            13,
            "Non-Married",
            "Managerial-Specialist",
            "Non-Husband",
            "White",
            "Male",
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
            38,
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
        [
            53,
            "Private",
            234721,
            7,
            "Married",
            "Other",
            "Husband",
            "Non-White",
            "Male",
            0,
            0,
            40,
            "US",
            0,
        ],
        [
            28,
            "Private",
            338409,
            13,
            "Married",
            "Managerial-Specialist",
            "Non-Husband",
            "Non-White",
            "Female",
            0,
            0,
            40,
            "Non-US",
            0,
        ],
    ]
    test_factual = pd.DataFrame(
        test_factual,
        columns=columns,
    )

    test_counterfactual = [
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
        [
            53,
            "Private",
            234721,
            7,
            "Married",
            "Other",
            "Husband",
            "Non-White",
            "Male",
            0,
            0,
            40,
            "US",
            0,
        ],
        [
            28,
            "Private",
            338409,
            13,
            "Married",
            "Managerial-Specialist",
            "Non-Husband",
            "Non-White",
            "Male",
            0,
            0,
            40,
            "Non-US",
            0,
        ],
    ]
    test_counterfactual = pd.DataFrame(
        test_counterfactual,
        columns=columns,
    )
    test_counterfactual = data.transform(test_counterfactual)
    test_factual = data.transform(test_factual)

    expected = [[2], [0], [1], [0], [1]]
    actual = constraint_violation(data, test_counterfactual, test_factual)

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
    actual_counterfactual, actual_factual = remove_nans(
        test_counterfactual, test_factual
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
    test_categorical = [
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

    actual = get_drop_columns_binary(test_categorical, test_columns)

    assert actual == expected
