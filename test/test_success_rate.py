import numpy as np
import pandas as pd

from carla.evaluation.success_rate import success_rate


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

    actual = success_rate(test_counterfactual)
    expected = 1 / 3

    assert actual == expected
