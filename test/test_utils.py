import numpy as np
import pandas as pd

from carla.evaluation import utils


def test_succes_rate():
    df = pd.DataFrame([1, 1, 1, np.nan, np.nan])
    actual_rate, actual_indices = utils.success_rate_and_indices(df)
    expected_rate = 0.6
    expected_indices = [0, 1, 2]

    assert actual_rate == expected_rate
    assert (actual_indices == expected_indices).all()
