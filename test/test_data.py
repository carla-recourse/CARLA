import pytest
from pandas._testing import assert_frame_equal
from sklearn import preprocessing

from carla.data.catalog import OnlineCatalog

testdata = ["adult", "give_me_some_credit", "compas", "heloc"]


@pytest.mark.parametrize("data_name", testdata)
def test_adult_col(data_name):
    data_catalog = OnlineCatalog(data_name)

    actual_col = (
        data_catalog.categorical + data_catalog.continuous + [data_catalog.target]
    )
    actual_col = actual_col.sort()
    expected_col = data_catalog.df.columns.values
    expected_col = expected_col.sort()

    assert actual_col == expected_col


def test_transform():
    data_name = "adult"
    raw_data = OnlineCatalog(
        data_name, scaling_method="Identity", encoding_method="Identity"
    )
    transformed_data_str = OnlineCatalog(
        data_name, scaling_method="MinMax", encoding_method="OneHot_drop_binary"
    )

    encoding_method = preprocessing.OneHotEncoder(handle_unknown="error", sparse=False)
    transformed_data_fn = OnlineCatalog(
        data_name, scaling_method="MinMax", encoding_method=encoding_method
    )
    # sort columns as order could be different
    assert_frame_equal(
        transformed_data_str.inverse_transform(transformed_data_str.df).sort_index(
            axis=1
        ),
        raw_data.df.sort_index(axis=1),
        check_dtype=False,
    )
    assert_frame_equal(
        transformed_data_str.transform(raw_data.df).sort_index(axis=1),
        transformed_data_str.df.sort_index(axis=1),
        check_dtype=False,
    )
    assert_frame_equal(
        transformed_data_str.transform(
            transformed_data_str.inverse_transform(transformed_data_str.df)
        ).sort_index(axis=1),
        transformed_data_str.df.sort_index(axis=1),
        check_dtype=False,
    )
    assert_frame_equal(
        transformed_data_str.inverse_transform(
            transformed_data_str.transform(raw_data.df)
        ).sort_index(axis=1),
        raw_data.df.sort_index(axis=1),
        check_dtype=False,
    )
    # check whether encoding with string gives the same as encoding with a function
    assert_frame_equal(
        transformed_data_str.inverse_transform(transformed_data_str.df).sort_index(
            axis=1
        ),
        transformed_data_fn.inverse_transform(transformed_data_fn.df).sort_index(
            axis=1
        ),
        check_dtype=False,
    )
