import pytest

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
