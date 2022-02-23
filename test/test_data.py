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


# TODO these tests probably can be deleted as test_transform should test the same
# @pytest.mark.parametrize("data_name", testdata)
# def test_adult_norm(data_name):
#     data = OnlineCatalog(data_name)
#     mlmodel = MLModelCatalog(data, "ann")
#
#     norm = data.df
#     norm[data.continuous] = mlmodel.data.scaler.transform(norm[data.continuous])
#
#     col = data.continuous
#
#     raw = data.df[col]
#     norm = norm[col]
#
#     assert ((raw != norm).all()).any()


# @pytest.mark.parametrize("data_name", testdata)
# def test_adult_enc(data_name):
#     data = OnlineCatalog(data_name)
#     mlmodel = MLModelCatalog(data, "ann")
#
#     cat = encode(mlmodel.data.encoder, data.categorical, data.df)
#     cat = cat[mlmodel.feature_input_order]
#
#     assert cat.select_dtypes(exclude=[np.number]).empty
