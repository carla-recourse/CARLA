import numpy as np
import pytest

from carla.data.catalog import DataCatalog
from carla.models.catalog import MLModelCatalog
from carla.models.pipelining import encode

testdata = ["adult", "give_me_some_credit", "compas"]


@pytest.mark.parametrize("data_name", testdata)
def test_adult_col(data_name):
    data_catalog = DataCatalog(data_name)

    actual_col = (
        data_catalog.categoricals + data_catalog.continous + [data_catalog.target]
    )
    actual_col = actual_col.sort()
    expected_col = data_catalog.raw.columns.values
    expected_col = expected_col.sort()

    assert actual_col == expected_col


@pytest.mark.parametrize("data_name", testdata)
def test_adult_norm(data_name):
    data = DataCatalog(data_name)

    mlmodel = MLModelCatalog(data, "ann")
    norm = data.raw
    norm[data.continous] = mlmodel.scaler.transform(norm[data.continous])

    col = data.continous

    raw = data.raw[col]
    norm = norm[col]

    assert ((raw != norm).all()).any()


@pytest.mark.parametrize("data_name", testdata)
def test_adult_enc(data_name):
    data = DataCatalog(data_name)

    mlmodel = MLModelCatalog(data, "ann")

    cat = encode(mlmodel.encoder, data.categoricals, data.raw)
    cat = cat[mlmodel.feature_input_order]

    assert cat.select_dtypes(exclude=[np.number]).empty
