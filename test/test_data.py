import numpy as np
import pytest

from carla.data.catalog import DataCatalog
from carla.models.catalog import MLModelCatalog
from carla.models.pipelining import encode

testdata = ["adult", "give_me_some_credit", "compas", "heloc"]


@pytest.mark.parametrize("data_name", testdata)
def test_adult_col(data_name):
    data_catalog = DataCatalog(data_name)

    actual_col = (
        data_catalog.categorical + data_catalog.continuous + [data_catalog.target]
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
    norm[data.continuous] = mlmodel.scaler.transform(norm[data.continuous])

    col = data.continuous

    raw = data.raw[col]
    norm = norm[col]

    assert ((raw != norm).all()).any()


@pytest.mark.parametrize("data_name", testdata)
def test_adult_enc(data_name):
    data = DataCatalog(data_name)

    mlmodel = MLModelCatalog(data, "ann")

    cat = encode(mlmodel.encoder, data.categorical, data.raw)
    cat = cat[mlmodel.feature_input_order]

    assert cat.select_dtypes(exclude=[np.number]).empty
