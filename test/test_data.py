import numpy as np

from carla.data.catalog import DataCatalog
from carla.models.catalog import MLModelCatalog
from carla.models.pipelining import encode


def test_adult_col():
    data_name = "adult"
    data_catalog = "adult_catalog.yaml"
    data_catalog = DataCatalog(data_name, data_catalog)

    actual_col = (
        data_catalog.categoricals + data_catalog.continous + [data_catalog.target]
    )
    actual_col = actual_col.sort()
    expected_col = data_catalog.raw.columns.values
    expected_col = expected_col.sort()

    assert actual_col == expected_col


def test_adult_norm():
    data_name = "adult"
    data_catalog = "adult_catalog.yaml"
    data = DataCatalog(data_name, data_catalog)

    mlmodel = MLModelCatalog(data, "ann")
    norm = data.raw
    norm[data.continous] = mlmodel.scaler.transform(norm[data.continous])

    col = data.continous

    raw = data.raw[col]
    norm = norm[col]

    assert ((raw != norm).all()).any()


def test_adult_enc():
    data_name = "adult"
    data_catalog = "adult_catalog.yaml"
    data = DataCatalog(data_name, data_catalog)

    mlmodel = MLModelCatalog(data, "ann")

    cat = encode(mlmodel.encoder, data.categoricals, data.raw)
    cat = cat[mlmodel.feature_input_order]

    assert cat.select_dtypes(exclude=[np.number]).empty
