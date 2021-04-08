from carla.data.catalog import DataCatalog
import numpy as np


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
    data_catalog = DataCatalog(data_name, data_catalog)

    col = data_catalog.continous

    raw = data_catalog.raw[col]
    norm = data_catalog.normalized[col]

    assert ((raw != norm).all()).any()


def test_adult_enc():
    data_name = "adult"
    data_catalog = "adult_catalog.yaml"
    data_catalog = DataCatalog(data_name, data_catalog)
    cat = data_catalog.encoded

    assert cat.select_dtypes(exclude=[np.number]).empty


def test_adult_norm_enc():
    data_name = "adult"
    data_catalog = "adult_catalog.yaml"
    data_catalog = DataCatalog(data_name, data_catalog)

    norm_col = data_catalog.continous
    norm_enc_col = data_catalog.encoded_normalized.columns

    cat = data_catalog.encoded.copy()
    cat[norm_col] = data_catalog.normalized[norm_col]
    cat = cat[norm_enc_col]

    cat_norm = data_catalog.encoded_normalized

    assert ((cat_norm == cat).all()).all()
