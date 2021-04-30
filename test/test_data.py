import numpy as np

from carla.data.catalog import DataCatalog
from carla.models.catalog import MLModelCatalog


def test_adult_col():
    data_name = "adult"
    data_catalog = "adult_catalog.yaml"
    data_catalog = DataCatalog(data_name, data_catalog, drop_first_encoding=True)

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
    data = DataCatalog(data_name, data_catalog, drop_first_encoding=True)

    feature_input_order = [
        "age",
        "fnlwgt",
        "education-num",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "workclass_Private",
        "marital-status_Non-Married",
        "occupation_Other",
        "relationship_Non-Husband",
        "race_White",
        "sex_Male",
        "native-country_US",
    ]

    mlmodel = MLModelCatalog(data, "ann", feature_input_order)
    data.set_normalized(mlmodel)

    col = data.continous

    raw = data.raw[col]
    norm = data.normalized[col]

    assert ((raw != norm).all()).any()


def test_adult_enc():
    data_name = "adult"
    data_catalog = "adult_catalog.yaml"
    data = DataCatalog(data_name, data_catalog, drop_first_encoding=True)

    feature_input_order = [
        "age",
        "fnlwgt",
        "education-num",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "workclass_Private",
        "marital-status_Non-Married",
        "occupation_Other",
        "relationship_Non-Husband",
        "race_White",
        "sex_Male",
        "native-country_US",
    ]

    mlmodel = MLModelCatalog(data, "ann", feature_input_order)
    data.set_encoded(mlmodel)

    cat = data.encoded

    assert cat.select_dtypes(exclude=[np.number]).empty

    data = DataCatalog(data_name, data_catalog, drop_first_encoding=False)

    feature_input_order = [
        "age",
        "age",
        "fnlwgt",
        "education-num",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "sex_Female",
        "sex_Male",
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
        "native-country_Non-US",
        "native-country_US",
    ]

    mlmodel = MLModelCatalog(data, "ann", feature_input_order)
    data.set_encoded(mlmodel)
    cat = data.encoded
    assert cat.select_dtypes(exclude=[np.number]).empty


def test_adult_norm_enc():
    data_name = "adult"
    data_catalog = "adult_catalog.yaml"
    data = DataCatalog(data_name, data_catalog, drop_first_encoding=True)

    feature_input_order = [
        "age",
        "fnlwgt",
        "education-num",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "workclass_Private",
        "marital-status_Non-Married",
        "occupation_Other",
        "relationship_Non-Husband",
        "race_White",
        "sex_Male",
        "native-country_US",
    ]

    mlmodel = MLModelCatalog(data, "ann", feature_input_order)
    data.set_encoded_normalized(mlmodel)

    norm_col = data.continous
    norm_enc_col = data.encoded_normalized.columns

    cat = data.encoded
    cat[norm_col] = data.normalized[norm_col]
    cat = cat[norm_enc_col]

    cat_norm = data.encoded_normalized

    assert ((cat_norm == cat).all()).all()
