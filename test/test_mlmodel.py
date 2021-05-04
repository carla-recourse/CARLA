import numpy as np

from carla.data.catalog import DataCatalog
from carla.models.catalog import MLModelCatalog


def test_properties():
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

    model_tf_adult = MLModelCatalog(data, "ann", feature_input_order)

    exp_backend_tf = "tensorflow"
    exp_feature_order_adult = [
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

    assert model_tf_adult.backend == exp_backend_tf
    assert model_tf_adult.feature_input_order == exp_feature_order_adult


def test_predictions():
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

    model_tf_adult = MLModelCatalog(
        data, "ann", feature_input_order, encode_normalize_data=True
    )

    single_sample = data.encoded_normalized.iloc[22]
    single_sample = single_sample[model_tf_adult.feature_input_order].values.reshape(
        (1, -1)
    )
    samples = data.encoded_normalized.iloc[0:22]
    samples = samples[model_tf_adult.feature_input_order].values

    # Test single and bulk non probabilistic predictions
    single_prediction_tf = model_tf_adult.predict(single_sample)
    expected_shape = tuple((1,))
    assert single_prediction_tf.shape == expected_shape

    predictions_tf = model_tf_adult.predict(samples)
    expected_shape = tuple((22,))
    assert predictions_tf.shape == expected_shape

    # Test single and bulk probabilistic predictions
    single_predict_proba_tf = model_tf_adult.predict_proba(single_sample)
    expected_shape = tuple((1, 2))
    assert single_predict_proba_tf.shape == expected_shape

    predictions_proba_tf = model_tf_adult.predict_proba(samples)
    expected_shape = tuple((22, 2))
    assert predictions_proba_tf.shape == expected_shape


def test_predictions_with_pipeline():
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

    model_tf_adult = MLModelCatalog(
        data, "ann", feature_input_order, encode_normalize_data=True
    )
    model_tf_adult.use_pipeline = True

    single_sample = data.raw.iloc[22].to_frame().T
    samples = data.raw.iloc[0:22]

    # Test single and bulk non probabilistic predictions
    single_prediction_tf = model_tf_adult.predict(single_sample)
    expected_shape = tuple((1,))
    assert single_prediction_tf.shape == expected_shape

    predictions_tf = model_tf_adult.predict(samples)
    expected_shape = tuple((22,))
    assert predictions_tf.shape == expected_shape

    # Test single and bulk probabilistic predictions
    single_predict_proba_tf = model_tf_adult.predict_proba(single_sample)
    expected_shape = tuple((1, 2))
    assert single_predict_proba_tf.shape == expected_shape

    predictions_proba_tf = model_tf_adult.predict_proba(samples)
    expected_shape = tuple((22, 2))
    assert predictions_proba_tf.shape == expected_shape


def test_pipeline():
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

    model = MLModelCatalog(data, "ann", feature_input_order, use_pipeline=True)

    samples = data.raw.iloc[0:22]

    enc_norm_samples = model.perform_pipeline(samples)

    rows, cols = samples.shape
    expected_shape = (rows, cols - 1)

    assert expected_shape == enc_norm_samples.shape
    assert enc_norm_samples.select_dtypes(exclude=[np.number]).empty
