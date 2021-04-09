from carla.data.catalog import DataCatalog
from carla.models.catalog import MLModelCatalog


def test_properties():
    data_name = "adult"
    model_tf_adult = MLModelCatalog(data_name, "ann")
    # TODO: Issue #16
    # model_pt_adult = MLModelCatalog(data_name, "ann", ext="pt")

    exp_backend_tf = "tensorflow"
    # TODO: Issue #16
    # exp_backend_pt = "pytorch"
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
    # TODO: Issue #16
    # assert model_pt_adult.backend == exp_backend_pt
    assert model_tf_adult.feature_input_order == exp_feature_order_adult
    # TODO: Issue #16
    # assert model_pt_adult.feature_input_order == exp_feature_order_adult


def test_predictions():
    data_name = "adult"
    data_catalog = "adult_catalog.yaml"
    data = DataCatalog(data_name, data_catalog)

    model_tf_adult = MLModelCatalog(data_name, "ann")
    # TODO: Issue #16
    # model_pt_adult = MLModelCatalog(data_name, "ann", ext="pt")

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

    # TODO: Issue #16
    # single_prediction_pt = model_pt_adult.predict(single_sample)
    # assert single_prediction_pt.shape == expected_shape

    predictions_tf = model_tf_adult.predict(samples)
    expected_shape = tuple((22,))
    assert predictions_tf.shape == expected_shape

    # TODO: Issue #16
    # predictions_pt = model_pt_adult.predict(samples)
    # assert predictions_pt.shape == expected_shape

    # Test single and bulk probabilistic predictions
    single_predict_proba_tf = model_tf_adult.predict_proba(single_sample)
    expected_shape = tuple((1, 2))
    assert single_predict_proba_tf.shape == expected_shape

    # TODO: Issue #16
    # single_predict_proba_pt = model_pt_adult.predict_proba(single_sample)
    # assert single_predict_proba_pt.shape == expected_shape

    predictions_proba_tf = model_tf_adult.predict_proba(samples)
    expected_shape = tuple((22, 2))
    assert predictions_proba_tf.shape == expected_shape

    # TODO: Issue #16
    # predictions_proba_pt = model_pt_adult.predict_proba(samples)
    # assert predictions_proba_pt.shape == expected_shape
