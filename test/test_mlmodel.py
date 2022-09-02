import numpy as np
import pytest
import torch

from carla.data.catalog import OnlineCatalog
from carla.models.catalog import MLModelCatalog

testmodel = ["ann", "linear"]
test_data = ["adult", "give_me_some_credit", "compas", "heloc"]


def test_properties():
    data_name = "adult"
    data = OnlineCatalog(data_name)

    model_tf_adult = MLModelCatalog(data, "ann", backend="tensorflow")

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


def test_forest_properties():
    data_name = "adult"
    data = OnlineCatalog(data_name)

    model = MLModelCatalog(data, "forest", backend="sklearn")

    assert model is not None


@pytest.mark.parametrize("model_type", testmodel)
@pytest.mark.parametrize("data_name", test_data)
def test_predictions_tf(model_type, data_name):
    data = OnlineCatalog(data_name)

    model_tf_adult = MLModelCatalog(data, model_type, backend="tensorflow")

    single_sample = data.df.iloc[22]
    single_sample = single_sample[model_tf_adult.feature_input_order].values.reshape(
        (1, -1)
    )
    samples = data.df.iloc[0:22]
    samples = samples[model_tf_adult.feature_input_order].values

    # Test single and bulk non probabilistic predictions
    single_prediction_tf = model_tf_adult.predict(single_sample)
    expected_shape = tuple((1, 1))
    assert single_prediction_tf.shape == expected_shape

    predictions_tf = model_tf_adult.predict(samples)
    expected_shape = tuple((22, 1))
    assert predictions_tf.shape == expected_shape

    # Test single and bulk probabilistic predictions
    single_predict_proba_tf = model_tf_adult.predict_proba(single_sample)
    expected_shape = tuple((1, 2))
    assert single_predict_proba_tf.shape == expected_shape

    predictions_proba_tf = model_tf_adult.predict_proba(samples)
    expected_shape = tuple((22, 2))
    assert predictions_proba_tf.shape == expected_shape


@pytest.mark.parametrize("model_type", testmodel)
@pytest.mark.parametrize("data_name", test_data)
def test_predictions_with_pipeline(model_type, data_name):
    data = OnlineCatalog(data_name)

    model_tf_adult = MLModelCatalog(data, model_type, backend="tensorflow")
    model_tf_adult.use_pipeline = True

    single_sample = data.df.iloc[22].to_frame().T
    samples = data.df.iloc[0:22]

    # Test single and bulk non probabilistic predictions
    single_prediction_tf = model_tf_adult.predict(single_sample)
    expected_shape = tuple((1, 1))
    assert single_prediction_tf.shape == expected_shape

    predictions_tf = model_tf_adult.predict(samples)
    expected_shape = tuple((22, 1))
    assert predictions_tf.shape == expected_shape

    # Test single and bulk probabilistic predictions
    single_predict_proba_tf = model_tf_adult.predict_proba(single_sample)
    expected_shape = tuple((1, 2))
    assert single_predict_proba_tf.shape == expected_shape

    predictions_proba_tf = model_tf_adult.predict_proba(samples)
    expected_shape = tuple((22, 2))
    assert predictions_proba_tf.shape == expected_shape


# TODO add parametrize backend
@pytest.mark.parametrize("model_type", testmodel)
@pytest.mark.parametrize("data_name", test_data)
def test_predictions_pt(model_type, data_name):
    data = OnlineCatalog(data_name)
    model = MLModelCatalog(data, model_type, backend="pytorch")

    single_sample = data.df.iloc[22]
    single_sample = single_sample[model.feature_input_order].values.reshape((1, -1))
    single_sample_torch = torch.Tensor(single_sample)

    samples = data.df.iloc[0:22]
    samples = samples[model.feature_input_order].values
    samples_torch = torch.Tensor(samples)

    # Test single non probabilistic predictions
    single_prediction = model.predict(single_sample)
    expected_shape = tuple((1, 1))
    assert single_prediction.shape == expected_shape
    assert isinstance(single_prediction, np.ndarray)

    single_prediction_torch = model.predict(single_sample_torch)
    expected_shape = tuple((1, 1))
    assert single_prediction_torch.shape == expected_shape
    assert torch.is_tensor(single_prediction_torch)

    # bulk non probabilistic predictions
    predictions = model.predict(samples)
    expected_shape = tuple((22, 1))
    assert predictions.shape == expected_shape
    assert isinstance(predictions, np.ndarray)

    predictions_torch = model.predict(samples_torch)
    expected_shape = tuple((22, 1))
    assert predictions_torch.shape == expected_shape
    assert torch.is_tensor(predictions_torch)

    # Test single probabilistic predictions
    single_predict_proba = model.predict_proba(single_sample)
    expected_shape = tuple((1, 2))
    assert single_predict_proba.shape == expected_shape
    assert isinstance(single_predict_proba, np.ndarray)

    single_predict_proba_torch = model.predict_proba(single_sample_torch)
    expected_shape = tuple((1, 2))
    assert single_predict_proba_torch.shape == expected_shape
    assert torch.is_tensor(single_predict_proba_torch)

    # bulk probabilistic predictions
    predictions_proba = model.predict_proba(samples)
    expected_shape = tuple((22, 2))
    assert predictions_proba.shape == expected_shape
    assert isinstance(single_predict_proba, np.ndarray)

    predictions_proba_torch = model.predict_proba(samples_torch)
    expected_shape = tuple((22, 2))
    assert predictions_proba_torch.shape == expected_shape
    assert torch.is_tensor(predictions_proba_torch)
