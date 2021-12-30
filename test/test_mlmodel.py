import numpy as np
import pytest
import torch
from pandas._testing import assert_frame_equal

from carla.data.catalog import DataCatalog
from carla.models.catalog import MLModelCatalog
from carla.models.pipelining import encode, scale

testmodel = ["ann", "linear"]
test_data = ["adult", "give_me_some_credit", "compas"]


def test_properties():
    data_name = "adult"
    data = DataCatalog(data_name)

    model_tf_adult = MLModelCatalog(data, "ann")

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


def test_inverse_pipeline():
    data_name = "adult"
    data = DataCatalog(data_name)

    model_tf_adult = MLModelCatalog(data, "ann")

    transformed_data = model_tf_adult.perform_pipeline(data.raw)
    detransformed_data = model_tf_adult.perform_inverse_pipeline(transformed_data)

    expected_data = data.raw[detransformed_data.columns]
    expected_data[data.continous] = expected_data[data.continous].astype("float")
    expected_data.columns = detransformed_data.columns

    assert_frame_equal(expected_data, detransformed_data)


@pytest.mark.parametrize("model_type", testmodel)
@pytest.mark.parametrize("data_name", test_data)
def test_predictions_tf(model_type, data_name):
    data = DataCatalog(data_name)

    model_tf_adult = MLModelCatalog(data, model_type)

    # normalize and encode data
    norm_enc_data = scale(model_tf_adult.scaler, data.continous, data.raw)
    norm_enc_data = encode(model_tf_adult.encoder, data.categoricals, norm_enc_data)
    norm_enc_data = norm_enc_data[model_tf_adult.feature_input_order]

    single_sample = norm_enc_data.iloc[22]
    single_sample = single_sample[model_tf_adult.feature_input_order].values.reshape(
        (1, -1)
    )
    samples = norm_enc_data.iloc[0:22]
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
    data = DataCatalog(data_name)

    model_tf_adult = MLModelCatalog(data, model_type)
    model_tf_adult.use_pipeline = True

    single_sample = data.raw.iloc[22].to_frame().T
    samples = data.raw.iloc[0:22]

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
def test_pipeline(model_type, data_name):
    data = DataCatalog(data_name)

    model = MLModelCatalog(data, model_type, use_pipeline=True)

    samples = data.raw.iloc[0:22]

    enc_norm_samples = model.perform_pipeline(samples)

    rows, cols = samples.shape
    expected_shape = (rows, cols - 1)

    assert expected_shape == enc_norm_samples.shape
    assert enc_norm_samples.select_dtypes(exclude=[np.number]).empty


@pytest.mark.parametrize("model_type", testmodel)
@pytest.mark.parametrize("data_name", test_data)
def test_predictions_pt(model_type, data_name):
    data = DataCatalog(data_name)
    model = MLModelCatalog(data, model_type, backend="pytorch")
    feature_input_order = model.feature_input_order

    # normalize and encode data
    norm_enc_data = scale(model.scaler, data.continous, data.raw)
    norm_enc_data = encode(model.encoder, data.categoricals, norm_enc_data)
    norm_enc_data = norm_enc_data[feature_input_order]

    single_sample = norm_enc_data.iloc[22]
    single_sample = single_sample[model.feature_input_order].values.reshape((1, -1))
    single_sample_torch = torch.Tensor(single_sample)

    samples = norm_enc_data.iloc[0:22]
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
