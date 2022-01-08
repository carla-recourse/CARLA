import numpy as np
import pytest

from carla.data.catalog import DataCatalog
from carla.models.catalog import MLModelCatalog

testmodel = ["ann", "linear"]
test_data = ["adult", "give_me_some_credit", "compas"]

training_params_linear = {
    "adult": {"lr": 0.002, "epochs": 100, "batch_size": 2048},
    "compas": {"lr": 0.002, "epochs": 25, "batch_size": 128},
    "give_me_some_credit": {"lr": 0.002, "epochs": 10, "batch_size": 2048},
}
training_params_ann = {
    "adult": {"lr": 0.002, "epochs": 10, "batch_size": 1024},
    "compas": {"lr": 0.002, "epochs": 25, "batch_size": 25},
    "give_me_some_credit": {"lr": 0.002, "epochs": 10, "batch_size": 2048},
}
training_params = {"linear": training_params_linear, "ann": training_params_ann}


def test_properties():
    data_name = "adult"
    data = DataCatalog(data_name)

    model_type = "linear"
    model_tf_adult = MLModelCatalog(
        data, model_type, load_online=False, use_pipeline=True
    )
    params = training_params[model_type][data_name]
    model_tf_adult.train(
        learning_rate=params["lr"],
        epochs=params["epochs"],
        batch_size=params["batch_size"],
    )

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


@pytest.mark.parametrize("model_type", testmodel)
@pytest.mark.parametrize("data_name", test_data)
def test_predictions_tf(model_type, data_name):

    data = DataCatalog(data_name)

    model_tf_adult = MLModelCatalog(
        data, model_type, load_online=False, use_pipeline=True
    )
    params = training_params[model_type][data_name]
    model_tf_adult.train(
        learning_rate=params["lr"],
        epochs=params["epochs"],
        batch_size=params["batch_size"],
    )

    single_sample = data.raw.iloc[[22]]
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
def test_predictions_pt(model_type, data_name):
    data = DataCatalog(data_name)
    model = MLModelCatalog(
        data, model_type, load_online=False, use_pipeline=True, backend="pytorch"
    )
    params = training_params[model_type][data_name]
    model.train(
        learning_rate=params["lr"],
        epochs=params["epochs"],
        batch_size=params["batch_size"],
    )

    single_sample = data.raw.iloc[[22]]
    samples = data.raw.iloc[0:22]

    # Test single non probabilistic predictions
    single_prediction = model.predict(single_sample)
    expected_shape = tuple((1, 1))
    assert single_prediction.shape == expected_shape
    assert isinstance(single_prediction, np.ndarray)

    # bulk non probabilistic predictions
    predictions = model.predict(samples)
    expected_shape = tuple((22, 1))
    assert predictions.shape == expected_shape
    assert isinstance(predictions, np.ndarray)

    # Test single probabilistic predictions
    single_predict_proba = model.predict_proba(single_sample)
    expected_shape = tuple((1, 2))
    assert single_predict_proba.shape == expected_shape
    assert isinstance(single_predict_proba, np.ndarray)

    # bulk probabilistic predictions
    predictions_proba = model.predict_proba(samples)
    expected_shape = tuple((22, 2))
    assert predictions_proba.shape == expected_shape
    assert isinstance(single_predict_proba, np.ndarray)
