import numpy as np
import pytest

from carla.data.catalog import OnlineCatalog
from carla.models.catalog import MLModelCatalog

testmodel = ["ann", "linear"]
test_data = ["adult", "give_me_some_credit", "compas", "heloc"]

training_params_linear = {
    "adult": {"lr": 0.002, "epochs": 100, "batch_size": 2048},
    "compas": {"lr": 0.002, "epochs": 25, "batch_size": 128},
    "give_me_some_credit": {"lr": 0.002, "epochs": 10, "batch_size": 2048},
    "heloc": {"lr": 0.002, "epochs": 25, "batch_size": 128},
}
training_params_ann = {
    "adult": {"lr": 0.002, "epochs": 10, "batch_size": 1024},
    "compas": {"lr": 0.002, "epochs": 25, "batch_size": 25},
    "give_me_some_credit": {"lr": 0.002, "epochs": 10, "batch_size": 2048},
    "heloc": {"lr": 0.002, "epochs": 25, "batch_size": 25},
}
training_params_forest = {
    "adult": {"max_depth": 2, "n_estimators": 5},
    "compas": {"max_depth": 2, "n_estimators": 5},
    "give_me_some_credit": {"max_depth": 2, "n_estimators": 5},
    "heloc": {"max_depth": 2, "n_estimators": 5},
}
training_params = {
    "linear": training_params_linear,
    "ann": training_params_ann,
    "forest": training_params_forest,
}


def _train_model(data, model_type, backend):
    model = MLModelCatalog(data, model_type, load_online=False, backend=backend)
    params = training_params[model_type][data.name]
    if model_type == "forest":
        model.train(
            force_train=True,
            max_depth=params["max_depth"],
            n_estimators=params["n_estimators"],
        )
    else:
        model.train(
            force_train=True,
            learning_rate=params["lr"],
            epochs=params["epochs"],
            batch_size=params["batch_size"],
        )
    return model


def test_properties():
    data_name = "adult"
    model_type = "linear"
    data = OnlineCatalog(data_name)
    model_tf_adult = _train_model(data, model_type, backend="tensorflow")

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


@pytest.mark.parametrize("backend", ["sklearn", "xgboost"])
def test_forest(backend):
    data_name = "give_me_some_credit"
    model_type = "forest"

    data = OnlineCatalog(data_name)
    model = _train_model(data, model_type, backend=backend)

    single_sample = data.df.iloc[[22]]
    samples = data.df.iloc[0:22]

    # Test single and bulk non-probabilistic predictions
    single_prediction = model.predict(single_sample)
    expected_shape = tuple((1,))
    assert single_prediction.shape == expected_shape

    predictions = model.predict(samples)
    expected_shape = tuple((22,))
    assert predictions.shape == expected_shape

    # Test single and bulk probabilistic predictions
    single_predict_proba = model.predict_proba(single_sample)
    expected_shape = tuple((1, 2))
    assert single_predict_proba.shape == expected_shape

    predictions_proba = model.predict_proba(samples)
    expected_shape = tuple((22, 2))
    assert predictions_proba.shape == expected_shape


@pytest.mark.parametrize("model_type", testmodel)
@pytest.mark.parametrize("data_name", test_data)
def test_predictions_tf(model_type, data_name):
    data = OnlineCatalog(data_name)
    model = _train_model(data, model_type, backend="tensorflow")

    single_sample = data.df.iloc[[22]]
    samples = data.df.iloc[0:22]

    # Test single and bulk non-probabilistic predictions
    single_prediction_tf = model.predict(single_sample)
    expected_shape = tuple((1, 1))
    assert single_prediction_tf.shape == expected_shape

    predictions_tf = model.predict(samples)
    expected_shape = tuple((22, 1))
    assert predictions_tf.shape == expected_shape

    # Test single and bulk probabilistic predictions
    single_predict_proba_tf = model.predict_proba(single_sample)
    expected_shape = tuple((1, 2))
    assert single_predict_proba_tf.shape == expected_shape

    predictions_proba_tf = model.predict_proba(samples)
    expected_shape = tuple((22, 2))
    assert predictions_proba_tf.shape == expected_shape


@pytest.mark.parametrize("model_type", testmodel)
@pytest.mark.parametrize("data_name", test_data)
def test_predictions_pt(model_type, data_name):
    data = OnlineCatalog(data_name)
    model = _train_model(data, model_type, backend="pytorch")

    single_sample = data.df.iloc[[22]]
    samples = data.df.iloc[0:22]

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
