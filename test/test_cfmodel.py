import numpy as np
import pandas as pd
import pytest
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.preprocessing import OneHotEncoder
from tensorflow import Graph, Session

from carla.data.catalog import OnlineCatalog
from carla.models.catalog import MLModelCatalog
from carla.models.negative_instances import predict_negative_instances
from carla.recourse_methods.catalog.actionable_recourse import ActionableRecourse
from carla.recourse_methods.catalog.cchvae import CCHVAE
from carla.recourse_methods.catalog.cem import CEM
from carla.recourse_methods.catalog.clue import Clue
from carla.recourse_methods.catalog.crud import CRUD
from carla.recourse_methods.catalog.dice import Dice
from carla.recourse_methods.catalog.face import Face
from carla.recourse_methods.catalog.fare import EFARE, FARE, fare_actions_factory
from carla.recourse_methods.catalog.feature_tweak import FeatureTweak
from carla.recourse_methods.catalog.focus import FOCUS
from carla.recourse_methods.catalog.growing_spheres.model import GrowingSpheres
from carla.recourse_methods.catalog.revise import Revise
from carla.recourse_methods.catalog.roar import Roar
from carla.recourse_methods.catalog.wachter import Wachter

testmodel = ["ann", "linear"]


@pytest.mark.parametrize("backend", ["xgboost", "sklearn"])
def test_feature_tweak_get_counterfactuals(backend):
    # Build data and mlmodel
    data_name = "adult"
    data = OnlineCatalog(data_name)
    model = MLModelCatalog(data, "forest", backend, load_online=False)
    model.train(max_depth=2, n_estimators=5)

    # get factuals
    factuals = predict_negative_instances(model, data.df)
    test_factual = factuals.iloc[:5]

    feature_tweak = FeatureTweak(model)
    cfs = feature_tweak.get_counterfactuals(test_factual)

    assert test_factual[data.continuous].shape == cfs.shape
    assert isinstance(cfs, pd.DataFrame)


@pytest.mark.parametrize("backend", ["sklearn", "xgboost"])
def test_focus_get_counterfactuals(backend):
    # Build data and mlmodel
    data_name = "adult"
    data = OnlineCatalog(data_name)
    model = MLModelCatalog(data, "forest", backend, load_online=False)
    model.train(max_depth=2, n_estimators=5)

    # get factuals
    factuals = predict_negative_instances(model, data.df)
    test_factual = factuals.iloc[:5]

    focus = FOCUS(model)
    cfs = focus.get_counterfactuals(test_factual)

    assert test_factual[data.continuous].shape == cfs.shape
    assert isinstance(cfs, pd.DataFrame)


@pytest.mark.parametrize("model_type", testmodel)
def test_dice_get_counterfactuals(model_type):
    # Build data and mlmodel
    data_name = "adult"
    data = OnlineCatalog(data_name)
    model = MLModelCatalog(data, model_type, backend="tensorflow")

    # get factuals
    factuals = predict_negative_instances(model, data.df)
    test_factual = factuals.iloc[:5]

    df_cfs = Dice(model).get_counterfactuals(factuals=test_factual)
    cfs = model.get_ordered_features(df_cfs)

    assert test_factual.shape[0] == cfs.shape[0]
    assert (cfs.columns == model.feature_input_order).all()
    assert isinstance(cfs, pd.DataFrame)


@pytest.mark.parametrize("model_type", testmodel)
def test_ar_get_counterfactual(model_type):
    # Build data and mlmodel
    data_name = "adult"
    data = OnlineCatalog(data_name)
    model = MLModelCatalog(data, model_type, backend="tensorflow")

    coeffs, intercepts = None, None

    if model_type == "linear":
        # get weights and bias of linear layer for negative class 0
        coeffs_neg = model.raw_model.layers[0].get_weights()[0][:, 0]
        intercepts_neg = np.array(model.raw_model.layers[0].get_weights()[1][0])

        # get weights and bias of linear layer for positive class 1
        coeffs_pos = model.raw_model.layers[0].get_weights()[0][:, 1]
        intercepts_pos = np.array(model.raw_model.layers[0].get_weights()[1][1])

        coeffs = -(coeffs_neg - coeffs_pos)
        intercepts = -(intercepts_neg - intercepts_pos)

    # get factuals
    factuals = predict_negative_instances(model, data.df)
    test_factual = factuals.iloc[:5]

    # get counterfactuals
    cfs = ActionableRecourse(
        model, coeffs=coeffs, intercepts=intercepts
    ).get_counterfactuals(test_factual)

    assert test_factual.shape[0] == cfs.shape[0]
    assert (cfs.columns == model.feature_input_order).all()
    assert isinstance(cfs, pd.DataFrame)


@pytest.mark.parametrize("model_type", testmodel)
def test_cem_get_counterfactuals(model_type):
    data_name = "adult"
    data = OnlineCatalog(data_name=data_name)

    hyperparams_cem = {
        "data_name": data_name,
    }

    graph = Graph()
    with graph.as_default():
        ann_sess = Session()
        with ann_sess.as_default():
            model_ann = MLModelCatalog(
                data=data,
                model_type=model_type,
                encoding_method="Binary",
                backend="tensorflow",
            )

            factuals = predict_negative_instances(model_ann, data.df)
            test_factuals = factuals.iloc[:5]

            recourse = CEM(
                sess=ann_sess,
                mlmodel=model_ann,
                hyperparams=hyperparams_cem,
            )

            counterfactuals_df = recourse.get_counterfactuals(factuals=test_factuals)

    assert (
        counterfactuals_df.shape == model_ann.get_ordered_features(test_factuals).shape
    )
    assert isinstance(counterfactuals_df, pd.DataFrame)


@pytest.mark.parametrize("model_type", testmodel)
def test_cem_vae(model_type):
    data_name = "adult"
    data = OnlineCatalog(data_name=data_name)

    hyperparams_cem = {
        "beta": 0.0,
        "gamma": 6.0,
        "data_name": data_name,
    }

    graph = Graph()
    with graph.as_default():
        ann_sess = Session()
        with ann_sess.as_default():
            model_ann = MLModelCatalog(
                data=data,
                model_type=model_type,
                encoding_method="Binary",
                backend="tensorflow",
            )

            factuals = predict_negative_instances(model_ann, data.df)
            test_factuals = factuals.iloc[:5]

            recourse = CEM(
                sess=ann_sess,
                mlmodel=model_ann,
                hyperparams=hyperparams_cem,
            )

            counterfactuals_df = recourse.get_counterfactuals(factuals=test_factuals)

    assert (
        counterfactuals_df.shape == model_ann.get_ordered_features(test_factuals).shape
    )
    assert isinstance(counterfactuals_df, pd.DataFrame)


@pytest.mark.parametrize("model_type", testmodel)
def test_face_get_counterfactuals(model_type):
    # Build data and mlmodel
    data_name = "adult"
    data = OnlineCatalog(data_name)

    model = MLModelCatalog(data, model_type, backend="tensorflow")
    # get factuals
    factuals = predict_negative_instances(model, data.df)
    test_factual = factuals.iloc[:5]

    # Test for knn mode
    face = Face(model)
    df_cfs = face.get_counterfactuals(test_factual)

    assert test_factual.shape[0] == df_cfs.shape[0]
    assert (df_cfs.columns == model.feature_input_order).all()

    # Test for epsilon mode
    face.mode = "epsilon"
    df_cfs = face.get_counterfactuals(test_factual)

    assert test_factual.shape[0] == df_cfs.shape[0]
    assert (df_cfs.columns == model.feature_input_order).all()
    assert isinstance(df_cfs, pd.DataFrame)


@pytest.mark.parametrize("model_type", testmodel)
def test_growing_spheres(model_type):
    # Build data and mlmodel
    data_name = "adult"
    data = OnlineCatalog(data_name)

    model = MLModelCatalog(data, model_type, backend="tensorflow")
    # get factuals
    factuals = predict_negative_instances(model, data.df)
    test_factual = factuals.iloc[:5]

    gs = GrowingSpheres(model)
    df_cfs = gs.get_counterfactuals(test_factual)

    assert test_factual.shape[0] == df_cfs.shape[0]
    assert (df_cfs.columns == model.feature_input_order).all()
    assert isinstance(df_cfs, pd.DataFrame)


@pytest.mark.parametrize("model_type", testmodel)
def test_clue(model_type):
    # Build data and mlmodel
    data_name = "adult"
    data = OnlineCatalog(data_name)

    model = MLModelCatalog(data, model_type, backend="pytorch")
    # get factuals
    factuals = predict_negative_instances(model, data.df)
    test_factual = factuals.iloc[:20]

    hyperparams = {
        "data_name": data_name,
        "depth": 2,
        "latent_dim": 8,
        "epochs": 1,  # Only for test purpose, else at least 10 epochs
    }
    df_cfs = Clue(data, model, hyperparams).get_counterfactuals(test_factual)

    assert test_factual.shape[0] == df_cfs.shape[0]
    assert (df_cfs.columns == model.feature_input_order).all()
    assert isinstance(df_cfs, pd.DataFrame)


@pytest.mark.parametrize("model_type", testmodel)
def test_wachter(model_type):
    # Build data and mlmodel
    data_name = "adult"
    data = OnlineCatalog(data_name)

    model = MLModelCatalog(data, model_type, backend="pytorch")
    # get factuals
    factuals = predict_negative_instances(model, data.df)
    test_factual = factuals.iloc[:10]

    hyperparams = {"loss_type": "MSE", "y_target": [1]}
    df_cfs = Wachter(model, hyperparams).get_counterfactuals(test_factual)

    assert test_factual.shape[0] == df_cfs.shape[0]
    assert (df_cfs.columns == model.feature_input_order).all()
    assert isinstance(df_cfs, pd.DataFrame)

    hyperparams = {"loss_type": "BCE", "y_target": [0, 1]}
    df_cfs = Wachter(model, hyperparams).get_counterfactuals(test_factual)

    assert test_factual.shape[0] == df_cfs.shape[0]
    assert (df_cfs.columns == model.feature_input_order).all()
    assert isinstance(df_cfs, pd.DataFrame)


@pytest.mark.parametrize("model_type", testmodel)
def test_roar(model_type):
    # Build data and mlmodel
    data_name = "adult"
    data = OnlineCatalog(data_name)

    model = MLModelCatalog(data, model_type, backend="pytorch")
    # get factuals
    factuals = predict_negative_instances(model, data.df)
    test_factual = factuals.iloc[:10]

    hyperparams = {"loss_type": "MSE", "y_target": [1]}
    df_cfs = Roar(model, hyperparams).get_counterfactuals(test_factual)

    assert test_factual.shape[0] == df_cfs.shape[0]
    assert (df_cfs.columns == model.feature_input_order).all()
    assert isinstance(df_cfs, pd.DataFrame)

    hyperparams = {"loss_type": "BCE", "y_target": [0, 1]}
    df_cfs = Roar(model, hyperparams).get_counterfactuals(test_factual)

    assert test_factual.shape[0] == df_cfs.shape[0]
    assert (df_cfs.columns == model.feature_input_order).all()
    assert isinstance(df_cfs, pd.DataFrame)


@pytest.mark.parametrize("model_type", testmodel)
def test_revise(model_type):
    data_name = "adult"
    data = OnlineCatalog(data_name)

    model = MLModelCatalog(data, model_type, backend="pytorch")
    # get factuals
    factuals = predict_negative_instances(model, data.df)
    test_factual = factuals.iloc[:5]

    vae_params = {
        "layers": [sum(model.get_mutable_mask()), 512, 256, 8],
        "epochs": 1,
    }

    hyperparams = {
        "data_name": data_name,
        "vae_params": vae_params,
    }

    revise = Revise(model, data, hyperparams)
    df_cfs = revise.get_counterfactuals(test_factual)

    assert test_factual.shape[0] == df_cfs.shape[0]
    assert (df_cfs.columns == model.feature_input_order).all()
    assert isinstance(df_cfs, pd.DataFrame)


@pytest.mark.parametrize("model_type", testmodel)
def test_cchvae(model_type):
    data_name = "compas"
    data = OnlineCatalog(data_name)

    model = MLModelCatalog(data, model_type, backend="pytorch")
    # get factuals
    factuals = predict_negative_instances(model, data.df)
    test_factual = factuals.iloc[:5]

    hyperparams = {
        "data_name": data_name,
        "n_search_samples": 100,
        "vae_params": {
            "layers": [sum(model.get_mutable_mask()), 512, 256, 8],
        },
    }

    cchvae = CCHVAE(model, hyperparams)
    df_cfs = cchvae.get_counterfactuals(test_factual)

    assert test_factual.shape[0] == df_cfs.shape[0]
    assert (df_cfs.columns == model.feature_input_order).all()
    assert isinstance(df_cfs, pd.DataFrame)


@pytest.mark.parametrize("model_type", testmodel)
def test_crud(model_type):
    # Build data and mlmodel
    data_name = "adult"
    data = OnlineCatalog(data_name)

    model = MLModelCatalog(data, model_type, backend="pytorch")
    # get factuals
    factuals = predict_negative_instances(model, data.df)
    test_factual = factuals.iloc[:5]

    hyperparams = {
        "data_name": data_name,
        "vae_params": {
            "layers": [sum(model.get_mutable_mask()), 16, 8],
        },
    }

    crud = CRUD(model, hyperparams)
    df_cfs = crud.get_counterfactuals(test_factual)

    assert test_factual.shape[0] == df_cfs.shape[0]
    assert isinstance(df_cfs, pd.DataFrame)


@pytest.mark.parametrize("model_type", testmodel)
def test_fare(model_type):
    # Build data and mlmodel
    data_name = "adult"
    data = OnlineCatalog(data_name)

    model = MLModelCatalog(data, model_type, backend="pytorch")
    # get factuals
    factuals = predict_negative_instances(model, data.df)
    test_factual = factuals.iloc[:5]

    # Filter and get only the negative instances
    training_data = predict_negative_instances(model, data.df)
    training_data = data.inverse_transform(training_data)

    # Build the actions and types for this environment.
    # It might require some tinkering
    actions, types = fare_actions_factory(
        data.inverse_transform(data.df_train), data.immutables
    )

    # Instantiate configurations for FARE
    hyperparams = {
        "environment_config": {
            "class_name": "carla.recourse_methods.catalog.fare.MockEnv",
            "additional_parameters": {
                "preprocessor": data,
                "preprocessor_fare": data,
                "programs_library": actions,
                "arguments": types,
            },
        },
        "policy_config": {
            "observation_dim": 14,
            "encoding_dim": 10,
            "hidden_size": 10,
        },
        "mcts_config": {
            "exploration": True,
            "number_of_simulations": 10,
            "dir_epsilon": 0.4,
            "dir_noise": 0.3,
            "action_duplicate_cost": 10,
        },
    }

    fare = FARE(model, hyperparams)
    fare.fit(training_data, max_iter=5, verbose=True)

    df_cfs = fare.get_counterfactuals(data.inverse_transform(test_factual))

    assert test_factual.shape[0] == df_cfs.shape[0]
    assert isinstance(df_cfs, pd.DataFrame)

    # Test the EFARE model
    cat_selector = make_column_selector(dtype_include=[object, "category"])
    preprocessor_efare = make_column_transformer(
        (OneHotEncoder(handle_unknown="ignore", sparse=False), cat_selector),
        remainder="passthrough",
    )

    # We fit the EFARE preprocessor
    preprocessor_efare.fit(training_data)

    hyperparams_efare = {
        "fare_model": fare.fare_model,
        "preprocessor": data,
        "preprocessor_efare": preprocessor_efare,
    }

    efare = EFARE(model, hyperparams_efare)
    efare.fit(training_data[0:10], verbose=True)

    df_cfs = efare.get_counterfactuals(data.inverse_transform(test_factual))
    print(df_cfs)

    assert test_factual.shape[0] == df_cfs.shape[0]
    assert isinstance(df_cfs, pd.DataFrame)
