import numpy as np
import pytest
from tensorflow import Graph, Session

from carla.data.catalog import DataCatalog
from carla.models.catalog import MLModelCatalog
from carla.models.negative_instances import predict_negative_instances
from carla.recourse_methods.catalog.actionable_recourse import ActionableRecourse
from carla.recourse_methods.catalog.cchvae import CCHVAE
from carla.recourse_methods.catalog.cem import CEM
from carla.recourse_methods.catalog.clue import Clue
from carla.recourse_methods.catalog.crud import CRUD
from carla.recourse_methods.catalog.dice import Dice
from carla.recourse_methods.catalog.face import Face
from carla.recourse_methods.catalog.feature_tweak import FeatureTweak
from carla.recourse_methods.catalog.focus import FOCUS
from carla.recourse_methods.catalog.focus.tree_model import ForestModel, XGBoostModel
from carla.recourse_methods.catalog.growing_spheres.model import GrowingSpheres
from carla.recourse_methods.catalog.revise import Revise
from carla.recourse_methods.catalog.wachter import Wachter

testmodel = ["ann", "linear"]


@pytest.mark.parametrize("model_type", ["xgboost", "sklearn"])
def test_feature_tweak_get_counterfactuals(model_type):

    data_name = "adult"
    data = DataCatalog(data_name)

    if model_type == "xgboost":
        model = XGBoostModel(data)
    elif model_type == "sklearn":
        model = ForestModel(data)
    else:
        raise ValueError("model type not recognized")

    hyperparams = {
        "eps": 0.1,
    }

    # get factuals
    factuals = predict_negative_instances(model, data)
    test_factual = factuals.iloc[:5]

    feature_tweak = FeatureTweak(model, hyperparams)
    cfs = feature_tweak.get_counterfactuals(test_factual)

    assert test_factual[data.continous + [data.target]].shape == cfs.shape

    non_nan_cfs = cfs.dropna()
    assert non_nan_cfs.shape[0] > 0


@pytest.mark.parametrize("model_type", ["xgboost", "sklearn"])
def test_focus_get_counterfactuals(model_type):

    data_name = "adult"
    data = DataCatalog(data_name)

    if model_type == "xgboost":
        model = XGBoostModel(data)
    elif model_type == "sklearn":
        model = ForestModel(data)
    else:
        raise ValueError("model type not recognized")

    hyperparams = {
        "optimizer": "adam",
        "lr": 0.001,
        "n_class": 2,
        "n_iter": 1000,
        "sigma": 1.0,
        "temperature": 1.0,
        "distance_weight": 0.01,
        "distance_func": "l1",
    }

    # get factuals
    factuals = predict_negative_instances(model, data)
    test_factual = factuals.iloc[:5]

    focus = FOCUS(model, data, hyperparams)
    cfs = focus.get_counterfactuals(test_factual)

    assert test_factual[data.continous].shape == cfs.shape

    non_nan_cfs = cfs.dropna()
    assert non_nan_cfs.shape[0] > 0


@pytest.mark.parametrize("model_type", testmodel)
def test_dice_get_counterfactuals(model_type):
    # Build data and mlmodel
    data_name = "adult"
    data = DataCatalog(data_name)

    model_tf = MLModelCatalog(data, model_type)
    # get factuals
    factuals = predict_negative_instances(model_tf, data)

    hyperparams = {
        "num": 1,
        "desired_class": 1,
        "posthoc_sparsity_param": 0.1,
    }
    # Pipeline needed for dice, but not for predicting negative instances
    model_tf.use_pipeline = True
    test_factual = factuals.iloc[:5]

    cfs = Dice(model_tf, hyperparams).get_counterfactuals(factuals=test_factual)

    assert test_factual.shape[0] == cfs.shape[0]
    assert (cfs.columns == model_tf.feature_input_order + [data.target]).all()

    non_nan_cfs = cfs.dropna()
    assert non_nan_cfs.shape[0] > 0


@pytest.mark.parametrize("model_type", testmodel)
def test_ar_get_counterfactual(model_type):
    # Build data and mlmodel
    data_name = "adult"
    data = DataCatalog(data_name)
    model_tf = MLModelCatalog(data, model_type)

    coeffs, intercepts = None, None

    if model_type == "linear":
        # get weights and bias of linear layer for negative class 0
        coeffs_neg = model_tf.raw_model.layers[0].get_weights()[0][:, 0]
        intercepts_neg = np.array(model_tf.raw_model.layers[0].get_weights()[1][0])

        # get weights and bias of linear layer for positive class 1
        coeffs_pos = model_tf.raw_model.layers[0].get_weights()[0][:, 1]
        intercepts_pos = np.array(model_tf.raw_model.layers[0].get_weights()[1][1])

        coeffs = -(coeffs_neg - coeffs_pos)
        intercepts = -(intercepts_neg - intercepts_pos)

    # get factuals
    factuals = predict_negative_instances(model_tf, data)
    test_factual = factuals.iloc[:5]

    # get counterfactuals
    hyperparams = {"fs_size": 150}
    cfs = ActionableRecourse(
        model_tf, hyperparams, coeffs=coeffs, intercepts=intercepts
    ).get_counterfactuals(test_factual)

    assert test_factual.shape[0] == cfs.shape[0]
    assert (cfs.columns == model_tf.feature_input_order + [data.target]).all()

    non_nan_cfs = cfs.dropna()
    assert non_nan_cfs.shape[0] > 0


@pytest.mark.parametrize("model_type", testmodel)
def test_cem_get_counterfactuals(model_type):
    data_name = "adult"
    data = DataCatalog(data_name=data_name)

    hyperparams_cem = {
        "batch_size": 1,
        "kappa": 0.1,
        "init_learning_rate": 1e-2,
        "binary_search_steps": 9,
        "max_iterations": 100,
        "initial_const": 10,
        "beta": 0.9,
        "gamma": 0.0,
        "mode": "PN",
        "num_classes": 2,
        "data_name": data_name,
        "ae_params": {"hidden_layer": [20, 10, 7], "train_ae": True, "epochs": 5},
    }

    graph = Graph()
    with graph.as_default():
        ann_sess = Session()
        with ann_sess.as_default():
            model_ann = MLModelCatalog(
                data=data, model_type=model_type, encoding_method="Binary"
            )

            factuals = predict_negative_instances(model_ann, data)
            test_factuals = factuals.iloc[:5]

            recourse = CEM(
                sess=ann_sess,
                mlmodel=model_ann,
                hyperparams=hyperparams_cem,
            )

            counterfactuals_df = recourse.get_counterfactuals(factuals=test_factuals)

    assert counterfactuals_df.shape == test_factuals.shape

    non_nan_cfs = counterfactuals_df.dropna()
    assert non_nan_cfs.shape[0] > 0


@pytest.mark.parametrize("model_type", testmodel)
def test_cem_vae(model_type):
    data_name = "adult"
    data = DataCatalog(data_name=data_name)

    hyperparams_cem = {
        "batch_size": 1,
        "kappa": 0.1,
        "init_learning_rate": 1e-2,
        "binary_search_steps": 9,
        "max_iterations": 100,
        "initial_const": 10,
        "beta": 0.0,
        "gamma": 6.0,
        "mode": "PN",
        "num_classes": 2,
        "data_name": data_name,
        "ae_params": {"hidden_layer": [20, 10, 7], "train_ae": True, "epochs": 5},
    }

    graph = Graph()
    with graph.as_default():
        ann_sess = Session()
        with ann_sess.as_default():
            model_ann = MLModelCatalog(
                data=data, model_type=model_type, encoding_method="Binary"
            )

            factuals = predict_negative_instances(model_ann, data)
            test_factuals = factuals.iloc[:5]

            recourse = CEM(
                sess=ann_sess,
                mlmodel=model_ann,
                hyperparams=hyperparams_cem,
            )

            counterfactuals_df = recourse.get_counterfactuals(factuals=test_factuals)

    assert counterfactuals_df.shape == test_factuals.shape

    non_nan_cfs = counterfactuals_df.dropna()
    assert non_nan_cfs.shape[0] > 0


@pytest.mark.parametrize("model_type", testmodel)
def test_face_get_counterfactuals(model_type):
    # Build data and mlmodel
    data_name = "adult"
    data = DataCatalog(data_name)

    model_tf = MLModelCatalog(data, model_type)
    # get factuals
    factuals = predict_negative_instances(model_tf, data)
    test_factual = factuals.iloc[:5]

    # Test for knn mode
    hyperparams = {"mode": "knn", "fraction": 0.05}
    face = Face(model_tf, hyperparams)
    df_cfs = face.get_counterfactuals(test_factual)

    assert test_factual.shape[0] == df_cfs.shape[0]
    assert (df_cfs.columns == model_tf.feature_input_order + [data.target]).all()

    # Test for epsilon mode
    face.mode = "epsilon"
    df_cfs = face.get_counterfactuals(test_factual)

    assert test_factual.shape[0] == df_cfs.shape[0]
    assert (df_cfs.columns == model_tf.feature_input_order + [data.target]).all()

    non_nan_cfs = df_cfs.dropna()
    assert non_nan_cfs.shape[0] > 0


@pytest.mark.parametrize("model_type", testmodel)
def test_growing_spheres(model_type):
    # Build data and mlmodel
    data_name = "adult"
    data = DataCatalog(data_name)

    model_tf = MLModelCatalog(data, model_type)
    # get factuals
    factuals = predict_negative_instances(model_tf, data)
    test_factual = factuals.iloc[:5]

    gs = GrowingSpheres(model_tf)
    df_cfs = gs.get_counterfactuals(test_factual)

    assert test_factual.shape[0] == df_cfs.shape[0]
    assert (df_cfs.columns == model_tf.feature_input_order + [data.target]).all()

    non_nan_cfs = df_cfs.dropna()
    assert non_nan_cfs.shape[0] > 0


@pytest.mark.parametrize("model_type", testmodel)
def test_clue(model_type):
    # Build data and mlmodel
    data_name = "adult"
    data = DataCatalog(data_name)

    model = MLModelCatalog(data, model_type, backend="pytorch")
    # get factuals
    factuals = predict_negative_instances(model, data)
    test_factual = factuals.iloc[:20]

    hyperparams = {
        "data_name": data_name,
        "train_vae": True,
        "width": 10,
        "depth": 2,
        "latent_dim": 8,
        "batch_size": 64,
        "epochs": 1,  # Only for test purpose, else at least 10 epochs
        "lr": 1e-3,
        "early_stop": 10,
    }
    df_cfs = Clue(data, model, hyperparams).get_counterfactuals(test_factual)

    assert test_factual.shape[0] == df_cfs.shape[0]
    assert (df_cfs.columns == model.feature_input_order + [data.target]).all()

    non_nan_cfs = df_cfs.dropna()
    assert non_nan_cfs.shape[0] > 0


@pytest.mark.parametrize("model_type", testmodel)
def test_wachter(model_type):
    # Build data and mlmodel
    data_name = "adult"
    data = DataCatalog(data_name)

    model = MLModelCatalog(data, model_type, backend="pytorch")
    # get factuals
    factuals = predict_negative_instances(model, data)
    test_factual = factuals.iloc[:10]

    hyperparams = {"loss_type": "BCE", "binary_cat_features": False}
    df_cfs = Wachter(model, hyperparams).get_counterfactuals(test_factual)

    assert test_factual.shape[0] == df_cfs.shape[0]
    assert (df_cfs.columns == model.feature_input_order + [data.target]).all()

    non_nan_cfs = df_cfs.dropna()
    assert non_nan_cfs.shape[0] > 0


@pytest.mark.parametrize("model_type", testmodel)
def test_revise(model_type):
    data_name = "adult"
    data = DataCatalog(data_name)

    model = MLModelCatalog(data, model_type, backend="pytorch")
    # get factuals
    factuals = predict_negative_instances(model, data)
    test_factual = factuals.iloc[:5]

    vae_params = {
        "layers": [len(model.feature_input_order), 512, 256, 8],
        "train": True,
        "lambda_reg": 1e-6,
        "epochs": 1,
        "lr": 1e-3,
        "batch_size": 32,
    }

    hyperparams = {
        "data_name": data_name,
        "lambda": 0.5,
        "optimizer": "adam",
        "lr": 0.1,
        "max_iter": 1500,
        "target_class": [0, 1],
        "vae_params": vae_params,
        "binary_cat_features": False,
    }

    revise = Revise(model, data, hyperparams)
    df_cfs = revise.get_counterfactuals(test_factual)

    assert test_factual.shape[0] == df_cfs.shape[0]
    assert (df_cfs.columns == model.feature_input_order + [data.target]).all()

    non_nan_cfs = df_cfs.dropna()
    assert non_nan_cfs.shape[0] > 0


@pytest.mark.parametrize("model_type", testmodel)
def test_cchvae(model_type):
    data_name = "compas"
    data = DataCatalog(data_name)

    model = MLModelCatalog(data, model_type, backend="pytorch")
    # get factuals
    factuals = predict_negative_instances(model, data)
    test_factual = factuals.iloc[:5]

    hyperparams = {
        "data_name": data_name,
        "n_search_samples": 100,
        "p_norm": 1,
        "step": 0.1,
        "max_iter": 1000,
        "clamp": True,
        "binary_cat_features": False,
        "vae_params": {
            "layers": [len(model.feature_input_order), 512, 256, 8],
            "train": True,
            "lambda_reg": 1e-6,
            "epochs": 5,
            "lr": 1e-3,
            "batch_size": 32,
        },
    }

    cchvae = CCHVAE(model, hyperparams)
    df_cfs = cchvae.get_counterfactuals(test_factual)

    assert test_factual.shape[0] == df_cfs.shape[0]
    assert (df_cfs.columns == model.feature_input_order + [data.target]).all()

    non_nan_cfs = df_cfs.dropna()
    assert non_nan_cfs.shape[0] > 0


@pytest.mark.parametrize("model_type", testmodel)
def test_crud(model_type):
    # Build data and mlmodel
    data_name = "adult"
    data = DataCatalog(data_name)

    model = MLModelCatalog(data, model_type, backend="pytorch")
    # get factuals
    factuals = predict_negative_instances(model, data)
    test_factual = factuals.iloc[:5]

    hyperparams = {
        "data_name": data_name,
        "target_class": [0, 1],
        "lambda_param": 0.001,
        "optimizer": "RMSprop",
        "lr": 0.008,
        "max_iter": 2000,
        "binary_cat_features": False,
        "vae_params": {
            "layers": [len(model.feature_input_order), 16, 8],
            "train": True,
            "epochs": 5,
            "lr": 1e-3,
            "batch_size": 32,
        },
    }

    crud = CRUD(model, hyperparams)
    df_cfs = crud.get_counterfactuals(test_factual)

    assert test_factual.shape[0] == df_cfs.shape[0]
    assert (df_cfs.columns == model.feature_input_order + [data.target]).all()

    non_nan_cfs = df_cfs.dropna()
    assert non_nan_cfs.shape[0] > 0
