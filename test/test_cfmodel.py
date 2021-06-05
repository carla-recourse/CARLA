import numpy as np
import pytest
from tensorflow import Graph, Session
from torch import nn

from carla.data.catalog import DataCatalog
from carla.models.catalog import MLModelCatalog
from carla.models.negative_instances import predict_negative_instances
from carla.recourse_methods import Revise
from carla.recourse_methods.catalog.actionable_recourse import ActionableRecourse
from carla.recourse_methods.catalog.cem import CEM
from carla.recourse_methods.catalog.clue import Clue
from carla.recourse_methods.catalog.dice import Dice
from carla.recourse_methods.catalog.face import Face
from carla.recourse_methods.catalog.growing_spheres.model import GrowingSpheres
from carla.recourse_methods.catalog.wachter import Wachter

testmodel = ["ann", "linear"]


@pytest.mark.parametrize("model_type", testmodel)
def test_dice_get_counterfactuals(model_type):
    # Build data and mlmodel
    data_name = "adult"
    data = DataCatalog(data_name)

    model_tf = MLModelCatalog(data, model_type)
    # get factuals
    factuals = predict_negative_instances(model_tf, data)

    hyperparams = {"num": 1, "desired_class": 1}
    # Pipeline needed for dice, but not for predicting negative instances
    model_tf.use_pipeline = True
    test_factual = factuals.iloc[:5]

    cfs = Dice(model_tf, hyperparams).get_counterfactuals(factuals=test_factual)

    assert test_factual.shape[0] == cfs.shape[0]
    assert (cfs.columns == model_tf.feature_input_order + [data.target]).all()


@pytest.mark.parametrize("model_type", testmodel)
def test_ar_get_counterfactual(model_type):
    # Build data and mlmodel
    data_name = "adult"
    data = DataCatalog(data_name)
    model_tf = MLModelCatalog(data, model_type)

    coeffs, intercepts = None, None

    if model_type == "linear":
        # get weights and bias of linear layer for negative class 0
        coeffs = model_tf.raw_model.layers[0].get_weights()[0][:, 0]
        intercepts = np.array(model_tf.raw_model.layers[0].get_weights()[1][0])

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
        "ae_params": {"h1": 20, "h2": 10, "d": 7, "train_ae": True, "epochs": 5},
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
                catalog_model=model_ann,
                hyperparams=hyperparams_cem,
            )

            counterfactuals_df = recourse.get_counterfactuals(factuals=test_factuals)

    assert counterfactuals_df.shape == test_factuals.shape


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
        "ae_params": {"h1": 20, "h2": 10, "d": 7, "train_ae": True, "epochs": 5},
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
                catalog_model=model_ann,
                hyperparams=hyperparams_cem,
            )

            counterfactuals_df = recourse.get_counterfactuals(factuals=test_factuals)

    assert counterfactuals_df.shape == test_factuals.shape


@pytest.mark.parametrize("model_type", testmodel)
def test_face_get_counterfactuals(model_type):
    # Build data and mlmodel
    data_name = "adult"
    data = DataCatalog(data_name)

    model_tf = MLModelCatalog(data, model_type)
    # get factuals
    factuals = predict_negative_instances(model_tf, data)
    test_factual = factuals.iloc[:2]

    # Test for knn mode
    hyperparams = {"mode": "knn", "fraction": 0.25}
    face = Face(model_tf, hyperparams)
    df_cfs = face.get_counterfactuals(test_factual)

    assert test_factual.shape[0] == df_cfs.shape[0]
    assert (df_cfs.columns == model_tf.feature_input_order + [data.target]).all()

    # Test for epsilon mode
    face.mode = "epsilon"
    df_cfs = face.get_counterfactuals(test_factual)

    assert test_factual.shape[0] == df_cfs.shape[0]
    assert (df_cfs.columns == model_tf.feature_input_order + [data.target]).all()


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
        "data_name": "adult",
        "train_vae": True,
        "width": 10,
        "depth": 3,
        "latent_dim": 12,
        "batch_size": 64,
        "epochs": 1,  # Only for test purpose, else at least 10 epochs
        "lr": 1e-3,
        "early_stop": 10,
    }
    df_cfs = Clue(data, model, hyperparams).get_counterfactuals(test_factual)

    assert test_factual.shape[0] == df_cfs.shape[0]
    assert (df_cfs.columns == model.feature_input_order + [data.target]).all()


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


# @pytest.mark.parametrize("model_type", testmodel)
def test_revise():
    data_name = "adult"
    data = DataCatalog(data_name)

    model = MLModelCatalog(data, "ann", backend="pytorch")
    # get factuals
    factuals = predict_negative_instances(model, data)
    test_factual = factuals.iloc[:20]

    vae_params = {
        "d": 8,  # latent space
        "D": test_factual.shape[1],  # input size
        "H1": 512,
        "H2": 256,
        "activFun": nn.ReLU(),
        "train": False,
        "lambda_reg": 1e-6,
        "epochs": 5,
        "lr": 1e-3,
        "batch_size": 32,
    }

    hyperparams = {
        "data_name": data_name,
        "lambda": 1,
        "optimizer": "adam",
        "lr": 0.05,
        "max_iter": 500,
        "vae_params": vae_params,
    }

    revise = Revise(model, data, hyperparams)
    df_cfs = revise.get_counterfactuals(test_factual)

    assert test_factual.shape[0] == df_cfs.shape[0]
    assert (df_cfs.columns == model.feature_input_order + [data.target]).all()
