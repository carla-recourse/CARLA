from tensorflow import Graph, Session

from carla.data.catalog import DataCatalog
from carla.models.catalog import MLModelCatalog
from carla.models.negative_instances import predict_negative_instances
from carla.recourse_methods.catalog.actionable_recourse import ActionableRecourse
from carla.recourse_methods.catalog.cem.cem import CEM
from carla.recourse_methods.catalog.dice import Dice
from carla.recourse_methods.catalog.face import Face


def test_dice_get_counterfactuals():
    # Build data and mlmodel
    data_name = "adult"
    data = DataCatalog(data_name)

    model_tf = MLModelCatalog(data, "ann")
    # get factuals
    factuals = predict_negative_instances(model_tf, data)

    hyperparams = {"num": 1, "desired_class": 1}
    # Pipeline needed for dice, but not for predicting negative instances
    model_tf.use_pipeline = True
    test_factual = factuals.iloc[:5]

    cfs = Dice(model_tf, hyperparams).get_counterfactuals(factuals=test_factual)

    assert test_factual.shape[0] == cfs.shape[0]
    assert (cfs.columns == model_tf.feature_input_order + [data.target]).all()


def test_ar_get_counterfactual():
    # Build data and mlmodel
    data_name = "adult"
    data = DataCatalog(data_name)
    model_tf = MLModelCatalog(data, "ann")

    # get factuals
    factuals = predict_negative_instances(model_tf, data)
    test_factual = factuals.iloc[:5]

    # get counterfactuals
    hyperparams = {"fs_size": 150}
    cfs = ActionableRecourse(model_tf, hyperparams).get_counterfactuals(test_factual)

    assert test_factual.shape[0] == cfs.shape[0]
    assert (cfs.columns == model_tf.feature_input_order + [data.target]).all()


def test_cem_get_counterfactuals():
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
        "ae_params": {"h1": 20, "h2": 10, "d": 7},
    }

    graph = Graph()
    with graph.as_default():
        ann_sess = Session()
        with ann_sess.as_default():
            model_ann = MLModelCatalog(
                data=data, model_type="ann", encoding_method="Binary"
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


def test_face_get_counterfactuals():
    # Build data and mlmodel
    data_name = "adult"
    data = DataCatalog(data_name)

    model_tf = MLModelCatalog(data, "ann")
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
