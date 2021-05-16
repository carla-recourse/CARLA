# flake8: noqa
import pandas as pd
import tensorflow as tf
from keras.models import Model, Sequential, model_from_json
from tensorflow import Graph, Session

from carla.data.catalog.catalog import DataCatalog
from carla.models.catalog.catalog import MLModelCatalog
from carla.models.negative_instances import predict_negative_instances
from carla.recourse_methods.catalog.cem.cem import CEM


def load_AE():

    model_filename = "ae_tf.json"
    weigth_filename = "ae_tf.h5"

    json_file = open(model_filename, "r")
    decoder = model_from_json(json_file.read(), custom_objects={"tf": tf})
    json_file.close()
    decoder.load_weights(weigth_filename)

    return decoder


def test_cem_get_counterfactuals():
    data = DataCatalog(data_name="adult")

    feature_input_order = [
        "age",
        "fnlwgt",
        "education-num",
        "capital-gain",
        "hours-per-week",
        "capital-loss",
        "workclass_Private",
        "marital-status_Non-Married",
        "occupation_Other",
        "relationship_Non-Husband",
        "race_White",
        "sex_Male",
        "native-country_US",
    ]

    hyperparams_cem = {"kappa": 0.1, "beta": 0.9, "gamma": 0.0, "mode": "PN"}

    graph = Graph()
    with graph.as_default():
        ann_sess = Session()
        with ann_sess.as_default():
            model_ann = MLModelCatalog(
                data=data, model_type="ann", feature_input_order=feature_input_order
            )
            model_ae = load_AE()

            factuals = predict_negative_instances(model_ann, data)
            test_factuals = factuals.iloc[:5]

            recourse = CEM(
                sess=ann_sess,
                catalog_model=model_ann,
                mode=hyperparams_cem["mode"],
                AE=model_ae,
                batch_size=1,
                kappa=hyperparams_cem["kappa"],
                init_learning_rate=1e-2,
                binary_search_steps=9,
                max_iterations=100,
                initial_const=10,
                beta=hyperparams_cem["beta"],
                gamma=hyperparams_cem["gamma"],
            )

            result = recourse.get_counterfactuals(factuals=test_factuals)
            instance_list, cf_list, times_list, succes_rate = result

            assert pd.concat(instance_list).shape == test_factuals.shape
            assert pd.concat(cf_list).shape == test_factuals.shape
