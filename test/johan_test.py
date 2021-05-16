# flake8: noqa
import pandas as pd
import tensorflow as tf
from keras.models import Model, Sequential, model_from_json
from tensorflow import Graph, Session

from carla.data.catalog.catalog import DataCatalog
from carla.models.catalog.catalog import MLModelCatalog
from carla.models.negative_instances import predict_negative_instances
from carla.recourse_methods.autoencoder import Autoencoder, train_autoencoder
from carla.recourse_methods.catalog.cem.cem import CEM


def test_cem_get_counterfactuals():
    data_name = "adult"
    data = DataCatalog(data_name=data_name)

    hyperparams_cem = {"kappa": 0.1, "beta": 0.9, "gamma": 0.0, "mode": "PN"}

    graph = Graph()
    with graph.as_default():
        ann_sess = Session()
        with ann_sess.as_default():
            model_ann = MLModelCatalog(data=data, model_type="ann")

            ae = Autoencoder([len(model_ann.feature_input_order), 20, 10, 7], data_name)
            model_ae = train_autoencoder(
                ae,
                data,
                model_ann.scaler,
                model_ann.encoder,
                model_ann.feature_input_order,
                epochs=5,
                save=False,
            )

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
