import numpy as np
import tensorflow as tf

from carla.data.catalog import DataCatalog
from carla.models.catalog import MLModelCatalog
from carla.recourse_methods.autoencoder import Autoencoder, train_autoencoder


def test_autoencoder():
    # Build data and mlmodel
    data_name = "adult"
    data_catalog = "adult_catalog.yaml"
    data = DataCatalog(data_name, data_catalog)

    feature_input_order = [
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

    model = MLModelCatalog(data, "ann", feature_input_order)

    ae = Autoencoder(len(model.feature_input_order), 20, 10, 5, data_name)
    fitted_ae = train_autoencoder(
        ae,
        data,
        model.scaler,
        model.encoder,
        model.feature_input_order,
        epochs=5,
        save=False,
    )

    test_input = tf.Variable(np.zeros((1, 13)), dtype=tf.float32)
    test_output = fitted_ae(test_input)
    expected_shape = (1, 13)

    assert test_output.shape == expected_shape
