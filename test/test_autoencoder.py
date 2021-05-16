import numpy as np
import tensorflow as tf
from keras import backend as K

from carla.data.catalog import DataCatalog
from carla.models.catalog import MLModelCatalog
from carla.recourse_methods.autoencoder import Autoencoder, train_autoencoder


def test_autoencoder():
    # Build data and mlmodel
    data_name = "adult"
    data = DataCatalog(data_name)

    model = MLModelCatalog(data, "ann")
    test_input = tf.Variable(np.zeros((1, 13)), dtype=tf.float32)

    ae = Autoencoder([len(model.feature_input_order), 20, 10, 5], data_name)
    fitted_ae = train_autoencoder(
        ae,
        data,
        model.scaler,
        model.encoder,
        model.feature_input_order,
        epochs=5,
        save=False,
    )
    test_output = fitted_ae(test_input)

    expected_shape = (1, 13)
    assert test_output.shape == expected_shape

    # test with different lengths
    ae = Autoencoder([len(model.feature_input_order), 5], data_name)
    fitted_ae = train_autoencoder(
        ae,
        data,
        model.scaler,
        model.encoder,
        model.feature_input_order,
        epochs=5,
        save=False,
    )
    test_output = fitted_ae(test_input)

    expected_shape = (1, 13)
    assert test_output.shape == expected_shape

    # test with different loss function
    def custom_loss(y_true, y_pred):
        return K.max(y_true - y_pred)

    ae = Autoencoder(
        [len(model.feature_input_order), 20, 15, 10, 8, 5], data_name, loss=custom_loss
    )
    fitted_ae = train_autoencoder(
        ae,
        data,
        model.scaler,
        model.encoder,
        model.feature_input_order,
        epochs=5,
        save=False,
    )
    test_output = fitted_ae(test_input)

    expected_shape = (1, 13)
    assert test_output.shape == expected_shape
