import numpy as np
import tensorflow as tf
import torch
from keras import backend as K
from torch import nn

from carla.data.catalog import DataCatalog
from carla.models.catalog import MLModelCatalog
from carla.recourse_methods.autoencoder import (
    Autoencoder,
    VariationalAutoencoder,
    train_autoencoder,
    train_variational_autoencoder,
)


def test_variational_autoencoder():
    # Build data and mlmodel
    data_name = "adult"
    data = DataCatalog(data_name)

    model = MLModelCatalog(data, "ann", backend="pytorch")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_input = np.zeros((1, 20))
    test_input = torch.Tensor(test_input).to(device)

    vae_params = {
        "d": 8,  # latent space
        "H1": 512,
        "H2": 256,
        "activFun": nn.ReLU(),
    }

    vae = VariationalAutoencoder(
        data_name,
        vae_params["d"],
        test_input.shape[1],
        vae_params["H1"],
        vae_params["H2"],
    )

    fitted_vae = train_variational_autoencoder(
        vae, data, model.scaler, model.encoder, model.feature_input_order
    )

    test_reconstructed, _, _, _, _ = fitted_vae.predict(test_input)

    assert test_reconstructed.shape == test_input.shape

    # test loading vae
    new_vae = VariationalAutoencoder(
        data_name,
        vae_params["d"],
        test_input.shape[1],
        vae_params["H1"],
        vae_params["H2"],
    )

    new_vae.load(test_input.shape[1])


def test_autoencoder():
    # Build data and mlmodel
    data_name = "adult"
    data = DataCatalog(data_name)

    model = MLModelCatalog(data, "ann")
    test_input = tf.Variable(np.zeros((1, 13)), dtype=tf.float32)

    ae = Autoencoder(data_name, [len(model.feature_input_order), 20, 10, 5])
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
    ae = Autoencoder(data_name, [len(model.feature_input_order), 5])
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
        data_name, [len(model.feature_input_order), 20, 15, 10, 8, 5], loss=custom_loss
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


def test_save_and_load():
    with tf.Session() as sess:
        # Build data and mlmodel
        data_name = "adult"
        data = DataCatalog(data_name)

        model = MLModelCatalog(data, "ann")
        test_input = tf.Variable(np.zeros((1, 13)), dtype=tf.float32)

        ae = Autoencoder(data_name, [len(model.feature_input_order), 20, 10, 5])
        fitted_ae = train_autoencoder(
            ae,
            data,
            model.scaler,
            model.encoder,
            model.feature_input_order,
            epochs=5,
            save=True,
        )

        expected = fitted_ae(test_input)

        loaded_ae = Autoencoder(data_name).load(len(model.feature_input_order))
        actual = loaded_ae(test_input)

        assert (actual.eval(session=sess) == expected.eval(session=sess)).all()
