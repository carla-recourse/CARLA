from sklearn.model_selection import train_test_split

from carla.models.pipelining import encode, scale


def train_autoencoder(
    autoencoder, data, scaler, encoder, input_order, epochs=25, batch_size=64, save=True
):
    # normalize and encode data
    df_dataset = scale(scaler, data.continous, data.raw)
    df_dataset = encode(encoder, data.categoricals, df_dataset)
    df_label_data = df_dataset[data.target]
    df_dataset = df_dataset[input_order]

    xtrain, xtest, _, _ = train_test_split(
        df_dataset.values, df_label_data.values, train_size=0.7
    )

    fitted_ae = autoencoder.train(xtrain, xtest, epochs, batch_size)

    if save:
        autoencoder.save(fitted_ae)

    return fitted_ae


def train_variational_autoencoder(
    vae,
    data,
    scaler,
    encoder,
    input_order,
    lambda_reg=1e-6,
    epochs=5,
    lr=1e-3,
    batch_size=32,
):
    # normalize and encode data
    df_dataset = scale(scaler, data.continous, data.raw)
    df_dataset = encode(encoder, data.categoricals, df_dataset)
    df_dataset = df_dataset[input_order + [data.target]]

    vae.fit(df_dataset.values, lambda_reg, epochs, lr, batch_size)
    vae.eval()

    return vae
