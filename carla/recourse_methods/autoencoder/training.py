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
