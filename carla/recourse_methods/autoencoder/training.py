from sklearn.model_selection import train_test_split


def train_autoencoder(
    autoencoder, data, input_order, epochs=25, batch_size=64, save=True
):
    df_label_data = data.df[data.target]
    df_dataset = data.df[input_order]

    xtrain, xtest, _, _ = train_test_split(
        df_dataset.values, df_label_data.values, train_size=0.7
    )

    fitted_ae = autoencoder.train(xtrain, xtest, epochs, batch_size)

    if save:
        autoencoder.save(fitted_ae)

    return fitted_ae
