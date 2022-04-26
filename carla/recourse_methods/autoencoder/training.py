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


def train_variational_autoencoder(
    vae,
    data,
    input_order,
    lambda_reg=1e-6,
    epochs=5,
    lr=1e-3,
    batch_size=32,
):
    df_dataset = data.df[input_order + [data.target]]

    vae.fit(df_dataset.values, lambda_reg, epochs, lr, batch_size)
    vae.eval()

    return vae
