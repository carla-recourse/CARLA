# flake8: noqa
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

# from MLmodels.VAE import model
# from MLmodels.data_loader import csvDataset

vae_params = {
    "d": 1,  # latent space
    "D": 2,  # input size
    "H1": 512,
    "H2": 256,
    "activFun": nn.ReLU(),
}

train_params = {"lambda_reg": 1e-6, "epochs": 50, "lr": 1e-3, "batch_size": 32}


def train(model, train_dataset, train_params, save_path="foo.pt"):

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=train_params["batch_size"], shuffle=True
    )

    initial = int(0.33 * train_params["epochs"])
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_params["lr"],
        weight_decay=train_params["lambda_reg"],
    )

    criterion = nn.MSELoss()

    # Train the VAE with the new prior
    ELBO = np.zeros((train_params["epochs"], 1))
    for epoch in tqdm(range(train_params["epochs"])):

        # Initialize the losses
        train_loss = 0
        train_loss_num = 0

        # Train for all the batches
        for batch_idx, (data, _) in enumerate(train_loader):

            data = data.view(data.shape[0], -1)

            # forward pass
            MU_X_eval, LOG_VAR_X_eval, Z_ENC_eval, MU_Z_eval, LOG_VAR_Z_eval = model(
                data
            )

            # Compute the regularization parameter
            if initial == 0:
                r = 0
            else:
                r = 1.0 * epoch / initial
                if r > 1.0:
                    r = 1.0

            # The VAE loss
            # loss = model.VAE_loss(
            #     x=data,
            #     mu_x=MU_X_eval,
            #     log_var_x=LOG_VAR_X_eval,
            #     mu_z=MU_Z_eval,
            #     log_var_z=LOG_VAR_Z_eval,
            #     r=r,
            # )
            reconstruction = MU_X_eval
            mse_loss = criterion(reconstruction, data)
            loss = model.VAE_loss(mse_loss, MU_Z_eval, LOG_VAR_Z_eval)
            # Update the parameters
            optimizer.zero_grad()
            # Compute the loss
            loss.backward()
            # Update the parameters
            optimizer.step()

            # Collect the ways
            train_loss += loss.item()
            train_loss_num += 1

        ELBO[epoch] = train_loss / train_loss_num
        if epoch % 10 == 0:
            print(
                "[Epoch: {}/{}] [objective: {:.3f}]".format(
                    epoch, train_params["epochs"], ELBO[epoch, 0]
                )
            )
            # torch.save(model.state_dict(), "checkpoints/{}.pt".format(epoch))

    ELBO_train = ELBO[epoch, 0].round(2)
    print("[ELBO train: " + str(ELBO_train) + "]")
    del MU_X_eval, MU_Z_eval, Z_ENC_eval
    del LOG_VAR_X_eval, LOG_VAR_Z_eval
    print("Training finished")

    plt.figure()
    plt.plot(ELBO)
    plt.show()

    if save_path is not None:
        torch.save(model.state_dict(), save_path)


def test(model, test_dataset, train_params):

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=train_params["batch_size"], shuffle=True
    )

    test_loss = 0
    loss = nn.BCELoss()
    for data, _ in test_loader:
        data = data.view(data.shape[0], -1)
        # forward pass
        MU_X_eval, LOG_VAR_X_eval, Z_ENC_eval, MU_Z_eval, LOG_VAR_Z_eval = model(data)
        test_loss += loss(
            data,
        )


# if __name__ == "__main__":
#     # The model and the optimizer for the VAE
#     model = model.VAE_model(vae_params["d"], vae_params["D"], vae_params["H1"], vae_params["H2"],
#                             vae_params["activFun"])
#
#     train_dataset = csvDataset("../../data/SCM/train_data.csv")
#     # test_dataset = csvDataset("../../data/SCM/test_data.csv")
#     # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=training_params["batch_size"], shuffle=True)
#
#     train(model, train_dataset, train_params)
