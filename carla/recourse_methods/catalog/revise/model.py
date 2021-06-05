import numpy as np
import pandas as pd
import torch
from torch import nn

from carla.recourse_methods.api import RecourseMethod
from carla.recourse_methods.autoencoder import (
    Dataloader,
    VariationalAutoencoder,
    train_variational_autoencoder,
)
from carla.recourse_methods.processing.counterfactuals import check_counterfactuals


class Revise(RecourseMethod):
    def __init__(self, model_classification, data, hyperparams) -> None:
        super().__init__(model_classification)
        self.params = hyperparams

        self._target_column = data.target

        df_enc_norm_data = self.encode_normalize_order_factuals(
            data.raw, with_target=True
        )

        vae_params = hyperparams["vae_params"]
        self.vae = VariationalAutoencoder(
            self.params["data_name"],
            vae_params["d"],
            df_enc_norm_data.shape[1] - 1,  # num features - target
            vae_params["H1"],
            vae_params["H2"],
            vae_params["activFun"],
        )

        if vae_params["train"]:
            self.vae = train_variational_autoencoder(
                self.vae,
                self._mlmodel.data,
                self._mlmodel.scaler,
                self._mlmodel.encoder,
                self._mlmodel.feature_input_order,
                lambda_reg=vae_params["lambda_reg"],
                epochs=vae_params["epochs"],
                lr=vae_params["lr"],
                batch_size=vae_params["batch_size"],
            )
        else:
            try:
                self.vae.load(df_enc_norm_data.shape[1] - 1)
            except FileNotFoundError as exc:
                raise FileNotFoundError(
                    "Loading of Autoencoder failed. {}".format(str(exc))
                )

    def get_counterfactuals(self, factuals: pd.DataFrame) -> pd.DataFrame:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        df_enc_norm_fact = factuals.copy()
        # targets = 1 - df_enc_norm_fact[self._target_column]
        df_enc_norm_fact = self.encode_normalize_order_factuals(
            df_enc_norm_fact, with_target=True
        )
        # df_enc_norm_fact[self._target_column] = targets
        df_enc_norm_fact[self._target_column] = 1

        test_loader = torch.utils.data.DataLoader(
            Dataloader(df_enc_norm_fact.values), batch_size=1, shuffle=False
        )

        cfs = []
        for i, (query_instance, y) in enumerate(test_loader):

            self._lambda = self.params["lambda"]

            target = torch.FloatTensor([0, 1]).to(device)
            target_prediction = 1

            z = self.vae.encode(query_instance)[0].clone().detach().requires_grad_(True)

            if self.params["optimizer"] == "adam":
                optim = torch.optim.Adam([z], self.params["lr"])
                # z.requires_grad = True
            else:
                optim = torch.optim.RMSprop([z], self.params["lr"])

            counterfactuals = []  # all possible counterfactuals
            # distance of the possible counterfactuals from the intial value -
            # considering distance as the loss function (can even change it just the distance)
            distances = []
            all_loss = []

            for i in range(self.params["max_iter"]):
                cf = self.vae.decode(z)
                output = self._mlmodel.predict_proba(cf[0])[0]
                _, predicted = torch.max(output, 0)

                if predicted == target_prediction:
                    counterfactuals.append(cf[0])

                z.requires_grad = True
                loss = self.compute_loss(cf[0], query_instance, target, i)
                all_loss.append(loss)
                if predicted == target_prediction:
                    distances.append(loss)
                loss.backward()
                optim.step()
                optim.zero_grad()
                cf[0].detach_()

            # Choose the nearest counterfactual
            if len(counterfactuals):
                print("succes")
                torch_counterfactuals = torch.stack(counterfactuals)
                torch_distances = torch.stack(distances)

                torch_counterfactuals = torch_counterfactuals.detach().numpy()
                torch_distances = torch_distances.detach().numpy()

                index = np.argmin(torch_distances)
                counterfactuals = torch_counterfactuals[index]
            else:
                print("fail")
                counterfactuals = cf[0].detach().numpy()

            cf_df = check_counterfactuals(self._mlmodel, counterfactuals)

            # cfs.append(counterfactuals[index][0])
            cfs.append(cf_df)

        print("done")
        print(cfs)

        result = pd.concat(cfs)
        # TODO right now this also appends the correct cf-label 1 to the nan rows
        targets = "Lösch mich"
        result[self._target_column] = targets
        result.columns = df_enc_norm_fact.columns
        return result

    def compute_loss(self, cf_initialize, query_instance, target, i):

        loss_function = nn.BCELoss()
        output = self._mlmodel.predict_proba(cf_initialize)[0]

        # classification loss
        loss1 = loss_function(output, target)
        # distance loss
        loss2 = torch.norm((cf_initialize - query_instance), 1)

        return loss1 + self._lambda * loss2
