# flake8: noqa
import sys

import pandas as pd

sys.path.append("../")
import numpy as np
import torch
from torch import nn

from carla.data.catalog import DataCatalog
from carla.models.catalog import MLModelCatalog
from carla.models.negative_instances import predict_negative_instances

# from carla.recourse_methods.catalog.revise
from carla.recourse_methods.api import RecourseMethod
from carla.recourse_methods.catalog.revise import VAE_model, dfDataset, train
from carla.recourse_methods.processing.counterfactuals import check_counterfactuals

# import torch.nn.functional as F


class REVISE(RecourseMethod):
    def __init__(self, model_classification, data, hyperparams):
        super().__init__(model_classification)
        self.model_classification = model_classification
        self.params = hyperparams

        self.target_column = data.target

        temp = self.encode_normalize_order_factuals(data.raw)
        temp["target"] = data.raw[self.target_column]
        self.data = dfDataset(temp)

        vae_params = hyperparams["vae_params"]
        self.model_vae = VAE_model(
            vae_params["d"],
            temp.shape[1] - 1,  # num features - target
            # vae_params["D"],
            vae_params["H1"],
            vae_params["H2"],
            vae_params["activFun"],
        )
        # try:
        self.model_vae.load("foo.pt")
        # except:
        # train(self.model_vae, self.data, hyperparams["vae_training"])

    def get_counterfactuals(
        self,
        factuals: pd.DataFrame
        # query_instance,
        # target_digit,
        # features_to_vary=None,
        # target=1,
        # feature_weights=None,
        # _lambda=0.1,
        # optimizer="adam",
        # lr=0.05,
        # max_iter=500,
    ):

        instances = factuals.copy()
        targets = 1 - instances[self.target_column]
        instances = self.encode_normalize_order_factuals(instances)
        instances[self.target_column] = targets

        # counterfactual targets
        # targets = 1 - torch.max(self.model_classification.predict(factuals))

        test_loader = torch.utils.data.DataLoader(
            dfDataset(instances), batch_size=1, shuffle=False
        )

        self.model_vae.eval()

        cfs = pd.DataFrame()

        cfs = []
        for i, (query_instance, y) in enumerate(test_loader):

            # print(i)
            # print(query_instance)
            # print(instances.iloc[i])

            # query_instance = row
            target_digit = y

            self._lambda = self.params["lambda"]

            target = torch.FloatTensor([0, 1])
            target_prediction = 1

            # if target_digit == 1:
            #     target = torch.FloatTensor([[1]])
            #     target_prediction = 1
            # elif target_digit == 0:
            #     target = torch.FloatTensor([[0]])
            #     target_prediction = 0

            # feature_weights = torch.ones(
            #     query_instance.shape[0]
            # )  # might be useful - for now all weights are 1
            # feature_weights = torch.FloatTensor(feature_weights)

            # cf_initialize = torch.zeros(query_instance.shape)
            # cf_initialize = torch.FloatTensor(cf_initialize)

            z = (
                self.model_vae.encode(query_instance)[0]
                .clone()
                .detach()
                .requires_grad_(True)
            )

            if self.params["optimizer"] == "adam":
                optim = torch.optim.Adam([z], self.params["lr"])
                # z.requires_grad = True
            else:
                optim = torch.optim.RMSprop([z], self.params["lr"])

            counterfactuals = []  # all possible counterfactuals
            distances = (
                []
            )  # distance of the possible counterfactuals from the intial value - considering distance as the loss function (can even change it just the distance)
            all_loss = []

            for i in range(self.params["max_iter"]):
                cf = self.model_vae.decode(z)
                output = self.model_classification.predict_proba(cf[0])[0]
                # print(output, "predicted probability")
                # print(output)
                _, predicted = torch.max(output, 0)
                # print(predicted)
                if predicted == target_prediction:
                    # print(predicted, target_prediction)
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

            # all_loss = torch.stack(all_loss)
            # all_loss = all_loss.detach().numpy()
            # plt.plot(all_loss)
            # plt.show()  # plot the loss

            # if not len(counterfactuals):
            #     cf[0][:] = np.nan
            #     cfs.append([cf[0].detach().numpy()[0], np.nan])
            #     print("fail")
            #     continue
            #     # return cf[
            #     #     0
            #     # ].numpy()  # if no counterfactual is present, returing the last counterfactual - Needs to be changed
            # print("succes")

            # Choose the nearest counterfactual
            if len(counterfactuals):
                print("succes")
                counterfactuals = torch.stack(counterfactuals)
                distances = torch.stack(distances)

                ## COMMENTED BECAUSE OF "MYPY" COMPLAINING
                ## UNCOMMENT FOR USE
                # counterfactuals = counterfactuals.detach().numpy()
                # distances = distances.detach().numpy()

                index = np.argmin(distances)
                counterfactuals = counterfactuals[index]
            else:
                print("fail")
                counterfactuals = cf[0].detach().numpy()

            cf_df = check_counterfactuals(self.model_classification, counterfactuals)

            # cfs.append(counterfactuals[index][0])
            cfs.append(cf_df)

        print("done")
        print(cfs)

        result = pd.concat(cfs)
        # TODO right now this also appends the correct cf-label 1 to the nan rows
        result[self.target_column] = targets
        result.columns = instances.columns
        return result

    def compute_loss(self, cf_initialize, query_instance, target, i):

        loss_function = nn.BCELoss()
        # loss_function = nn.CrossEntropyLoss()
        output = self.model_classification.predict_proba(cf_initialize)[0]
        loss1 = loss_function(output, target)  # classification loss
        # loss2 = torch.sum((cf_initialize - query_instance) ** 2)  # distance loss
        loss2 = torch.norm((cf_initialize - query_instance), 1)
        # print("loss for", i)
        # print(loss1, "\t", loss2)
        total_loss = loss1 + self._lambda * loss2
        # print(total_loss)
        return total_loss


def revise():
    data_name = "adult"
    data = DataCatalog(data_name)

    model = MLModelCatalog(data, "ann", backend="pytorch", encoding_method="Binary")
    # get factuals
    factuals = predict_negative_instances(model, data)
    test_factual = factuals.iloc[:20]

    vae_params = {
        "d": 8,  # latent space
        "D": test_factual.shape[1],  # input size
        "H1": 512,
        "H2": 256,
        "activFun": nn.ReLU(),
    }

    vae_training = {"lambda_reg": 1e-6, "epochs": 5, "lr": 1e-3, "batch_size": 32}

    hyperparams = {
        "lambda": 1,
        "optimizer": "adam",
        "lr": 0.05,
        "max_iter": 500,
        "vae_params": vae_params,
        "vae_training": vae_training,
    }

    explainer = REVISE(model, data, hyperparams)
    df_cfs = explainer.get_counterfactuals(test_factual)

    assert test_factual.shape[0] == df_cfs.shape[0]
    assert (df_cfs.columns == model.feature_input_order + [data.target]).all()


if __name__ == "__main__":
    revise()
