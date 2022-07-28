from typing import Dict
from typing import Union

import torch 

import numpy as np
import pandas as pd

from carla.self_explaining_model.api import SelfExplainingModel
from carla.recourse_methods.processing import (
    check_counterfactuals,
    merge_default_parameters,
)
from carla.self_explaining_model.catalog.vcnet.library.utils import fix_seed
from carla.self_explaining_model.catalog.vcnet.library.vcnet_tabular_data_v0.load_config import Load_config,load_config_dict
from carla.self_explaining_model.catalog.vcnet.library.vcnet_tabular_data_v0.join_training_network import CVAE_join,Predictor
from carla.self_explaining_model.catalog.vcnet.library.vcnet_tabular_data_v0.train_network import Train_CVAE 
from carla.self_explaining_model.catalog.vcnet.library.load_data import Load_dataset_carla
from carla.self_explaining_model.catalog.vcnet.library.load_classif_model import load_classif_model

class VCNet(SelfExplainingModel) :
    """
    Implementation of VCNet [1]_

    Parameters
    ----------
    data : carla.data.Datacatalog
        Dataset to perform the method
    hyperparams : dict
        Dictionary containing hyperparameters. See Notes below to see its content.

    Methods
    -------
    predict:
        One-dimensional prediction of ml model for an output interval of [0, 1].
    predict_proba:
        Two-dimensional probability prediction of ml model.
    get_counterfactuals:
        Generate counterfactual examples for given factuals.

    Notes
    -----
    - Hyperparams
        Hyperparameter contains important information for the recourse method to initialize.
        Please make sure to pass all values as dict with the following keys.

        * "data_name": str
            name of the dataset
        * "cvae_params": Dict
            With parameter for CVAE.

            + "train": bool, default: True
                Decides if a new Autoencoder will be learned.
            + "lr": float, default: 1e-3
                Learning rate for VAE training
            + "batch_size": int, default: 32
                Batch-size for VAE training
            + "epochs": int, default: 5
                Number of epochs to train VAE
            + "lambda_1": float, default: 1e-6
                Hyperparameter for conditional variational autoencoder.
            + "lambda_2": float, default: 1e-6
                Hyperparameter for conditional variational autoencoder.
            + "lambda_3": float, default: 1e-6
                Hyperparameter for conditional variational autoencoder.
            + "latent_size": float, default: 1e-6
                Hyperparameter for conditional variational autoencoder.
            + "latent_size_share": float, default: 1e-6
                Hyperparameter for conditional variational autoencoder.
            + "mid_reduce_size": float, default: 1e-6
                Hyperparameter for conditional variational autoencoder.
            
            

    .. [1] Guyomard
    """

    _DEFAULT_HYPERPARAMS = {   
    "name" : None ,
    "cvae_params" : {
    "train" : True,
    "lr":  0.0009,
    "batch_size": 40,
    "epochs" : 5,
    "lambda_1": 0.38,
    "lambda_2": 0.20,
    "lambda_3": 0.0001,
    "latent_size" : 7,
    "latent_size_share" : 284 , 
    "mid_reduce_size" : 142
    }
    }

    def __init__(self,data,hyperparams : Dict = None, train_cvae=False) :
        super().__init__(data)
        checked_hyperparams = merge_default_parameters(
            hyperparams, self._DEFAULT_HYPERPARAMS
        )
        make_train = checked_hyperparams["cvae_params"]["train"]
        del checked_hyperparams["cvae_params"]["train"]
        checked_hyperparams = {"name" : checked_hyperparams["name"],**checked_hyperparams["cvae_params"]}
        # Create a load dataset object 
        self.dataset = Load_dataset_carla(data,checked_hyperparams,subsample=False)

        # Prepare dataset and return dataloaders + ohe index 
        loaders,cat_arrays,cont_shape = self.dataset.prepare_data()

        ### Prepare training 
        self._training = Train_CVAE(data,checked_hyperparams,cat_arrays,cont_shape,loaders,self.dataset,ablation="remove_enc",condition="change_dec_only",cuda_name="cpu",shared_layers=True)

        if make_train :
            self._training.train_and_valid_cvae(tensorboard=True)

        # Load Vcnet model 
        self._training.load_weights()

        # Predictor part of VCnet 
        self._model = load_classif_model(self._training)
            
    @property
    def feature_input_order(self):
        # this property contains a list of the correct input order of features for the ml model
        #test = self.dataset.test.drop(columns=[self.dataset.target])
        test = self.data.df_test.drop(columns=[self.dataset.target])
        self._feature_input_order = list(test)
        return self._feature_input_order

    @property
    def backend(self):
        """
        Describes the type of backend which is used for the classifier.

        E.g., tensorflow, pytorch, sklearn, xgboost

        Returns
        -------
        str
        """
        return "pytorch"
    
    @property
    def raw_model(self):
        """
        Contains the raw ML model built on its framework

        Returns
        -------
        object
            Classifier, depending on used framework
        """
        return self._model

    def predict(self, x: Union[np.ndarray, pd.DataFrame]):
        # the predict function outputs the continuous prediction of the model, similar to sklearn.
        return torch.argmax(self._model.forward(torch.from_numpy(x.to_numpy()).float()),axis=1).detach().numpy()

    def predict_proba(self, x: Union[np.ndarray, pd.DataFrame, torch.Tensor]):
        
        x = self.get_ordered_features(x)

        if len(x.shape) != 2:
            raise ValueError("Input shape has to be two-dimensional")

        # Keep model and input on the same device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = self._model.to(device)

        if isinstance(x, pd.DataFrame):
            _x = x.values
        elif isinstance(x, torch.Tensor):
            _x = x.clone()
        else:
            _x = x.copy()

        # If the input was a tensor, return a tensor. Else return a np array.
        tensor_output = torch.is_tensor(x)
        if not tensor_output:
            _x = torch.Tensor(_x)

        # input, tensor_output = (
        #     (torch.Tensor(x), False) if not torch.is_tensor(x) else (x, True)
        # )

        _x = _x.to(device)
        output = self._model(_x)

        if tensor_output:
            return output
        else:
            return output.detach().cpu().numpy()


    def get_counterfactuals(self , factuals : pd.DataFrame,eps=0.1) :
        data = torch.from_numpy(factuals.to_numpy()).float()
        labels = None
        results = self._training.compute_counterfactuals(data,labels)
        counterfactuals = self._training.round_counterfactuals(results,eps,data)["cf"]
        #print(f"PROBA X : {self._training.round_counterfactuals(results,eps,data)['proba_x']} \n  ======================================= \n PROBA_C :{self.training.round_counterfactuals(results,eps,data)['proba_c']} ")
        return pd.DataFrame(counterfactuals.numpy(),columns=self.feature_input_order)
