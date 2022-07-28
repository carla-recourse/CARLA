#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from import_essentials import *
from utils import *
import torch
from load_data import Load_dataset
from torch import nn,optim
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from itertools import product 
from post_hoc_models import Classif_model,CVAE_model
#from ray import tune
from complementary_metrics import *
import optuna 
 

class Train_CVAE_post_hoc(Load_dataset) : 
    def __init__(self, config,cat_arrays,cont_shape,loaders,dataset,epochs_cvae,lr_cvae,epochs_classif,lr_classif,lambda_3,lambda_1,cuda_name):
        super().__init__(config)
        self.cuda_device = torch.device(cuda_name)
        self.dataset = dataset 
        self.loaders = loaders 
        self.cat_arrays = cat_arrays
        self.cont_shape = cont_shape
        self.config = config 
        # Create models 
        self.classif_model = Classif_model(config,cat_arrays,cont_shape)
        self.cvae_model = CVAE_model(config,cat_arrays,cont_shape) 
        # Hyperparameters for cvae training in a post-hoc fashion 
        self.epochs_cvae = epochs_cvae
        self.epochs_classif = epochs_classif
        self.lr_cvae = lr_cvae
        self.lr_classif = lr_classif
        self.lambda_1 = lambda_1
        self.lambda_3 = lambda_3
        
    # Loss for classification model    
    def loss_classif(self,output_class,y_true) : 
        
        # Classification loss 
        CE = nn.BCELoss(reduction="sum")(output_class,y_true)
        
        return(CE)
    # Loss for CVAE model 
    def loss_cvae(self,recon_x,x,mu,logvar) : 
        
        # Loss for reconstruction (||x-x'||)
        BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
        
        # KL-divergence loss
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        return(self.lambda_3*BCE + self.lambda_1 *KLD) 
    
    # Train the classif model 
    def train_classif(self,epoch,optimizer) : 
        # Acc 
        correct = 0 
        total = 0
        train_ce_loss = 0
        for data, labels in self.loaders["train"]:
            
            output_class = self.classif_model(data.to(self.cuda_device))
            
            # Accumulate accuracy on the batch
            pred_y_batch = torch.round(output_class).squeeze()
            
            correct += (pred_y_batch == labels.to(self.cuda_device)).sum()
            total += labels.size(0)
            # Clear gradients
            optimizer.zero_grad() 
            
            # Compute loss function 
            loss = self.loss_classif(output_class,labels.unsqueeze(1).to(self.cuda_device)) 
            loss.backward()
            
            # Add loss of the batch
            train_ce_loss += loss.item()
            
            # opt step 
            optimizer.step()
            
        print('====> Epoch: {} Average classif loss train : {:.4f}'.format(epoch, train_ce_loss / len(self.loaders["train"])))
         
        print('Epoch: {} Average classif accuracy train : {:.4f} ' .format(epoch,(float(correct) / total)))   
        
    # Valid steps for classification model     
    def valid_classif(self,epoch) : 
        # Acc 
        correct = 0 
        total = 0
        valid_ce_loss = 0
        for data, labels in self.loaders["val"]:
            
            output_class = self.classif_model(data.to(self.cuda_device))
            
            # Accumulate accuracy on the batch
            pred_y_batch = torch.round(output_class).squeeze()
            
            correct += (pred_y_batch == labels.to(self.cuda_device)).sum()
            total += labels.size(0)
            
            # Compute loss function 
            loss = self.loss_classif(output_class,labels.unsqueeze(1).to(self.cuda_device)) 
            loss.backward()
            
            # Add loss of the batch
            valid_ce_loss += loss.item()
            
        print('====> Epoch: {} Average classif loss valid: {:.4f}'.format(epoch, valid_ce_loss / len(self.loaders["val"])))
         
        print('Epoch: {} Average classif accuracy valid : {:.4f} ' .format(epoch,(float(correct) / total))) 
        
    # Train the CVAE model    
    def train_cvae(self,epoch,optimizer,return_class=False) : 
        train_cvae_loss = 0    
        self.classif_model.eval()
        for data, labels in self.loaders["train"] :
            
            if return_class : 
                output_class = torch.round(self.classif_model(data.to(self.cuda_device)))
            else :
                output_class = self.classif_model(data.to(self.cuda_device))
           
            recon_batch, mu, logvar = self.cvae_model(data.to(self.cuda_device),output_class)
            loss = self.loss_cvae(recon_batch,data,mu, logvar)
            loss.backward()
            # Add loss of the batch
            train_cvae_loss += loss.item()
            # opt step 
            optimizer.step()
            
        print('====> Epoch: {} Average cvae loss train: {:.4f}'.format(epoch, train_cvae_loss / len(self.loaders["train"])))
            
            
    
     # Validation step for cvae model    
    def valid_cvae(self,epoch,return_metric=False,return_class=False) : 
        valid_cvae_loss = 0
        # Counterfactuals metrics 
        Valid_gain = []
        Valid_proximity = []
        valid_validity = 0 
        total = 0
        self.classif_model.eval()
        for data, labels in self.loaders["val"] :
            if return_class : 
                output_class = torch.round(self.classif_model(data.to(self.cuda_device)))
            else : 
                output_class = self.classif_model(data)
            recon_batch, mu, logvar = self.cvae_model(data.to(self.cuda_device),output_class)
            total += labels.size(0)
            loss = self.loss_cvae(recon_batch,data.to(self.cuda_device),mu, logvar)
            # Add loss of the batch
            valid_cvae_loss += loss.item()
           
            
           # Compute counterfactuals and metrics on the batch
            results = self.compute_counterfactuals(data.to(self.cuda_device),labels.to(self.cuda_device))
            Gain,Proximity,validity = self.compute_metrics(data.to(self.cuda_device),labels.to(self.cuda_device),results)
            valid_validity += validity 
            Valid_gain.append(Gain)
            Valid_proximity.append(Proximity)
            
        batch_mean_gain = float(torch.mean(torch.hstack(Valid_gain)))
        batch_mean_prox = float(torch.mean(torch.hstack(Valid_proximity)))
        print('====> Epoch: {} Average cvae loss valid : {:.4f}'.format(epoch, valid_cvae_loss / len(self.loaders["val"])))
        if return_metric : 
            return((float(valid_validity) / total),float(batch_mean_gain),float(batch_mean_prox))
    
    # Run optimization for classif model    
    def train_and_valid_classif(self) :
        # Send model to device 
        self.classif_model.to(self.cuda_device)
        optimizer =  optim.Adam(self.classif_model.parameters(), lr=self.lr_classif) 
        for epoch in range(1, self.epochs_classif + 1):
            self.train_classif(epoch,optimizer) 
            self.valid_classif(epoch)
        torch.save(self.classif_model.state_dict(), "save_models/" + self.name + "classif_model" + "post_hoc") 
            
    # Run optimization for cvae model (if return_class conditioned on predicted classes instead of predicted probabilites )    
    def train_and_valid_cvae(self,return_class=False) :  
        # Send model to device 
        self.cvae_model.to(self.cuda_device)
        self.classif_model.load_state_dict(torch.load("save_models/" + self.name + "classif_model" + "post_hoc"))
        optimizer =  optim.Adam(self.cvae_model.parameters(), lr=self.lr_cvae) 
        for epoch in range(1, self.epochs_cvae +1):
            self.train_cvae(epoch,optimizer,return_class=return_class) 
            self.valid_cvae(epoch,return_class=return_class)
        torch.save(self.cvae_model.state_dict(), "save_models/" + self.name + "cvae_model" + "post_hoc")
        
    # Load trained models     
    def load_weights(self,name) : 
        # Load classifier model weights
        self.classif_model.load_state_dict(torch.load("save_models/" + name + "classif_model" + "post_hoc"))
        # Load C-VAE model weights 
        self.cvae_model.load_state_dict(torch.load("save_models/" + name + "cvae_model" + "post_hoc"))
        
    # Run metrics on the test set
    def test(self) : 
        self.classif_model.eval() 
        self.cvae_model.eval()
        with torch.no_grad():
            correct = 0
            total = 0 
            # Counterfactuals metrics 
            Gain_test = []
            Proximity_test = []
            validity_test = 0  
            for data, labels in self.loaders["test"] :
                output_class = self.classif_model(data.to(self.cuda_device))
                recon_batch, mu, logvar = self.cvae_model(data.to(self.cuda_device),output_class) 
                pred_y = torch.round(output_class).squeeze()
                correct += (pred_y == labels.to(self.cuda_device)).sum()
                total += labels.size(0)
                
                # Compute counterfactuals and metrics on the batch
                results = self.compute_counterfactuals(data.to(self.cuda_device),labels.to(self.cuda_device))
                Gain,Proximity,validity = self.compute_metrics(data.to(self.cuda_device),labels.to(self.cuda_device),results)
                validity_test += validity 
                Gain_test.append(Gain)
                Proximity_test.append(Proximity)
                
        
               
        
        batch_mean_gain = float(torch.mean(torch.hstack(Gain_test)))
        batch_mean_prox = float(torch.mean(torch.hstack(Proximity_test)))
        batch_std_gain = float(torch.std(torch.hstack(Gain_test)))
        batch_std_prox = float(torch.std(torch.hstack(Proximity_test)))
        print('Test Accuracy of the prediction model on the test set : ' ,(float(correct) / total))
        print("Validity on the test set :",float(validity_test) / total)
        print('Proximity on the test set {} +/- {}:'.format(round(batch_mean_prox,3),round(batch_std_prox,3)))
        print("Gain on the test set {} +/- {}: ".format(round(batch_mean_gain,3),round(batch_std_gain,3)))
            
            
    # Function to compute counterfactuals on a given batch 
    def compute_counterfactuals(self,data,labels,return_examples=False) :
        self.classif_model.eval()
        self.cvae_model.eval()
        with torch.no_grad() : 
            # Predicted probas for examples 
            predicted_examples_proba = self.classif_model(data).squeeze(1)
           
            # Predicted classes for examples 
            label_examples = torch.round(predicted_examples_proba).long()
           
            # Pass to the model to have counterfactuals 
            counterfactuals, _ = self.cvae_model.forward_counterfactuals(data,(predicted_examples_proba < 0.5).int().unsqueeze(1),predicted_examples_proba.unsqueeze(1))
            
            # Predicted probas for counterfactuals
            predicted_counterfactuals_proba = self.classif_model(counterfactuals).squeeze(1)
            # Predicted classes for counterfactuals 
            predicted_counterfactuals_classes = torch.round(predicted_counterfactuals_proba).long()
            
             
        if return_examples :     
            return { "x" : data, # Examples
                    "cf" : counterfactuals, # Counterfactuals
                    "y_x" : label_examples, # Predicted examples classes
                    "y_c" : predicted_counterfactuals_classes, #Predicted counterfactuals classes 
                    "proba_x" : predicted_examples_proba, # Predicted probas for examples 
                    "proba_c" : predicted_counterfactuals_proba #Predicted probas for counterfactuals  
                    }
        else : 
            return  {"cf" : counterfactuals, # Counterfactuals
                    "y_x" : label_examples, # Predicted examples classes
                    "y_c" : predicted_counterfactuals_classes, #Predicted counterfactuals classes 
                    "proba_x" : predicted_examples_proba, # Predicted probas for examples 
                    "proba_c" : predicted_counterfactuals_proba #Predicted probas for counterfactuals  
                    }
            
    
    # Compute metrics on a given batch
    def compute_metrics(self,data,labels,result,not_on_batch=False) : 
        counterfactuals = result["cf"]
        label_examples = result["y_x"].flatten()
        predicted_examples_proba = result["proba_x"] 
        predicted_counterfactuals_proba = result["proba_c"]
        predicted_counterfactuals_classes = result["y_c"]
        
        labels_counterfactuals = 1-label_examples   
        
        # Validity metric
        validity = (labels_counterfactuals ==predicted_counterfactuals_classes).sum()
        if not_on_batch : 
            validity = validity /labels_counterfactuals.size(0)
        
        
         
        
        # L_1 distance metric 
        def proximity(x, c):
            return torch.abs(x - c).sum(dim=1)
        
        # Proximity on valid counterfactuals
        Proximity = proximity(data,counterfactuals)
        
        # Predicted proba for counterfactuals in columns (size= one column for each class)
        proba_vector_counterfactuals = torch.vstack(((1-predicted_counterfactuals_proba),predicted_counterfactuals_proba)).T
        
        # Predicted proba for examples in columns (size= one column for each class)
        proba_vector_examples = torch.vstack(((1-predicted_examples_proba),predicted_examples_proba)).T
        
        # Predicted probabilties of counterfactuals for counterfactual predicted classes 
        P_c = proba_vector_counterfactuals[np.arange(len(proba_vector_counterfactuals)),predicted_counterfactuals_classes]
    
        # Predicted probabilties of examples for counterfactual predicted classes 
        P_e = proba_vector_examples[np.arange(len(proba_vector_examples)),predicted_counterfactuals_classes] 
        
        
        
        # Compute gain on valid counterfactuals
        Gain = P_c-P_e 
        
        
        return(Gain,Proximity,validity) 
             
     
    # Compute complementary metrics    
    def compute_others_metrics(self,Counterfactuals,y_pred_counterfactuals,name) : 
        # Load train set 
        x_train,y_train = self.dataset.train_dataset[:] 
        # Compute predicted classes 
        self.classif_model.eval()
        with torch.no_grad() : 
            y_pred_train = torch.round(self.classif_model(x_train).squeeze(1)).long()
        
        # Laugel proximity 
        Proximity_laugel =  Compute_prox(x_train, Counterfactuals,y_pred_counterfactuals,y_pred_train,name)
        
        return(Proximity_laugel)
 
    # Hyperparameter optimization for cvae training in a post-hoc fashion    
    def objective(self,trial) : 
        self.cvae_model.eval()
        # Init params 
        epochs = trial.suggest_int("epochs", 40, 250)
        self.lambda_1,self.lambda_3 =  trial.suggest_float("lambda_1", 0.1,1) , trial.suggest_float("lambda_2", 0.1,1)
        # Init a cvae model
        self.cvae_model = CVAE_model(self.config,self.cat_arrays,self.cont_shape,self.separate_encoding)
        optimizer =  optim.Adam(self.cvae_model.parameters(), lr=1e-4) 
        for epoch in range(1, epochs + 1):
            self.train_cvae(epoch,optimizer) 
            validity_valid,gain_valid,prox_valid = self.valid_cvae(epoch,return_metric=True)
        
        return(validity_valid,gain_valid,prox_valid)
    
    # Run optimization of hyperparameters 
    def run_optuna(self) : 
        study = optuna.create_study(directions=["maximize", "maximize","minimize"])
        study.optimize(self.objective, n_trials=500)
        # Save the study 
        joblib.dump(study, "studies/study_{}_post_hoc.pkl".format(self.name))
        
        