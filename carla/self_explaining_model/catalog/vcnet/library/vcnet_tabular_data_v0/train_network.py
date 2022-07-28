#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from carla.self_explaining_model.catalog.save_load import get_home
from import_essentials import *
from utils import *
import torch
from load_data import Load_dataset_base,Load_dataset_carla
from torch import nn,optim
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from itertools import product 
from vcnet_tabular_data_v0.join_training_network import CVAE_join,Predictor
from metrics import * 
#from ray import tune
import optuna 
import joblib
import pathlib 
import os
main_path =  str(pathlib.Path().resolve()) + '/'
main_path = main_path +  "/vcnet_tabular_data_v0/"
#from sklearn.model_selection import train_test_split
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.metrics import accuracy_score
from used_carla import set_carla

    
# Train the network 
class Train_CVAE(Load_dataset_carla if set_carla() else Load_dataset_base) :
    def __init__(self, dataset_config_dict,model_config_dict,cat_arrays,cont_shape,loaders,dataset,ablation,condition,cuda_name,shared_layers=True):
        super().__init__(dataset_config_dict,model_config_dict)  
        self.dataset = dataset  
        self.loaders = loaders 
        self.cat_arrays = cat_arrays 
        self.cont_shape = cont_shape
        self.cuda_device = torch.device(cuda_name)
        self.ablation = ablation 
        self.condition = condition 
        self.shared_layers = shared_layers
        self.used_carla = set_carla()
        self.dataset_config_dict = dataset_config_dict
        self.model = CVAE_join(dataset_config_dict,model_config_dict,self.cat_arrays,self.cont_shape,self.ablation,self.condition,shared_layers=self.shared_layers)
        self.model.to(self.cuda_device)
         
    def loss_functions(self,recon_x, x, mu, logvar,output_class,y_true):
        # Loss for reconstruction (||x-x'||)
        
        #BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
        
        
        BCE = torch.mean((recon_x-x)**2)
        # KL-divergence loss
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Classification loss 
        CE = nn.BCELoss(reduction="sum")(output_class,y_true)
        
        # Return individual losses + weighted sum of losses
        return [BCE,KLD,CE],self.lambda_3*BCE + self.lambda_1 *KLD + self.lambda_2*CE
    
    
    # Train the network on a batch
    def train(self,epoch,optimizer,tb,return_metric=False,tensorboard=False): 
        self.model.train()
        # Losses
        train_bce_loss = 0
        train_kld_loss = 0 
        train_ce_loss = 0
        train_loss = 0
        # Acc 
        correct = 0 
        total = 0
        # Counterfactuals metrics 
        Train_gain = []
        Train_proximity = []
        Train_sparsity = []
        train_validity = 0 
        
        
        for data, labels in self.loaders["train"]:
            # Pass batch to the model 
            recon_batch, mu, logvar,output_class = self.model(data.to(self.cuda_device))
            
            
            # Accumulate accuracy on the batch
            pred_y_batch = torch.round(output_class).squeeze()
            
            correct += (pred_y_batch == labels.to(self.cuda_device)).sum()
            total += labels.size(0)
            # Clear gradients
            optimizer.zero_grad() 
            # Compute loss function 
            losses,loss = self.loss_functions(recon_batch, data.to(self.cuda_device), mu, logvar,output_class,labels.unsqueeze(1).to(self.cuda_device))
            loss.backward()
            # Add loss of the batch
            train_loss += loss.item()
            # Add invididual losses over the batch 
            
            bce_loss,kld_loss,ce_loss = losses
            train_bce_loss += bce_loss.item()
            train_kld_loss += kld_loss.item()
            train_ce_loss += ce_loss.item()
            # Compute counterfactuals and metrics on the batch
            results = self.compute_counterfactuals(data.to(self.cuda_device),labels.to(self.cuda_device))
            Sparsity,Gain,Proximity,validity = compute_metrics(data.to(self.cuda_device),labels,results)
            train_validity += validity 
            Train_gain.append(Gain)
            Train_proximity.append(Proximity)
            Train_sparsity.append(Sparsity)
            # opt step 
            optimizer.step()
            
        print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(self.loaders["train"])))
         
        print('Epoch: {} Average Accuracy : {:.4f} ' .format(epoch,(float(correct) / total)))
        
        batch_mean_gain = float(torch.mean(torch.hstack(Train_gain)))
        batch_mean_prox = float(torch.mean(torch.hstack(Train_proximity)))
        batch_mean_sparsity = float(torch.mean(torch.hstack(Train_sparsity)))
        if tensorboard : 
            # Tensorboard tracking losses
            tb.add_scalars('Overall_loss', {'train' :train_loss / len(self.loaders["train"])},epoch)
    
            tb.add_scalars("Bce_loss", {"train" : train_bce_loss / len(self.loaders["train"])}, epoch)
            tb.add_scalars("Kld_loss", {"train" : train_kld_loss / len(self.loaders["train"])}, epoch)
            tb.add_scalars("Ce_loss", {"train" : train_ce_loss / len(self.loaders["train"])}, epoch)
            
            # Tensorboard tracking metrics 
            tb.add_scalars("Accuracy", {"train" : (float(correct) / total)},epoch)
            tb.add_scalars("Validity", {"train" : (float(train_validity) / total)},epoch) 
            tb.add_scalars("Gain", {"train" : batch_mean_gain},epoch) 
            tb.add_scalars("Proximity", {"train" : batch_mean_prox},epoch) 
            tb.add_scalars("Sparsity", {"train" : batch_mean_sparsity},epoch) 
        # Return metrics on train set  (accuracy,validity,gain,proximity)
        if return_metric : 
            return((float(correct) / total),(float(train_validity) / total),float(batch_mean_gain),float(batch_mean_prox),float(batch_mean_sparsity))
        
    # Validation 
    def valid(self,epoch,tb,return_metric=False,tensorboard=False) :
        self.model.eval()
        valid_bce_loss = 0 
        valid_kld_loss = 0 
        valid_ce_loss = 0
        valid_loss = 0.0
        correct = 0 
        total = 0 
        
        # Counterfactuals metrics 
        Valid_gain = []
        Valid_proximity = []
        Valid_sparsity =  []
        valid_validity = 0 
        
        with torch.no_grad() :
            for data, labels in self.loaders["val"]: 
                # Pass batch to the model 
                recon_batch, mu, logvar,output_class = self.model(data.to(self.cuda_device))
                
                # Accumulate accuracy on the batch
                pred_y_batch = torch.round(output_class).squeeze()
                correct += (pred_y_batch == labels.to(self.cuda_device)).sum()
                total += labels.size(0)
        
                # Compute loss function 
                losses,loss = self.loss_functions(recon_batch, data.to(self.cuda_device), mu, logvar,output_class,labels.unsqueeze(1).to(self.cuda_device))
                
                # Add loss of the batch
                valid_loss += loss.item()
                
                bce_loss,kld_loss,ce_loss = losses
                valid_bce_loss += bce_loss.item()
                valid_kld_loss += kld_loss.item()
                valid_ce_loss += ce_loss.item()
                # Compute counterfactuals and metrics on the batch
                results = self.compute_counterfactuals(data.to(self.cuda_device),labels.to(self.cuda_device))
                Sparsity,Gain,Proximity,validity = compute_metrics(data.to(self.cuda_device),labels,results)
                valid_validity += validity 
                Valid_gain.append(Gain)
                Valid_proximity.append(Proximity)
                Valid_sparsity.append(Sparsity)
               
        print('====> Epoch: {} Average loss on validation step: {:.4f}'.format(epoch, valid_loss / len(self.loaders["val"])))
        
        print('Epoch: {} Average Accuracy on validation : {:.4f} ' .format(epoch,(float(correct) / total)))
        
        batch_mean_gain = float(torch.mean(torch.hstack(Valid_gain)))
        batch_mean_prox = float(torch.mean(torch.hstack(Valid_proximity)))
        batch_mean_sparsity = float(torch.mean(torch.hstack(Valid_sparsity)))
        if tensorboard : 
            # Tensorboard tracking losses
            tb.add_scalars("Overall_loss", {"valid" : valid_loss / len(self.loaders["val"])}, epoch)
            
            tb.add_scalars("Bce_loss", {"valid" : valid_bce_loss / len(self.loaders["val"])}, epoch)
            tb.add_scalars("Kld_loss", {"valid" : valid_kld_loss / len(self.loaders["val"])}, epoch)
            tb.add_scalars("Ce_loss", {"valid" : valid_ce_loss / len(self.loaders["val"])}, epoch)
            
            
            # Tensorboard tracking metrics 
            
            tb.add_scalars("Accuracy", {"valid" : (float(correct) / total)},epoch)
            tb.add_scalars("Validity", {"valid" : (float(valid_validity) / total)},epoch) 
            tb.add_scalars("Gain", {"valid" : batch_mean_gain},epoch) 
            tb.add_scalars("Proximity", {"valid" : batch_mean_prox},epoch) 
        # Return metrics on valid set  (accuracy,validity,gain,proximity)
        if return_metric : 
            return((float(correct) / total),(float(valid_validity) / total),float(batch_mean_gain),float(batch_mean_prox),float(batch_mean_sparsity))
        
    # Training and validation of the network + save the model     
    def train_and_valid_cvae(self,tensorboard=False) :
        tb = SummaryWriter(main_path + "/runs/logs")
        optimizer =  optim.Adam(self.model.parameters(), lr=self.lr) 
        for epoch in range(1, self.epochs + 1):
            # Train
            self.train(epoch,optimizer,tb,tensorboard=tensorboard)
              
            # Valid 
            self.valid(epoch,tb,tensorboard=tensorboard)
            
        tb.close()
        
        #torch.save(self.model.state_dict(),main_path + "save_models/" + self.name + "join-CVAE_" + str(self.ablation) + "_" + str(self.condition)+"shared_layers_" + str(self.shared_layers) + "used_carla" + str(self.used_carla)) 
        #torch.save(self.model.state_dict(),get_home())
        self.save()
    # Run metrics on the test set  
    def test(self,df,save_dataframe=False) : 
        round_number = 3
        self.model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            # Counterfactuals metrics 
            Gain_test = []
            Proximity_test = []
            Sparsity_test = []
            validity_test = 0 
            if  self.dataset.subsample : 
                loader_test = self.loaders["sample"]
            else : 
                loader_test = self.loaders["test"]
            
            for data, labels in loader_test :
                recon_batch, mu, logvar,output_class = self.model(data.to(self.cuda_device))
                pred_y = torch.round(output_class).squeeze()
                correct += (pred_y == labels.to(self.cuda_device)).sum()
                total += labels.size(0)
                
                # Compute counterfactuals and metrics on the batch
                results = self.compute_counterfactuals(data.to(self.cuda_device),labels.to(self.cuda_device))
                Sparsity,Gain,Proximity,validity = compute_metrics(data.to(self.cuda_device),labels.to(self.cuda_device),results)
                validity_test += validity 
                Gain_test.append(Gain)
                Proximity_test.append(Proximity)
                Sparsity_test.append(Sparsity)
        
                
        #np.savetxt("Gain_test",torch.hstack(Gain_test).numpy())
        #print(torch.hstack(Gain_test)[:128])
        batch_mean_gain = float(torch.mean(torch.hstack(Gain_test)))
        batch_mean_prox = float(torch.mean(torch.hstack(Proximity_test)))
        batch_mean_sparsity = float(torch.mean(torch.hstack(Sparsity_test)))
        batch_std_gain = float(torch.std(torch.hstack(Gain_test)))
        batch_std_prox = float(torch.std(torch.hstack(Proximity_test)))
        batch_std_sparsity = float(torch.std(torch.hstack(Sparsity_test)))
        print('Test Accuracy of the prediction model on the test set : ' ,(float(correct) / total))
        print("Validity on the test set :",float(validity_test) / total)
        print('Proximity on the test set {} +/- {}:'.format(round(batch_mean_prox,round_number),round(batch_std_prox,round_number)))
        print("Gain on the test set {} +/- {}: ".format(round(batch_mean_gain,round_number),round(batch_std_gain,round_number)))
        print("Sparsity on the test set {} +/- {}:".format(round(batch_mean_sparsity,3),round(batch_std_sparsity,3)))
        # Save results in a dataframe
        if save_dataframe : 
            df["Accuracy"].loc[str(self.name)] = float(correct) / total
            df["Validity"].loc[str(self.name)] = float(validity_test) / total
            df["Proximity"].loc[str(self.name)] = str(round(batch_mean_prox,round_number)) + "+/-" + str(round(batch_std_prox,round_number))
            df["Prediction_gain"].loc[str(self.name)] = str(round(batch_mean_gain,round_number)) + "+/-" + str(round(batch_std_gain,round_number))
        
        
    def save(self):
        cache_path = get_home()

        save_path = os.path.join(
            cache_path,
            "{}join-CVAE_{}_{}shared_layers_{}.{}".format(self.dataset.name, self.ablation,self.condition,self.shared_layers, "pt")
        )

        torch.save(self.model.state_dict(), save_path)

    def load_weights(self):
        cache_path = get_home()

        load_path = os.path.join(
            cache_path,
            "{}join-CVAE_{}_{}shared_layers_{}.{}".format(self.dataset.name, self.ablation,self.condition,self.shared_layers, "pt"),
        )

        self.model.load_state_dict(torch.load(load_path))

        return self    
    """ # Load weights for a trained model 
    def load_weights(self,name) : 
        cache_path = get_home()
        self.model.load_state_dict(torch.load(main_path + "save_models/" + name+ "join-CVAE_" + str(self.ablation) + "_" + str(self.condition) +"shared_layers_" + str(self.shared_layers) + 		"used_carla" + str(self.used_carla),map_location=torch.device('cpu'))) 
              """
        
    # Save prediction newtork part to use it for post_hoc optimization
    def save_post_hoc_prediction_model(self) : 
        # Load weights of the predictor part 
        self.prediction_model = Predictor(**self.model.kwargs)
        state_dict = self.prediction_model.state_dict().copy()
        with torch.no_grad():
            for layer in self.model.state_dict():
                if layer in state_dict : 
                    state_dict[layer] = self.model.state_dict()[layer]

        self.prediction_model.load_state_dict(state_dict)
        # Save model as a pytorch model 
        save_path = str(pathlib.Path().resolve()) + '/'
        torch.save([self.prediction_model.kwargs, self.prediction_model.state_dict()], save_path + "post_hoc_counterfactuals/prediction_models/vcnet0/prediction_model_" + self.name)
        # Save model as a tensorflow model 
        X_test,y_test = self.dataset.val_dataset[:] 
        dummy_input = X_test[0].reshape(1, -1).float().to(self.cuda_device) 
        torch.onnx.export(self.prediction_model, dummy_input,save_path + "post_hoc_counterfactuals/prediction_models/vcnet0/" + "prediction_model_" + self.name + ".onnx", input_names=['input'], output_names=['output'],opset_version=11)
        
        
        
    # Function to compute counterfactuals on a given batch 
    def compute_counterfactuals(self,data,labels,laugel_metric=False) :
        self.model.eval()
        with torch.no_grad() : 
            # Predicted probas for examples 
            predicted_examples_proba = self.model.forward_pred(data).squeeze(1)
           
            # Predicted classes for examples 
            label_examples = torch.round(predicted_examples_proba).long()
            
            # Pass to the model to have counterfactuals 
            counterfactuals, z = self.model.forward_counterfactuals(data,(predicted_examples_proba < 0.5).int().unsqueeze(1))
            #counterfactuals, _ = self.model.forward_counterfactuals(data,(predicted_examples_proba).unsqueeze(1))
            # Predicted probas for counterfactuals
            predicted_counterfactuals_proba = self.model.forward_pred(counterfactuals).squeeze(1)
            # Predicted classes for counterfactuals 
            predicted_counterfactuals_classes = torch.round(predicted_counterfactuals_proba).long()            
             
        if laugel_metric :
            # Load train set 
            x_train,y_train = self.dataset.train_dataset[:] 
            with torch.no_grad() : 
                y_pred_train = torch.round(self.model.forward_pred(x_train.to(self.cuda_device)).squeeze(1)).long()
                
            return { "x_train" : x_train, # train examples 
                    "y_x_train" : y_pred_train, # Predicted example class for train set 
                    "cf" : counterfactuals, # Counterfactuals
                    "z" : z, # Latent space representation
                    "y_x" : label_examples, # Predicted examples classes
                    "y_c" : predicted_counterfactuals_classes, #Predicted counterfactuals classes 
                    "proba_x" : predicted_examples_proba, # Predicted probas for examples 
                    "proba_c" : predicted_counterfactuals_proba #Predicted probas for counterfactuals  
                    }
        else : 
            return  {"cf" : counterfactuals, # Counterfactuals
                     "z" : z, # Latent space representation
                    "y_x" : label_examples, # Predicted examples classes
                    "y_c" : predicted_counterfactuals_classes, #Predicted counterfactuals classes 
                    "proba_x" : predicted_examples_proba, # Predicted probas for examples 
                    "proba_c" : predicted_counterfactuals_proba #Predicted probas for counterfactuals  
                    }
    
    
    # Function to compute counterfactuals on a given batch 
    def compute_counterfactuals_custom_proba(self,data,labels,proba,laugel_metric=False) :
        self.model.eval()
        with torch.no_grad() : 
            # Predicted probas for examples 
            predicted_examples_proba = self.model.forward_pred(data).squeeze(1)
           
            # Predicted classes for examples 
            label_examples = torch.round(predicted_examples_proba).long()
            
            # Pass to the model to have counterfactuals 
            counterfactuals, _ = self.model.forward_counterfactuals(data,proba)
            #counterfactuals, _ = self.model.forward_counterfactuals(data,(predicted_examples_proba).unsqueeze(1))
            # Predicted probas for counterfactuals
            predicted_counterfactuals_proba = self.model.forward_pred(counterfactuals).squeeze(1)
            # Predicted classes for counterfactuals 
            predicted_counterfactuals_classes = torch.round(predicted_counterfactuals_proba).long()
            
             
        if laugel_metric :
            # Load train set 
            x_train,y_train = self.dataset.train_dataset[:] 
            with torch.no_grad() : 
                y_pred_train = torch.round(self.model.forward_pred(x_train.to(self.cuda_device)).squeeze(1)).long()
                
            return { "x_train" : x_train, # train examples 
                    "y_x_train" : y_pred_train, # Predicted example class for train set 
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
    
    
    
            
    # Function to round counterfactuals values (when the perturbation is lower than eps counterfactual value is equal example value )
    def round_counterfactuals(self,result,eps,X) : 
        
        counterfactuals = result["cf"] 
        predicted_counterfactuals_classes = result["y_c"] 
        
        # Rounded counterfactuals
        counterfactuals_round = torch.clone(counterfactuals) 
        counterfactuals_round[np.abs(X-counterfactuals) < eps] = X[np.abs(X-counterfactuals) < eps]
        
        # Percentage of counterfactuals that are still valid 
        
        # Predicted probas for counterfactuals
        predicted_counterfactuals_round_proba = self.model.forward_pred(counterfactuals_round).squeeze(1)
        # Predicted classes for counterfactuals 
        predicted_counterfactuals_round_classes = torch.round(predicted_counterfactuals_round_proba).long()
        
        percentage = 1 - torch.sum(predicted_counterfactuals_round_classes==predicted_counterfactuals_classes) /predicted_counterfactuals_round_classes.shape[0] 
        print("Percentage of counterfactuals that are not still valid : {}".format(float(percentage)))
        
        result["cf"] = counterfactuals_round
        result["y_c"] = predicted_counterfactuals_round_classes
        result["proba_c"] = predicted_counterfactuals_round_proba
        
        return(result)

          
    # Objective function for hyperparams optimization    
    def objective(self,trial) :  
        # Init params 
        epochs = trial.suggest_int("epochs", 40, 250)
        #epochs = trial.suggest_int("epochs", 2, 4)
        lr=trial.suggest_float("learning_rate",1e-5,1e-3,log=True)
        #ablation = trial.suggest_categorical("ablation",[None,"remove_enc"])
        self.lambda_1,self.lambda_2,self.lambda_3 =  trial.suggest_float("lambda_1", 0.001,1,step=0.001) , trial.suggest_float("lambda_2", 0.001,1,step=0.001) , trial.suggest_float("lambda_3", 0.0001,0.001,step=0.0001)
        batch_size = trial.suggest_int("batch_size",32,128)
        latent_size = trial.suggest_int("latent_size",2,40)
        mide_reduce_size = trial.suggest_int("mide_reduce_size",16,256)
        model_config_dict = {   "name" : self.name ,
            "lr": lr,
            "batch_size": batch_size,
            "epochs" : epochs,
            "lambda_1": self.lambda_1,
            "lambda_2": self.lambda_2,
            "lambda_3": self.lambda_3,
            "latent_size" : latent_size,
            "latent_size_share" : mide_reduce_size*2, 
            "mid_reduce_size" : mide_reduce_size
        }
        # Init a model 
        self.model = CVAE_join(dataset_config_dict = self.dataset_config_dict,model_config_dict = model_config_dict, 
                        cat_arrays = self.cat_arrays,cont_shape = self.cont_shape,ablation=None, condition = self.condition,
                        shared_layers=self.shared_layers)
        self.model.to(self.cuda_device)
        optimizer =  optim.Adam(self.model.parameters(), lr=lr)
        tb = None 
        for epoch in range(1, epochs + 1):
            # Train
            self.train(epoch,optimizer,tb,tensorboard=False)
              
            # Valid 
            accuracy_valid,validity_valid,gain_valid,prox_valid = self.valid(epoch,tb,return_metric=True,tensorboard=False)[0:4]
            
        return(accuracy_valid,validity_valid,gain_valid,prox_valid)
   
    # Run optimization of hyperparameters        
    def run_optuna(self) : 
        study = optuna.create_study(directions=["maximize", "maximize","maximize","minimize"])
        study.optimize(self.objective, n_trials=500)
        # Save the study 
        joblib.dump(study, main_path + "studies/study_{}.pkl".format(self.name))
        
        
    # Save results for ploting counterfactuals for toy datasets     
    def save_for_plot_toy(self) : 
        X,y = self.dataset.val_dataset[:]
        results = self.compute_counterfactuals(X, y)
        counterfactuals = results["cf"].numpy()
        predicted_counterfactual_class = results["y_c"].numpy()
        predicted_example_class =  results["y_x"].numpy()
         
        # Save results 
        save_name = self.name +"_test"
        np.savetxt(main_path + "toy_dataset_plot/" + save_name, X)
        np.savetxt(main_path+ "toy_dataset_plot/counterfactuals_" + save_name, counterfactuals)
        np.savetxt(main_path + "toy_dataset_plot/predicted_examples_classes_" + save_name, predicted_example_class)
        np.savetxt(main_path + "toy_dataset_plot/predicted_counterfactuals_classes_" + save_name, predicted_counterfactual_class)
        np.savetxt(main_path + "toy_dataset_plot/true_examples_classes_" + save_name, y)
        
    # Mehsgrid for 2d plot on toy datasets     
    def save_contourf_for_plot(self) :
        plot_step = 0.02
        X,y = self.dataset.val_dataset[:]
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                             np.arange(y_min, y_max, plot_step))
                
        Z = self.model.forward_pred(torch.from_numpy(np.c_[xx.ravel(), yy.ravel()]).float()) 
        
        
        Z = Z.reshape(xx.shape).detach().numpy()
        
        np.savetxt(main_path + "toy_dataset_plot/contourf_" +  self.name +"_test", Z)
        np.savetxt(main_path + "toy_dataset_plot/xx_" +  self.name +"_test", xx)
        np.savetxt(main_path + "toy_dataset_plot/yy_" +  self.name +"_test", yy)

    

 
    
    
    
    
            
