#!/usr/bin/env python3
# -*- coding: utf-8 -*-
 
from import_essentials import *
from utils import *
from load_data import Load_dataset
from post_hoc_training import Train_CVAE_post_hoc 
import torch 
from sklearn.metrics import pairwise_distances 
import pathlib

main_path =  str(pathlib.Path().resolve()) + '/'
  
# Load configuration file 
configs = {"adult" : json.load(open(main_path + "configs/adult.json")), "student" : json.load(open(main_path +"configs/student.json")),
          "home" : json.load(open(main_path +"configs/home.json")),"student_performance" : json.load(open(main_path +"configs/extra/student_performance.json")),
          "titanic" :  json.load(open(main_path +"configs/extra/titanic.json")), "breast_cancer" : json.load(open(main_path +"configs/extra/breast_cancer.json")) }

# Load the dataset 
name = "adult"
config = configs[name]
dataset = Load_dataset(config)
 

# Prepare dataset and return dataloaders + ohe index 
loaders,cat_arrays,cont_shape = dataset.prepare_data()

# Hyperparameters for training 
epochs_cvae = 10 
lr_cvae = 1e-4
lambda_1 = 1
lambda_3 = 0.001
epochs_classif = 50
lr_classif = 1e-3

 
training = Train_CVAE_post_hoc(config,cat_arrays,cont_shape,loaders,dataset,epochs_cvae,lr_cvae,epochs_classif,lr_classif,lambda_3,lambda_1,"cpu")
# Train the classification model 
#training.train_and_valid_classif()
#Train the cvae as a post-hoc model
#training.train_and_valid_cvae()

training.load_weights(dataset.name)
# Run counterfactuals metrics 
training.test()

 
# Counterfactuals results 
X,y = dataset.val_dataset[:]
results = training.compute_counterfactuals(X, y)
# Compute laugel proximity metric 
Counterfactuals = results["cf"].numpy()
Proximity_laugel = training.compute_others_metrics(Counterfactuals,results["y_c"],name)
mean_prox_laugel, std_prox_laugel = np.mean(Proximity_laugel),np.std(Proximity_laugel)
print("Proximity_laugel metric for {} dataset is {} +/- {}".format(name,str(round(mean_prox_laugel,3)),str(round(std_prox_laugel,3))))



# Run optimization hyperparameters 
#training.run_optuna()
 