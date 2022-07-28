#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from import_essentials import *
from utils import *
from load_data import Load_dataset
from join_training_network import CVAE_join
from train_network import Train_CVAE 
import torch 
from sklearn.metrics import pairwise_distances 
import sys
import pathlib

main_path =  str(pathlib.Path().resolve()) + '/'
   
# Load configuration file 
configs = {"adult" : json.load(open(main_path + "configs/adult.json")), "student" : json.load(open(main_path +"configs/student.json")),
          "home" : json.load(open(main_path +"configs/home.json")),"student_performance" : json.load(open(main_path +"configs/extra/student_performance.json")),
          "titanic" :  json.load(open(main_path +"configs/extra/titanic.json")), "breast_cancer" : json.load(open(main_path +"configs/extra/breast_cancer.json")) }

Name = ["adult","student","home"]
GPU = ["cuda:0","cuda:1","cuda:2"]


def run_exp(name,cuda_name) : 

    config = configs[name]
    dataset = Load_dataset(config,main_path)
 

    # Prepare dataset and return dataloaders + ohe index 
    loaders,cat_arrays,cont_shape = dataset.prepare_data()

    # Prepare training 
    training = Train_CVAE(config,main_path,cat_arrays,cont_shape,loaders,dataset,ablation=None,cuda_name=cuda_name)
 
    training.run_optuna()
 
run_exp(sys.argv[1],sys.argv[2]) 
     