#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  9 11:39:51 2022

@author: nwgl2572
"""
import torch 
import numpy as np
from complementary_metrics import *

 
# Compute metrics 
def compute_metrics(data,labels,result,not_on_batch=False,from_numpy=False) : 
    
    counterfactuals = result["cf"]
    label_examples = result["y_x"].flatten()
    predicted_examples_proba = result["proba_x"] 
    predicted_counterfactuals_proba = result["proba_c"]
    predicted_counterfactuals_classes = result["y_c"]
    
    # Convert to torch tensors if numpy array as input 
    if from_numpy : 
        data = torch.from_numpy(data) 
        counterfactuals = torch.from_numpy(counterfactuals) 
        predicted_examples_proba = torch.from_numpy(predicted_examples_proba)
        predicted_counterfactuals_proba = torch.from_numpy(predicted_counterfactuals_proba)
    
    labels_counterfactuals = 1-label_examples   
    
    # Validity metric
    validity = (labels_counterfactuals ==predicted_counterfactuals_classes).sum()
    
    
    # Sparsity metric (number of changes between example to explain and counterfactual)
    Sparsity = torch.sum(torch.abs(data-counterfactuals) > 1e-6,axis=1).float()
    
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
    
    # if counterfactuals are not computed on batch 
    if not_on_batch : 
        # validity is return as the total percentage of valid counterfactuals 
        validity = validity /labels_counterfactuals.shape[0]
        # diversity metric is mean pairwise distances for counterfactuals 
        diversity = torch.from_numpy(np.array([pdist(counterfactuals).mean()]))
        return(Sparsity,Gain,Proximity,validity,diversity)
    
    
    
     
    return(Sparsity,Gain,Proximity,validity) 


def compute_others_metrics(results,name,from_numpy=False) : 
    
    counterfactuals = results["cf"]
    x_train = results["x_train"]
    y_pred_counterfactuals = results["y_c"]
    y_pred_train = results["y_x_train"]
    
    if not from_numpy : 
        counterfactuals = counterfactuals.numpy() 
        y_pred_train = y_pred_train.cpu().numpy()
        
        
    # Laugel proximity 
    Proximity_laugel =  Compute_prox(x_train, counterfactuals,y_pred_counterfactuals,y_pred_train,name)
    
    return(Proximity_laugel)
 










