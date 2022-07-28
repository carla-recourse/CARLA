#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from import_essentials import * 
from utils import *
import torch 
from load_data import Load_dataset
from torch import nn
from torch.nn import functional as F

 
# Classification model 
class Classif_model(Load_dataset) :
    def __init__(self, config,cat_arrays,cont_shape):
        super().__init__(config)
        
        self.cat_arrays = cat_arrays
        self.cont_shape = cont_shape
       
        
        # Encoding
        self.se1  = nn.Linear(self.feature_size, self.latent_size_share)
        
        # Classification 
        
        self.fcl1 = nn.Linear(self.latent_size_share,self.latent_size_share)
        self.fcl2 = nn.Linear(self.latent_size_share,self.class_size -1)
        
        # Activation functions
        self.elu = nn.ELU()
        self.sigmoid = nn.Sigmoid()
        
        
    # Encoding     
    def encode(self,x) :
        z = self.elu(self.se1(x))
        return(z)
    
    # Classification layers (after shared encoding)
    def classif(self,z) :
        c1 = self.elu(self.fcl1(z))
        c2 = self.fcl2(c1)
        return self.sigmoid(c2)
    
    # Forward in train phase
    def forward(self, x):
        # Separate encoding for classification
        z = self.encode(x)
        # Output of classification layers
        output_class = self.classif(z)
        
        return(output_class)
    
# CVAE model     
class CVAE_model(Load_dataset) : 
        def __init__(self, config,cat_arrays,cont_shape):
            
            super().__init__(config)
            self.cat_arrays = cat_arrays
            self.cont_shape = cont_shape
            
            # Encoding
            self.se1  = nn.Linear(self.feature_size, self.latent_size_share)
            
            # C-VAE encoding 
            self.e1  = nn.Linear(self.latent_size_share + self.class_size-1, self.mid_reduce_size)
            self.e2 = nn.Linear(self.mid_reduce_size, self.latent_size)
            self.e3 = nn.Linear(self.mid_reduce_size, self.latent_size)
            
            # C-VAE Decoding
            self.fd1 = nn.Linear(self.latent_size + self.class_size-1, self.mid_reduce_size)
            self.fd2 = nn.Linear(self.mid_reduce_size, self.latent_size_share)
            self.fd3 = nn.Linear(self.latent_size_share, self.feature_size)
            
            # Activation functions
            self.elu = nn.ELU()
            self.sigmoid = nn.Sigmoid()
            
        # Softmax for counterfactual output + sigmoid on numerical variables (function in utils.py)
        def cat_normalize(self, c, hard=False):
            # categorical feature starting index
            cat_idx = len(self.continous_cols)
            return cat_normalize(c, self.cat_arrays, cat_idx,self.cont_shape,hard=hard)
         
            
        # Encoding     
        def encode_first(self,x) :
            z = self.elu(self.se1(x))
            return(z)
        
        
        # C-VAE encoding 
        def encode(self, z,c):  
            inputs = torch.cat([z, c], 1) 
            h1 = self.elu(self.e1(inputs))
            z_mu = self.e2(h1)
            z_var = self.e3(h1)
            return z_mu, z_var
        
        # Reparametrization trick
        def reparameterize(self, mu, logvar): 
            torch.manual_seed(0)
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return mu + eps*std
        
        # C-VAE decoding 
        def decode(self, z_prime, c): # P(x|z, c)
            '''
            z: (bs, latent_size)
            c: (bs, class_size)
            '''
            inputs = torch.cat([z_prime, c], 1) # (bs, latent_size+class_size)
            h1 = self.elu(self.fd1(inputs))
            h2 = self.elu(self.fd2(h1)) 
            h3 = self.fd3(h2)
            
            return h3
             
        # Forward in train phase
        def forward(self, x,output_class):
            # Encoding 
            z = self.encode_first(x)
            
            # C-VAE encoding  
            mu, logvar = self.encode(z,output_class)
            z_prime = self.reparameterize(mu, logvar)
            
            # Decoded output 
            c = self.decode(z_prime, output_class)
            
            # Softmax activation for ohe variables 
            c = self.cat_normalize(c, hard=False)
            
            # Return Decoded output + output class
            return c, mu, logvar
        
        
        def forward_counterfactuals(self,x,c_pred,c_pred_x) :
           
            # Encoding 
            z = self.encode_first(x)
            
            # C-VAE encoding  
            mu, logvar = self.encode(z,c_pred_x)
            z_prime = self.reparameterize(mu, logvar)
            
            
            # Decoded output 
            c = self.decode(z_prime, c_pred)
            
            # 0he format for c  
            c = self.cat_normalize(c, hard=True)
            
            
            return c, z_prime
        
        
        
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            