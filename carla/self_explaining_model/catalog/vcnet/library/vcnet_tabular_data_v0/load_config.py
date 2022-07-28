#!/usr/bin/env python3
# -*- coding: utf-8 -*-
 
from import_essentials import *
from utils import *
from torch import nn
from sklearn.preprocessing import StandardScaler,MinMaxScaler, OneHotEncoder
import pathlib 
pl_logger = logging.getLogger('lightning')
main_path =  str(pathlib.Path().resolve()) + '/'
main_path = main_path +  "/vcnet_tabular_data_v0/"
# Load configuration json file 
class Load_config(nn.Module):
    def __init__(self, model_config: Dict):
        super().__init__()
         
        
        # set training configs
        self.lr = model_config['lr']
        self.batch_size = model_config['batch_size']
        self.lambda_1 = model_config['lambda_1'] if 'lambda_1' in model_config.keys() else 1
        self.lambda_2 = model_config['lambda_2'] if 'lambda_2' in model_config.keys() else 1
        self.lambda_3 = model_config['lambda_3'] if 'lambda_3' in model_config.keys() else 1
        self.threshold = model_config['threshold'] if 'threshold' in model_config.keys() else 0.5
        self.smooth_y = model_config['smooth_y'] if 'smooth_y' in model_config.keys() else True
        self.epochs = model_config["epochs"]

       
       
        # set model configs 
        self.latent_size = model_config["latent_size"]
        self.latent_size_share = model_config["latent_size_share"] 
        self.mid_reduce_size =  model_config["mid_reduce_size"] 
        
        
def load_config_dict(name) : 
    return(json.load(open(main_path + "configs/" + name + ".json")))
        
 
