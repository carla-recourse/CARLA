    # copy essential imports from fastai2
# https://github.com/fastai/fastai/blob/master/fastai/imports.py

import matplotlib.pyplot as plt,numpy as np,pandas as pd,scipy
from typing import Union,Optional,Dict
import io,operator,sys,os,re,mimetypes,csv,itertools,json,shutil,glob,pickle,tarfile,collections
import hashlib,itertools,types,inspect,functools,random,time,math,bz2,typing,numbers,string
import multiprocessing,threading,urllib,tempfile,concurrent.futures,matplotlib,warnings,zipfile

# import pytorch-related
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataset import TensorDataset
from torch.utils.data import DataLoader

# import sklearn
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate

# misc.
from pprint import pprint
import logging
from joblib import Parallel, delayed
from tqdm import tqdm
from pathlib import Path

# pytorch-lightening
#import pytorch_lightning as pl
#from pytorch_lightning.callbacks import ModelCheckpoint
#from pytorch_lightning import loggers as pl_loggers

