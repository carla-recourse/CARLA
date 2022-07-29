from import_essentials import *
from utils import *
from torch import nn
from sklearn.preprocessing import StandardScaler,MinMaxScaler, OneHotEncoder
import pathlib  
pl_logger = logging.getLogger('lightning')
main_path =  str(pathlib.Path().resolve()) + '/'

# Load configuration json file and prepare data  
class Load_dataset_base(nn.Module):
    def __init__(self, data_config,model_config_dict,return_data_loader=True,subsample=False):
        super().__init__()
        
        self.return_data_loader = return_data_loader
        self.subsample = subsample
        
        # read data
        self.name = data_config["name"]
        self.data = pd.read_csv(main_path + data_config['data_dir']).fillna('')
        self.continous_cols = data_config['continous_cols']
        self.discret_cols = data_config['discret_cols']
        self.class_size = data_config["class_size"]
        self.feature_size = data_config["feature_size"]
        self.check_cols()
        
        
        # load training and model configs from model_config_dict
        for param in model_config_dict.keys() :
            setattr(self,param,model_config_dict[param])
        
        
          
         
    # Function to ohe labels     
    def one_hot(self,labels):
        targets = torch.zeros(labels.size(0), self.class_size)
        for i, label in enumerate(labels):
            targets[i, label] = 1
        return targets    
    
    
        
    def check_cols(self):
        self.data = self.data.astype({col: np.float for col in self.continous_cols})

    

    # Inverse_transform to obtain original dataset (inverse ohe + inverse normalize)
    def inverse_transform(self, x, return_tensor=True):
        """x should be a transformed tensor"""
        cat_idx = len(self.continous_cols)
        # inverse transform
        x_cont_inv = self.normalizer.inverse_transform(x[:, :cat_idx].cpu())
        x_cat_inv = self.encoder.inverse_transform(x[:, cat_idx:].cpu()) if self.discret_cols else np.array([[] for _ in range(len(x))])
        # First colomns are continuous variables and next are categorical variables 
        x = np.concatenate((x_cont_inv, x_cat_inv), axis=1)
        return torch.from_numpy(x).float() if return_tensor else x

    

    # Preprocessing and splitting 
    def prepare_data(self):
        def split_x_and_y(data):
            X = data[data.columns[:-1]]
            y = data[data.columns[-1]]
            return X, y
        
        
        
        X, y = split_x_and_y(self.data)
        # preprocessing 
        self.normalizer = MinMaxScaler()
        self.encoder = OneHotEncoder(sparse=False)
        X_cont = self.normalizer.fit_transform(X[self.continous_cols]) if self.continous_cols else np.array([[] for _ in range(len(X))])
        
        X_cat = self.encoder.fit_transform(X[self.discret_cols]) if self.discret_cols else np.array([[] for _ in range(len(X))])
        X = np.concatenate((X_cont, X_cat), axis=1)
        
        pl_logger.info(f"x_cont: {X_cont.shape}, x_cat: {X_cat.shape}")

        cat_arrays = self.encoder.categories_ if self.discret_cols else []
        pl_logger.info(X.shape)
        # Number of continious variables 
        cont_shape = X_cont.shape[1]

        # prepare train & test
        train_X, test_X, train_y, test_y = train_test_split(X, y.to_numpy(), shuffle=False)
        
   
        self.train_dataset = NumpyDataset(train_X, train_y)
        self.val_dataset = NumpyDataset(test_X, test_y)
        if self.subsample : 
            sample = np.loadtxt(main_path + "post_hoc_counterfactuals/random_sample_" + self.name).astype(int)
            self.test_sample_dataset =  NumpyDataset(test_X[sample], test_y[sample])
        
        
        if self.return_data_loader : 
            #Dataloaders 
            train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size,pin_memory=True, shuffle=True, num_workers=0)
            val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size,pin_memory=True, shuffle=True, num_workers=0)
            test_loader = DataLoader(self.val_dataset, batch_size=self.batch_size,pin_memory=True, shuffle=False, num_workers=0)
            if self.subsample : 
                test_sample_loader =  DataLoader(self.test_sample_dataset, batch_size=self.batch_size,pin_memory=True, shuffle=False, num_workers=0)
                loaders = { 'train' : train_loader, "test" : test_loader, "val" : val_loader , "sample" : test_sample_loader }
            else : 
                loaders = { 'train' : train_loader, "test" : test_loader, "val" : val_loader }
        
            return(loaders,cat_arrays,cont_shape)
        else : 
            return(cat_arrays,cont_shape)


def load_data_dict(name) :
    return(json.load(open(main_path + "data_configs/" + name +".json")))



# Load configuration and prepare data  
class Load_dataset_carla(nn.Module):
    def __init__(self, data_catalog,model_config_dict,return_data_loader=True,subsample=False):
        super().__init__()
        
        self.return_data_loader = return_data_loader
        # read data
        self.name = data_catalog.name
        self.continous_cols = data_catalog.continuous
        self.discret_cols = data_catalog.categorical
        self.class_size = 2 
        self.feature_size = data_catalog.df_train.shape[1] - 1
        self.data_catalog = data_catalog
        # load training and model configs from model_config_dict
        for param in model_config_dict.keys() :
            setattr(self,param,model_config_dict[param])
        
        
          
        
    # Function to ohe labels     
    def one_hot(self,labels):
        targets = torch.zeros(labels.size(0), self.class_size)
        for i, label in enumerate(labels):
            targets[i, label] = 1
        return targets    
        
        
    
    

     # Inverse_transform to obtain original dataset (inverse ohe + inverse normalize)
    def inverse_transform(self, x, return_tensor=True):
        """x should be a transformed tensor"""
        # Tensor to data_frame 
        x_pd = pd.DataFrame(data = x.numpy(),columns=list(self.train.drop(columns = [self.target])))
        x_numpy = self.data_catalog.inverse_transform(x_pd).to_numpy()
        return torch.from_numpy(x).float() if return_tensor else x_numpy


    # Preprocessing and splitting 
    def prepare_data(self):
        
 
        # target 
        self.target= self.data_catalog.target
        # Train and test 
        self.train = self.data_catalog.df_train 
        self.test = self.data_catalog.df_test
        self.val = self.data_catalog.df_val
        # preprocessing 
        self.normalizer = self.data_catalog.scaler
        self.encoder = self.data_catalog.encoder
        
        # Categories is second column for each categorical feature (drop first)
        cat_arrays = self.encoder.categories_ if self.discret_cols else []
        def convert_drop_first(cat_arrays):
            cat_arrays_drop_first = [] 
            for e in cat_arrays : 
                cat_arrays_drop_first.append(np.array([e[1]]))
            return(cat_arrays_drop_first)
        if self.encoder == "if_binary":
            cat_arrays = convert_drop_first(cat_arrays)
        #cat_arrays = self.data_catalog.encoder.get_feature_names(self.data_catalog.categorical) if self.discret_cols else []
        #print("ENCODER CATEGORIES: ",self.encoder.categories_)
        # Number of continious variables 
        cont_shape = len(self.data_catalog.continuous)


        train_X = self.train.drop(columns = [self.target]).to_numpy()
        train_y = self.train[self.target].to_numpy()
        
        val_X = self.val.drop(columns = [self.target]).to_numpy()
        val_y = self.val[self.target].to_numpy()

        test_X = self.test.drop(columns = [self.target]).to_numpy()
        test_y = self.test[self.target].to_numpy()
        

       
        
        self.train_dataset = NumpyDataset(train_X, train_y)
        self.test_dataset = NumpyDataset(test_X, test_y)
        self.val_dataset = NumpyDataset(val_X, val_y)
        
        
        if self.return_data_loader : 
            #Dataloaders 
            train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size,pin_memory=True, shuffle=True, num_workers=0)
            val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size,pin_memory=True, shuffle=True, num_workers=0)
            test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size,pin_memory=True, shuffle=False, num_workers=0)
            loaders = { 'train' : train_loader, "test" : test_loader, "val" : val_loader }
        
            return(loaders,cat_arrays,cont_shape)
        else : 
            return(cat_arrays,cont_shape)