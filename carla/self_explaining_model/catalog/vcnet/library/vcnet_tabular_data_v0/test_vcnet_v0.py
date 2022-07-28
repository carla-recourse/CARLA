#!/usr/bin/env python3
# -*- coding: utf-8 -*- 
 
from import_essentials import *
from utils import *
from load_data import Load_dataset,load_data_dict
from load_config import Load_config,load_config_dict
from join_training_network import CVAE_join,Predictor
from train_network import Train_CVAE 
import torch 
from sklearn.metrics import pairwise_distances 
import pathlib 
import seaborn as sns 
from metrics import *
from plot_distributions import numpy_to_dataframe,plot_distributions
main_path =  str(pathlib.Path().resolve()) + '/'
 
# All availaible datasets 
dataset_names = ['adult', 'student', 'home', 'student_performance', 'titanic', 'breast_cancer', 'blobs', 'moons', 'circles']



name = "blobs"
 
# Load the corresponding model config
model_config_dict = load_config_dict(name)
model_config = Load_config(model_config_dict)

# Load the dataset 
dataset_config_dict = load_data_dict(name)
dataset = Load_dataset(dataset_config_dict,model_config_dict,subsample=False)

# Prepare dataset and return dataloaders + ohe index 
loaders,cat_arrays,cont_shape = dataset.prepare_data()


# Prepare training 
training = Train_CVAE(dataset_config_dict,model_config_dict,cat_arrays,cont_shape,loaders,dataset,ablation="remove_enc",condition="change_dec_only",cuda_name="cpu",shared_layers=False)


training.train_and_valid_cvae()

#training.load_weights(dataset.name)

# Save the prediction model for comparision with post_hoc optimization methods 
#training.save_post_hoc_prediction_model()


# Save results for 2D plot  
#training.save_for_plot_toy()
 
#training.save_contourf_for_plot()

## Compute results 

# Select all the test data 
X,y = dataset.val_dataset[:]

# Select subsample 
#X,y =  dataset.test_sample_dataset[:] 

# Compute counterfactuals 
results = training.compute_counterfactuals(X.to(training.cuda_device), y.to(training.cuda_device),laugel_metric=True)
predicted_example_class = results["y_x"].cpu().numpy()
predicted_counterfactual_class = results["y_c"].cpu().numpy()

# Compute metrics 
Gain,Proximity,validity = compute_metrics(X,y,results,not_on_batch=True,from_numpy=False)

mean_gain = float(torch.mean(Gain))
mean_prox = float(torch.mean(Proximity))
std_gain = float(torch.std(Gain))
std_prox = float(torch.std(Proximity))
print("Validity on the test set :",float(validity))
print('Proximity on the test set {} +/- {}:'.format(round(mean_prox,3),round(std_prox,3)))
print("Gain on the test set {} +/- {}: ".format(round(mean_gain,3),round(std_gain,3)))

# Compute proximity score metric 
Proximity_laugel = compute_others_metrics(results,name,from_numpy=True)
mean_prox_laugel, std_prox_laugel = np.mean(Proximity_laugel),np.std(Proximity_laugel)
print("Proximity_laugel metric for {} dataset is {} +/- {}".format(name,str(round(mean_prox_laugel,3)),str(round(std_prox_laugel,3))))



   
#original_examples,original_counterfactuals = numpy_to_dataframe(X,results["cf"],dataset)



'''
#Plot distributions for each target values 
original_examples_target,original_counterfactuals_target = original_examples.copy(),original_counterfactuals.copy()
original_examples_target["Target"] = predicted_example_class
original_counterfactuals_target["Target"] = predicted_counterfactual_class 

plt.figure()
plot_distributions(name,original_examples_target,counterfactual=False,hue=True)
plt.figure()
plot_distributions(name,original_counterfactuals_target,counterfactual=True,hue=True)
'''
 

'''# Magnitude of change for numerical variables 
numerical_diff = (original_examples.select_dtypes(include=np.number) - original_counterfactuals.select_dtypes(include=np.number)).abs()
for feature in list(numerical_diff) : 
    plt.figure()
    sns.histplot(data=numerical_diff,y=feature)
    plt.title("Magnitude of change for {} features".format(feature))

# Plot the confusion matrix for each categorical variable attribute
fig, axs = plt.subplots(4, 4, figsize=(40, 30),constrained_layout=True)
features = list(original_examples)
for i in range(axs.shape[0]) :
    for j in range(axs.shape[1]) : 
        if features not in list(numerical_diff) : 
        cross_tab = pd.crosstab(original_examples[features],original_counterfactuals[features]) 
        cross_tab = cross_tab / cross_tab.sum().sum()
        sns.heatmap(cross_tab / cross_tab.sum().sum().T,annot=True,fmt='.2%')
        
'''    
'''        
        '''
        
        
        

