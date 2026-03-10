#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets as ds
from sklearn import decomposition

from sklearn import model_selection
from sklearn import metrics
from sklearn import feature_selection
from sklearn import preprocessing
from sklearn import neighbors
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier

#%% load functions from other files
from load_data import load_data
from Preprocessing_data import preprocessing_data
from Feature_selection_data import feature_selection_data

#%% load functions
data = load_data()
#%%
train_data_scaled, test_data_scaled, classification_train, classification_test= preprocessing_data(data)
#%%
X_train_pca,X_test_pca = feature_selection_data (train_data_scaled, test_data_scaled,classification_train)
# %%
print(X_train_pca.shape)