
# Een beginnetje aan classifiers maar het klopt nog niet hoor 
# Dus pas vooral alles aan wat je wilt 
# Voor nu heb ik hier de andere functies ingeladen
# Maar later moet dit een functie worden,
# die we in main inladen.
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


#%%
rf = RandomForestClassifier()
rf.fit(X_train_pca, classification_train)
y_pred_train = rf.predict(X_train_pca)
y_pred_test = rf.predict(X_test_pca)

t_train = ("Misclassified train: %d / %d" % ((classification_train != y_pred_train).sum(), X_train_pca.shape[0]))
t_test = ("Misclassified test: %d / %d" % ((classification_test != y_pred_test).sum(), X_test_pca.shape[0]))
print (t_train)
print(t_test)
# %%
