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


#%% load functions from other documents
from load_data import load_data
from Preprocessing_data import preprocessing_data
from Feature_selection_data import deleting_zero_variance, feature_selection_PCA
from Classifiers import random_forest_classifier

# vul in, soort feature selection(fs), en soort classifier(clf), op de volgende manier:
#y_pred_train_fs_clf, y_pred_test_fs_clf, train_data_elimination, test_data_elimination, classification_train, classification_test = clf_function(load_data, preprocessing_data, deleting_zero_variance, fs_function)
best_param, y_pred_train_PCA_rf, y_pred_test_PCA_rf, train_data_elimination, test_data_elimination, classification_train, classification_test = random_forest_classifier(load_data, preprocessing_data, deleting_zero_variance, feature_selection_PCA)
#y_pred_train_RFE_rf, y_pred_test_RFE_rf, train_data_elimination, test_data_elimination, classification_train, classification_test = random_forest_classifier(load_data, preprocessing_data, deleting_zero_variance, feature_selection_PCA)

# bepalen aantal misclassified
t_train_PCA_rf = ("Misclassified train: %d / %d" % ((classification_train != y_pred_train_PCA_rf).sum(), train_data_elimination.shape[0]))
t_test_PCA_rf = ("Misclassified test: %d / %d" % ((classification_test != y_pred_test_PCA_rf).sum(), test_data_elimination.shape[0]))
#t_train_RFE_rf = ("Misclassified train: %d / %d" % ((classification_train != y_pred_train_PCA_rf).sum(), train_data_elimination.shape[0]))
#t_test_RFE_rf = ("Misclassified test: %d / %d" % ((classification_test != y_pred_test_PCA_rf).sum(), test_data_elimination.shape[0]))
# %%
print(t_train_PCA_rf)
print(t_test_PCA_rf)
print(best_param)
#print(t_train_RFE_rf)
#print(t_test_RFE_rf)


# %%
