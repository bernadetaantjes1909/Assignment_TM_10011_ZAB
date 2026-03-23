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
import importlib
import Classifiers
import Feature_selection_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import LabelEncoder


#%% load functions from other documents

importlib.reload(Classifiers)
importlib.reload(Feature_selection_data)
from load_data import load_data
from Preprocessing_data import preprocessing_data
from Feature_selection_data import deleting_zero_variance, feature_selection_PCA, feature_selection_RFE
from Classifiers import random_forest_classifier, logistic_regression_classifier
from Feature_selection_data import remove_correlated_features

# vul in, soort feature selection(fs), en soort classifier(clf), op de volgende manier:
#y_pred_train_fs_clf, y_pred_test_fs_clf, train_data_elimination, test_data_elimination, classification_train, classification_test = clf_function(load_data, preprocessing_data, deleting_zero_variance, fs_function)
best_param_PCA_rf, y_pred_train_PCA_rf, y_pred_test_PCA_rf, train_data_elimination, test_data_elimination, classification_train, classification_test = random_forest_classifier(load_data, preprocessing_data, deleting_zero_variance, feature_selection_PCA)
best_param_RFE_rf, y_pred_train_RFE_rf, y_pred_test_RFE_rf, train_data_elimination, test_data_elimination, classification_train, classification_test = random_forest_classifier(load_data, preprocessing_data, deleting_zero_variance, feature_selection_RFE)
y_pred_train_lr, y_pred_test_lr, train_data_filtered, test_data_filtered, classification_train, classification_test = logistic_regression_classifier(load_data, preprocessing_data, deleting_zero_variance)

#%%
# 2. Voer de functie uit
df_clean = remove_redundant_features(df, 0.995)
# 3. Resultaat bekijken
print("\nNieuwe vorm van de dataset:", df_clean.shape)
print(df_clean.head())

classification_train= np.where(classification_train == 0, "malignant", "benign")
classification_test = np.where(classification_test == 0, "malignant", "benign")
# bepalen aantal misclassified
t_train_PCA_rf = ("Misclassified train: %d / %d" % ((classification_train != y_pred_train_PCA_rf).sum(), train_data_elimination.shape[0]))
t_test_PCA_rf = ("Misclassified test: %d / %d" % ((classification_test != y_pred_test_PCA_rf).sum(), test_data_elimination.shape[0]))
t_train_RFE_rf = ("Misclassified train: %d / %d" % ((classification_train != y_pred_train_RFE_rf).sum(), train_data_elimination.shape[0]))
t_test_RFE_rf = ("Misclassified test: %d / %d" % ((classification_test != y_pred_test_RFE_rf).sum(), test_data_elimination.shape[0]))
t_train_lr = ("Misclassified train: %d / %d" % ((classification_train != y_pred_train_lr).sum(), train_data_elimination.shape[0]))
t_test_lr = ("Misclassified test: %d / %d" % ((classification_test != y_pred_test_lr).sum(), test_data_elimination.shape[0]))

print(t_train_PCA_rf)
print(t_test_PCA_rf)
print(best_param_PCA_rf)
print(t_train_RFE_rf)
print(t_test_RFE_rf)
print(best_param_RFE_rf)
print(t_train_lr)
print(t_test_lr)




# %%
