#%%
# import packages
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

# Variance and deleting features
# This function is used in all feature selection techniques below:
def deleting_zero_variance(load_data,preprocessing_data):
    train_data_scaled,test_data_scaled, classification_train, classification_test = preprocessing_data(load_data)
    Variance = np.var(train_data_scaled, axis=0)
    zero_indices = np.where(Variance < 0.01)[0]
    train_data_filtered = np.delete(train_data_scaled, zero_indices, axis=1)
    test_data_filtered = np.delete(test_data_scaled, zero_indices, axis=1)
    return train_data_filtered, test_data_filtered, classification_train, classification_test
#%%


def remove_correlated_features(load_data, preprocessing_data, threshold=0.995):
    train_data_scaled, test_data_scaled, classification_train, classification_test = preprocessing_data(load_data)
    corr_matrix = np.abs(np.corrcoef(train_data_scaled, rowvar=False))
    upper_tri = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    to_drop = [i for i in range(corr_matrix.shape[1]) if any(corr_matrix[i, j] > threshold for j in range(i+1, corr_matrix.shape[1]))]
    train_data_filtered = np.delete(train_data_scaled, to_drop, axis=1)
    test_data_filtered = np.delete(test_data_scaled, to_drop, axis=1)
    print(f"Correlatie filter: {len(to_drop)} kolommen verwijderd.")
    return train_data_filtered, test_data_filtered, classification_train, classification_test
#%%

# feature selection using recursive feature elimination
def feature_selection_RFE (load_data, preprocessing_data, deleting_zero_variance):
    train_data_filtered, test_data_filtered, classification_train, classification_test  = deleting_zero_variance(load_data,preprocessing_data)
   # RFE object:
    svc = svm.SVC(kernel="linear")
    rfecv = feature_selection.RFECV(
        estimator=svc, step=1,
        cv=model_selection.StratifiedKFold(4),
        scoring='accuracy') # can be done because both classes occur equally in the dataset
    rfecv.fit(train_data_filtered,classification_train)
    optimal = rfecv.n_features_
    train_data_elimination = train_data_filtered[:, rfecv.support_]
    test_data_elimination  = test_data_filtered[:, rfecv.support_]

    #print(f"optimal amount of features is {optimal}")
    return train_data_elimination, test_data_elimination, classification_train, classification_test

#feature selection using principle component analysis
def feature_selection_PCA (load_data, preprocessing_data, deleting_zero_variance):
    train_data_filtered, test_data_filtered, classification_train, classification_test = deleting_zero_variance(load_data ,preprocessing_data)

    pca = decomposition.PCA(n_components=20)
    pca.fit(train_data_filtered)
    train_data_elimination = pca.transform(train_data_filtered)
    test_data_elimination = pca.transform(test_data_filtered)

    return train_data_elimination, test_data_elimination, classification_train, classification_test

# feature selection with LASSO?










# # feature selection - importance
# forest = RandomForestClassifier(n_estimators=100)

# forest.fit(train_data_scaled, classification_train)
# importances = forest.feature_importances_
# std = np.std([tree.feature_importances_ for tree in forest.estimators_],
#              axis=0)
# indices = np.argsort(importances)[::-1]

# # Print the feature ranking
# print("Feature ranking:")

# for f in range(train_data_scaled.shape[1]):
#     print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# #print(indices[0:optimal])

# # Plot the feature importances of the forest
# plt.figure()
# plt.title("Feature importances")
# plt.bar(range(train_data_scaled.shape[1]), importances[indices],
#        color="r", yerr=std[indices], align="center")
# plt.xticks(range(train_data_scaled.shape[1]), indices)
# plt.xlim([-1, train_data_scaled.shape[1]])
# plt.ylim([0, 0.09])
# plt.show()
# %%

# PCA doen voor feature extraction
# relu voor neural network
# laatste signoid of binary
