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

# 1. Data inladen
# Let op: de haakjes () zijn nodig om de functie uit te voeren
df = load_data() 

def remove_correlated_features(df_input, threshold=0.995):
    df_numeric = df_input.select_dtypes(include=[np.number])    # Dit zorgt ervoor dat de eerste 2 tekstkolommen automatisch worden overgeslagen
    corr_matrix = df_numeric.corr().abs()    # Stap B: Correlatiematrix berekenen (absolute waarden)
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]    # Stap D: Identificeer welke kolommen boven de 0.995 overlap zitten
    # Rapportage
    print(f"Totaal aantal kolommen: {df_input.shape[1]}")
    print(f"Aantal numerieke kolommen geanalyseerd: {df_numeric.shape[1]}")
    print(f"Aantal kolommen te verwijderen: {len(to_drop)}")
    if len(to_drop) > 0:
        print(f"De volgende kolommen worden verwijderd: {to_drop}")
    df_dropped = df_input.drop(columns=to_drop)    # Stap E: Verwijder de kolommen uit de originele dataframe
    return df_dropped, remove_correlated_features

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
