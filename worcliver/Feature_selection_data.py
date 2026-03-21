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
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel

# Variance and deleting features
# This function is used in all feature selection techniques below:
def deleting_zero_variance(load_data,preprocessing_data):
    train_data_scaled,test_data_scaled, classification_train, classification_test = preprocessing_data(load_data)
    Variance = np.var(train_data_scaled, axis=0)
    zero_indices = np.where(Variance < 0.01)[0]
    train_data_filtered = np.delete(train_data_scaled, zero_indices, axis=1)
    test_data_filtered = np.delete(test_data_scaled, zero_indices, axis=1)
    return train_data_filtered, test_data_filtered, classification_train, classification_test


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

from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel

from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import LabelEncoder

def feature_selection_LASSO(load_data, preprocessing_data, deleting_zero_variance):
    train_data_filtered, test_data_filtered, classification_train, classification_test = \
        deleting_zero_variance(load_data, preprocessing_data)

    le = LabelEncoder()
    classification_train_num = le.fit_transform(classification_train)

    lasso = Lasso(alpha=1e-4, random_state=42)
    selector = SelectFromModel(estimator=lasso, threshold='median')

    selector.fit(train_data_filtered, classification_train_num)

    train_data_elimination = selector.transform(train_data_filtered)
    test_data_elimination  = selector.transform(test_data_filtered)

    return train_data_elimination, test_data_elimination, classification_train, classification_test

# feature selection univariate 
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import cross_val_score
import numpy as np

def feature_selection_univariate(load_data, preprocessing_data, deleting_zero_variance):
    train_data_filtered, test_data_filtered, classification_train, classification_test = \
        deleting_zero_variance(load_data, preprocessing_data)

    best_score = 0
    best_k = 0
    best_selector = None

    for k in range(5, train_data_filtered.shape[1], 5):
        selector = SelectKBest(f_classif, k=k)
        X_new = selector.fit_transform(train_data_filtered, classification_train)

        score = np.mean(cross_val_score(
            svm.SVC(kernel="linear"),
            X_new,
            classification_train,
            cv=4
        ))

        if score > best_score:
            best_score = score
            best_k = k
            best_selector = selector

    print(f"Best k: {best_k}")

    train_data_elimination = best_selector.transform(train_data_filtered)
    test_data_elimination  = best_selector.transform(test_data_filtered)

    return train_data_elimination, test_data_elimination, classification_train, classification_test




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
