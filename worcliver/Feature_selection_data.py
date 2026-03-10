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

from load_data import load_data
from worcliver.Preprocessing_data import preprocessing_data

data = load_data()
train_data_scaled, test_data_scaled, classification_train, classification_test= preprocessing_data(data)



#%%
def feature_selection_data (train_data_scaled, test_data_scaled,classification_train):
# Variance and deleting features
    Variance = np.var(train_data_scaled, axis=0)
    zero_indices = np.where(Variance < 0.01)[0]
    train_data_filtered = np.delete(train_data_scaled, zero_indices, axis=1)
    test_data_filtered = np.delete(test_data_scaled, zero_indices, axis=1)

#Recursive Feature Elimination: 
# RFE object:
    # svc = svm.SVC(kernel="linear")
    # rfecv = feature_selection.RFECV(
    #     estimator=svc, step=1,
    #     cv=model_selection.StratifiedKFold(4),
    #     scoring='accuracy')
    # rfecv.fit(train_data_filtered,classification_train)
    # optimal = rfecv.n_features_
    # train_data_elimination = train_data_scaled[:, rfecv.support_]
    # test_data_elimination  = test_data_scaled[:, rfecv.support_]

    # print(f"optimal amount of features is {optimal}")

#principle component analysis
    from sklearn import decomposition

    pca = decomposition.PCA(n_components=20)
    pca.fit(train_data_filtered)
    X_train_pca = pca.transform(train_data_filtered)
    X_test_pca = pca.transform(test_data_filtered)

    return X_train_pca,X_test_pca


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
