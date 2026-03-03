#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets as ds
import seaborn
from sklearn import decomposition

from sklearn import model_selection
from sklearn import metrics
from sklearn import feature_selection
from sklearn import preprocessing
from sklearn import neighbors
from sklearn import svm

from load_data import load_data

data = load_data()

# %%
#split data
data_train, data_test, classification_train, classification_test = model_selection.train_test_split(data.iloc[:,2:], data['label'])

print(len(data_train))
print(len(data_test))
print(len(list(data_train.columns)))
print(classification_train.dtype)

#%%
# scaling data
def scaling_data(data_train, data_test):
    
    scaler = preprocessing.RobustScaler()
    scaler.fit(data_train)
    train_data_scaled = scaler.transform(data_train)
    test_data_scaled = scaler.transform(data_test)
    return train_data_scaled, test_data_scaled

#%%
#feature selection
train_data_scaled, test_data_scaled = scaling_data(data_train, data_test)
# RFE object:
svc = svm.SVC(kernel="linear")
rfecv = feature_selection.RFECV(
    estimator=svc, step=1,
    cv=model_selection.StratifiedKFold(4),
    scoring='roc_auc')
rfecv.fit(train_data_scaled,classification_train)

# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.cv_results_["mean_test_score"]) + 1), rfecv.cv_results_["mean_test_score"])
plt.show()

optimal = rfecv.n_features_
print(optimal)


# %%
