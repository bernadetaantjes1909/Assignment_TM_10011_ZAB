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
from sklearn.ensemble import RandomForestClassifier

def preprocessing_data(data):
#split data
    data_train, data_test, classification_train, classification_test = model_selection.train_test_split(data.iloc[:,2:], data['label'])

#%%
# scaling data
    scaler = preprocessing.RobustScaler()
    scaler.fit(data_train)
    train_data_scaled = scaler.transform(data_train)
    test_data_scaled = scaler.transform(data_test)
    return train_data_scaled, test_data_scaled, classification_train, classification_test
