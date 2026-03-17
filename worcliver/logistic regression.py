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
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import LabelEncoder

#%%  volgende classifier logistic
def logistic_regression_classifier(load_data, preprocessing_data, deleting_zero_variance):
#%%
    train_data_filtered, test_data_filtered, classification_train, classification_test = deleting_zero_variance(
        load_data, preprocessing_data
    )
    print("Data is geladen, nu trainen...") # <--- Voeg dit toe

    # Model
    model = LogisticRegression(
        solver="saga",   # nodig voor L1
        l1_ratio=1,
        max_iter=5000
    )

    # Train
    model.fit(train_data_filtered, classification_train)

    # Predict
    y_pred_train = model.predict(train_data_filtered)
    y_pred_test = model.predict(test_data_filtered)

    train_acc = accuracy_score(classification_train, y_pred_train)
    test_acc = accuracy_score(classification_test, y_pred_test)

    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")

    return y_pred_train, y_pred_test, train_data_filtered, test_data_filtered, classification_train, classification_test


    # Return ook het model zelf!
   # return model, y_pred_train, y_pred_test, X_train, X_test, y_train, y_test
