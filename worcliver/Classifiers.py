
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
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV


#%%
#random_forest_classifier
def random_forest_classifier(load_data, preprocessing_data, deleting_zero_variance,feature_selection):
    train_data_elimination, test_data_elimination, classification_train, classification_test = feature_selection (load_data, preprocessing_data, deleting_zero_variance)

    rf = RandomForestClassifier(random_state=42)

    param_dist = {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 5, 10],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
        "max_features": ["sqrt", "log2"]
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
  
    search = RandomizedSearchCV(
        rf,
        param_distributions=param_dist,
        n_iter=15,
        scoring="accuracy",
        cv=cv,
        n_jobs=-1,
        random_state=42
    )
    
    search.fit(train_data_elimination, classification_train)
    
    tuned_model = search.best_estimator_

    y_pred_train = tuned_model.predict(train_data_elimination)
    y_pred_test = tuned_model.predict(test_data_elimination)
    best_param = search.best_params_

    return best_param, y_pred_train, y_pred_test, train_data_elimination, test_data_elimination, classification_train, classification_test

# %%
