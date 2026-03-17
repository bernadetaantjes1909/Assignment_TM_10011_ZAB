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









#%%
def random_forest_classifier(load_data, preprocessing_data, deleting_zero_variance, feature_selection_fn):
    # Note: renamed parameter to feature_selection_fn to avoid shadowing the imported module
    train_data_elimination, test_data_elimination, classification_train, classification_test = feature_selection_fn(
        load_data, preprocessing_data, deleting_zero_variance
    )

    rf = RandomForestClassifier(random_state=42, bootstrap=True)

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
        random_state=42,
        return_train_score=True  # lets you inspect train vs. validation gap
    )

    # Only fit on training data — test set is untouched
    search.fit(train_data_elimination, classification_train)

    best_params = search.best_params_
    best_cv_score = search.best_score_          # mean CV accuracy on train folds
    tuned_model = search.best_estimator_

    print(f"Best parameters: {best_params}")
    print(f"Best cross-validated accuracy: {best_cv_score:.4f}")

    # Return test data too so it's ready when you need it later
    return (
        tuned_model,
        best_params,
        best_cv_score,
        train_data_elimination,
        test_data_elimination,
        classification_train,
        classification_test
    )


 # %%
tuned_model, best_params, best_cv_score, train_data, test_data, y_train, y_test = random_forest_classifier(
    load_data, preprocessing_data, deleting_zero_variance, feature_selection_fn
)

print(f"Accuracy: {best_cv_score:.4f}")        # e.g. 0.9321
print(f"Accuracy: {best_cv_score * 100:.2f}%") # e.g. 93.21%
# %%
