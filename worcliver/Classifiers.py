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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score
import numpy as np

#%%
def random_forest_classifier(load_data, preprocessing_data, deleting_zero_variance, feature_selection_fn):
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
        random_state=42
    )

    # Only fit on training data
    search.fit(train_data_elimination, classification_train)

    best_params = search.best_params_
    tuned_model = search.best_estimator_

    # Cross-validated accuracy (honest estimate)
    print(f"Best CV accuracy: {search.best_score_ * 100:.2f}%")

    # Predictions (needed for misclassification counts in main file)
    y_pred_train = tuned_model.predict(train_data_elimination)
    y_pred_test = tuned_model.predict(test_data_elimination)

    return best_params, y_pred_train, y_pred_test, train_data_elimination, test_data_elimination, classification_train, classification_test


#%% 


#%% Logistic Regression met L1 en hyperparameter search
def logistic_regression_classifier(load_data, preprocessing_data, deleting_zero_variance, n_iter_search=50, random_state=42):
    # Data voorbereiden
    train_data_filtered, test_data_filtered, classification_train, classification_test = deleting_zero_variance(load_data, preprocessing_data)

    # Basis Logistic Regression
    base_model = LogisticRegression(
        solver="saga",  # nodig voor L1
        penalty="elasticnet",
        l1_ratio=1.0,   # L1
        max_iter=5000,
        random_state=random_state
    )

    # Hyperparameter space
    param_dist = {
        "C": np.logspace(-4, 4, 50),  # regularisatie sterkte
    }

    # Stratified K-Fold
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    # Randomized Search
    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        n_iter=n_iter_search,
        scoring='accuracy',
        cv=cv,
        verbose=1,
        random_state=random_state,
        n_jobs=-1
    )

    # Fit search
    search.fit(train_data_filtered, classification_train)
    best_model = search.best_estimator_

    # Thresholding / feature selectie
    selector = SelectFromModel(best_model, prefit=True, threshold="mean")  # selecteer features boven gemiddelde gewicht
    X_train_sel = selector.transform(train_data_filtered)
    X_test_sel = selector.transform(test_data_filtered)
  
    # Hertrain op geselecteerde features
    final_model = LogisticRegression(
        solver="saga",
        penalty="elasticnet",
        l1_ratio=1.0,
        C=best_model.C,
        max_iter=5000,
        random_state=random_state
    )
    final_model.fit(X_train_sel, classification_train)

    # Predict
    y_pred_train = final_model.predict(X_train_sel)
    y_pred_test = final_model.predict(X_test_sel)

    train_acc = accuracy_score(classification_train, y_pred_train)
    test_acc = accuracy_score(classification_test, y_pred_test)

    print(f"Train Accuracy (after feature selection): {train_acc:.4f}")
    print(f"Test Accuracy (after feature selection): {test_acc:.4f}")

    return final_model, y_pred_train, y_pred_test, X_train_sel, X_test_sel, y_train, y_test