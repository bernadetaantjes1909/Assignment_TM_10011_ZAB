#%%
import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets as ds



from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, learning_curve

from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV

from sklearn.metrics import accuracy_score, roc_curve, auc



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
    print(f"Best CV accuracy random forest {feature_selection_fn.__name__}: {search.best_score_ * 100:.2f}%")


    y_pred_train = tuned_model.predict(train_data_elimination)
    y_pred_test = tuned_model.predict(test_data_elimination)
    # echte accuracy op test set (niet gebruikt voor hyperparameter tuning, dus eerlijk)

    train_acc = accuracy_score(classification_train, y_pred_train)
    test_acc  = accuracy_score(classification_test, y_pred_test)

    print(f"Train accuracy random forest {feature_selection_fn.__name__}: {train_acc * 100:.2f}%")
    print(f"Test accuracy random forest {feature_selection_fn.__name__}: {test_acc * 100:.2f}%")
    print(best_params)

    # ROC curve
    # ROC curve (binary classification)
    y_score_test = tuned_model.predict_proba(test_data_elimination)[:, 1]

    fpr, tpr, thresholds = roc_curve(
    classification_test,
    y_score_test,
    pos_label="malignant"
    )
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="Random classifier")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - Random Forest ({feature_selection_fn.__name__})")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()


    # learning curve
    train_sizes, train_scores, val_scores = learning_curve(
        estimator=tuned_model,
        X=train_data_elimination,
        y=classification_train,
        cv=cv,
        scoring="accuracy",
        train_sizes=np.linspace(0.1, 1.0, 5),
        n_jobs=-1,
        shuffle=True,
        random_state=42
    )

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)

    val_scores_mean = np.mean(val_scores, axis=1)
    val_scores_std = np.std(val_scores, axis=1)
    plt.figure(figsize=(7, 5))
    plt.plot(train_sizes, train_scores_mean, marker='o', label="Training accuracy")
    plt.plot(train_sizes, val_scores_mean, marker='o', label="Validation accuracy")

    plt.fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.2
    )
    plt.fill_between(
        train_sizes,
        val_scores_mean - val_scores_std,
        val_scores_mean + val_scores_std,
        alpha=0.2
    )
    plt.title(f"Trainingscurve - Random Forest ({feature_selection_fn.__name__})")

    return best_params, y_pred_train, y_pred_test, train_data_elimination, test_data_elimination, classification_train, classification_test


#%% 
# knn classifier met hyperparameter search
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

def knn_classifier(load_data, preprocessing_data, deleting_zero_variance, feature_selection_fn):
    train_data_elimination, test_data_elimination, classification_train, classification_test = feature_selection_fn(
        load_data, preprocessing_data, deleting_zero_variance
    )

    knn = KNeighborsClassifier()

    param_dist = {
        "n_neighbors": [3, 5, 7, 9, 11, 15, 21],
        "weights": ["uniform", "distance"],
        "metric": ["minkowski"],
        "p": [1, 2]
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    search = RandomizedSearchCV(
        knn,
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

    # Cross-validated accuracy
    print(f"Best CV accuracy kNN {feature_selection_fn.__name__}: {search.best_score_ * 100:.2f}%")

    # Predictions
    y_pred_train = tuned_model.predict(train_data_elimination)
    y_pred_test = tuned_model.predict(test_data_elimination)

    # echte accuracy op test set (niet gebruikt voor hyperparameter tuning, dus eerlijk)    train_acc = accuracy_score(classification_train, y_pred_train)
    train_acc = accuracy_score(classification_train, y_pred_train)
    test_acc  = accuracy_score(classification_test, y_pred_test)

    print(f"Train accuracy kNN {feature_selection_fn.__name__}: {train_acc * 100:.2f}%")
    print(f"Test accuracy kNN {feature_selection_fn.__name__}: {test_acc * 100:.2f}%")
    print(best_params)
    
    # ROC curve (binary classification)
    y_score_test = tuned_model.predict_proba(test_data_elimination)[:, 1]

    fpr, tpr, thresholds = roc_curve(
    classification_test,
    y_score_test,
    pos_label="malignant"
    )
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="Random classifier")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - Knn ({feature_selection_fn.__name__})")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()


    # learning curve
    train_sizes, train_scores, val_scores = learning_curve(
        estimator=tuned_model,
        X=train_data_elimination,
        y=classification_train,
        cv=cv,
        scoring="accuracy",
        train_sizes=np.linspace(0.1, 1.0, 5),
        n_jobs=-1,
        shuffle=True,
        random_state=42
    )

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)

    val_scores_mean = np.mean(val_scores, axis=1)
    val_scores_std = np.std(val_scores, axis=1)
    plt.figure(figsize=(7, 5))
    plt.plot(train_sizes, train_scores_mean, marker='o', label="Training accuracy")
    plt.plot(train_sizes, val_scores_mean, marker='o', label="Validation accuracy")

    plt.fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.2
    )
    plt.fill_between(
        train_sizes,
        val_scores_mean - val_scores_std,
        val_scores_mean + val_scores_std,
        alpha=0.2
    )
    plt.title(f"Trainingscurve - Knn ({feature_selection_fn.__name__})")

    return best_params, y_pred_train, y_pred_test, train_data_elimination, test_data_elimination, classification_train, classification_test

#%% Logistic Regression met L1 en hyperparameter search
