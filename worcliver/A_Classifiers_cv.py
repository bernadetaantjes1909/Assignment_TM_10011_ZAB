
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, learning_curve
from sklearn.metrics import accuracy_score, roc_curve, auc


#%%
# Random Forest
def random_forest_classifier(X_train, X_test, y_train, y_test, plot=False, title_suffix=""):
    rf = RandomForestClassifier(random_state=42, bootstrap=True)

    param_dist = {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 5, 10],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
        "max_features": ["sqrt", "log2"]
    }

    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_dist,
        n_iter=5, #WAS 15 PAS DIT WEER AAN
        scoring="accuracy",
        cv=inner_cv,
        n_jobs=-1,
        random_state=42,
        refit=True
    )

    search.fit(X_train, y_train)

    tuned_model = search.best_estimator_
    best_params = search.best_params_

    y_pred_train = tuned_model.predict(X_train)
    y_pred_test = tuned_model.predict(X_test)

    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)

    print(f"Best CV accuracy Random Forest {title_suffix}: {search.best_score_ * 100:.2f}%")
    print(f"Train accuracy Random Forest {title_suffix}: {train_acc * 100:.2f}%")
    print(f"Test accuracy Random Forest {title_suffix}: {test_acc * 100:.2f}%")
    print(f"Best parameters Random Forest {title_suffix}: {best_params}")

    roc_auc = None
    if hasattr(tuned_model, "predict_proba"):
        y_score_test = tuned_model.predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, y_score_test)
        roc_auc = auc(fpr, tpr)

        if plot:
            plt.figure(figsize=(6, 6))
            plt.plot(fpr, tpr, linewidth=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
            plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="Random classifier")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"ROC Curve - Random Forest {title_suffix}")
            plt.legend(loc="lower right")
            plt.grid(True)
            plt.show()

    if plot:
        train_sizes, train_scores, val_scores = learning_curve(
            estimator=tuned_model,
            X=X_train,
            y=y_train,
            cv=inner_cv,
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
        plt.plot(train_sizes, train_scores_mean, marker="o", label="Training accuracy")
        plt.plot(train_sizes, val_scores_mean, marker="o", label="Validation accuracy")
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
        plt.xlabel("Training set size")
        plt.ylabel("Accuracy")
        plt.title(f"Learning Curve - Random Forest {title_suffix}")
        plt.legend()
        plt.grid(True)
        plt.show()

    return {
        "best_params": best_params,
        "best_cv_score": search.best_score_,
        "train_acc": train_acc,
        "test_acc": test_acc,
        "roc_auc": roc_auc,
        "y_pred_train": y_pred_train,
        "y_pred_test": y_pred_test,
        "y_score_test": y_score_test,
        "model": tuned_model
    }

#%%
# kNN

def knn_classifier(X_train, X_test, y_train, y_test, plot=False, title_suffix=""):
    knn = KNeighborsClassifier()

    param_dist = {
        "n_neighbors": [3, 5, 7, 9, 11, 15, 21],
        "weights": ["uniform", "distance"],
        "metric": ["minkowski"],
        "p": [1, 2]
    }

    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    search = RandomizedSearchCV(
        estimator=knn,
        param_distributions=param_dist,
        n_iter=15,
        scoring="accuracy",
        cv=inner_cv,
        n_jobs=-1,
        random_state=42,
        refit=True
    )

    search.fit(X_train, y_train)

    tuned_model = search.best_estimator_
    best_params = search.best_params_

    y_pred_train = tuned_model.predict(X_train)
    y_pred_test = tuned_model.predict(X_test)

    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)

    print(f"Best CV accuracy kNN {title_suffix}: {search.best_score_ * 100:.2f}%")
    print(f"Train accuracy kNN {title_suffix}: {train_acc * 100:.2f}%")
    print(f"Test accuracy kNN {title_suffix}: {test_acc * 100:.2f}%")
    print(f"Best parameters kNN {title_suffix}: {best_params}")

    roc_auc = None
    if hasattr(tuned_model, "predict_proba"):
        y_score_test = tuned_model.predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, y_score_test)
        roc_auc = auc(fpr, tpr)

        if plot:
            plt.figure(figsize=(6, 6))
            plt.plot(fpr, tpr, linewidth=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
            plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="Random classifier")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"ROC Curve - kNN {title_suffix}")
            plt.legend(loc="lower right")
            plt.grid(True)
            plt.show()

    if plot:
        train_sizes, train_scores, val_scores = learning_curve(
            estimator=tuned_model,
            X=X_train,
            y=y_train,
            cv=inner_cv,
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
        plt.plot(train_sizes, train_scores_mean, marker="o", label="Training accuracy")
        plt.plot(train_sizes, val_scores_mean, marker="o", label="Validation accuracy")
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
        plt.xlabel("Training set size")
        plt.ylabel("Accuracy")
        plt.title(f"Learning Curve - kNN {title_suffix}")
        plt.legend()
        plt.grid(True)
        plt.show()

    return {
        "best_params": best_params,
        "best_cv_score": search.best_score_,
        "train_acc": train_acc,
        "test_acc": test_acc,
        "roc_auc": roc_auc,
        "y_pred_train": y_pred_train,
        "y_pred_test": y_pred_test,
        "y_score_test": y_score_test,
        "model": tuned_model
    }


#%%
# SVM

def svm_classifier(X_train, X_test, y_train, y_test, plot=False, title_suffix=""):
    svm_model = SVC(random_state=42, probability=False)

    param_dist = {
        "C": [0.0001, 0.001], # WAS  [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
        "kernel": ["linear"],
        "gamma": ["scale", "auto"]
    }

    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    search = RandomizedSearchCV(
        estimator=svm_model,
        param_distributions=param_dist,
        n_iter=15,
        scoring="accuracy",
        cv=inner_cv,
        n_jobs=-1,
        random_state=42,
        refit=True
    )

    search.fit(X_train, y_train)

    tuned_model = search.best_estimator_
    best_params = search.best_params_

    y_pred_train = tuned_model.predict(X_train)
    y_pred_test = tuned_model.predict(X_test)

    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)

    print(f"Best CV accuracy SVM {title_suffix}: {search.best_score_ * 100:.2f}%")
    print(f"Train accuracy SVM {title_suffix}: {train_acc * 100:.2f}%")
    print(f"Test accuracy SVM {title_suffix}: {test_acc * 100:.2f}%")
    print(f"Best parameters SVM {title_suffix}: {best_params}")

    roc_auc = None
    if hasattr(tuned_model, "predict_proba"):
        y_score_test = tuned_model.predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, y_score_test)
        roc_auc = auc(fpr, tpr)

        if plot:
            plt.figure(figsize=(6, 6))
            plt.plot(fpr, tpr, linewidth=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
            plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="Random classifier")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"ROC Curve - SVM {title_suffix}")
            plt.legend(loc="lower right")
            plt.grid(True)
            plt.show()

    if plot:
        train_sizes, train_scores, val_scores = learning_curve(
            estimator=tuned_model,
            X=X_train,
            y=y_train,
            cv=inner_cv,
            scoring="accuracy",
            train_sizes=np.linspace(0.2, 1.0, 5),
            n_jobs=-1,
            shuffle=True,
            random_state=42
        )

        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        val_scores_mean = np.mean(val_scores, axis=1)
        val_scores_std = np.std(val_scores, axis=1)

        plt.figure(figsize=(7, 5))
        plt.plot(train_sizes, train_scores_mean, marker="o", label="Training accuracy")
        plt.plot(train_sizes, val_scores_mean, marker="o", label="Validation accuracy")
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
        plt.xlabel("Training set size")
        plt.ylabel("Accuracy")
        plt.title(f"Learning Curve - SVM {title_suffix}")
        plt.legend()
        plt.grid(True)
        plt.show()

    return {
        "best_params": best_params,
        "best_cv_score": search.best_score_,
        "train_acc": train_acc,
        "test_acc": test_acc,
        "roc_auc": roc_auc,
        "y_pred_train": y_pred_train,
        "y_pred_test": y_pred_test,
        "y_score_test": y_score_test,
        "model": tuned_model
    }