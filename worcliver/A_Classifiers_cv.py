import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, GridSearchCV, learning_curve
from sklearn.metrics import accuracy_score, roc_curve, auc


#%%
# Random Forest classifier
def random_forest_classifier(X_train, X_test, y_train, y_test, plot=False, title_suffix=""):
    rf = RandomForestClassifier(random_state=42, bootstrap=True)

# Afkaderen van het hyperparameter grid
    param_grid = {
        "n_estimators": [100, 150, 200, 250, 300],
        "max_depth": [5, 6, 7, 8, 9, 10],
        "min_samples_split": [ 3, 5, 7],
        "min_samples_leaf": [2, 4, 6],
        "max_features": ["sqrt", "log2"]
    }
# De cors validatie opbouwen, mbv stratifiedKFold wordt de balans in de klasse gehouden
    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# Hyperparameter optimalisatie toepassen
    search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        scoring="accuracy",
        cv=inner_cv,
        n_jobs=-1
    )

    search.fit(X_train, y_train)
# Het beste model selecteren en de bijbehorende parameters opslaan
    tuned_model = search.best_estimator_
    best_params = search.best_params_

    y_pred_train = tuned_model.predict(X_train)
    y_pred_test = tuned_model.predict(X_test)
# De accuraatheid van het model berekenen op zowel de trainings- als testset
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
# Resultaten printen
    print(f"Best CV accuracy Random Forest {title_suffix}: {search.best_score_ * 100:.2f}%")
    print(f"Train accuracy Random Forest {title_suffix}: {train_acc * 100:.2f}%")
    print(f"Test accuracy Random Forest {title_suffix}: {test_acc * 100:.2f}%")
    print(f"Best parameters Random Forest {title_suffix}: {best_params}")

    y_score_test = None
    roc_auc = None

# probability berekenen en op basis daar van een ROC curve plotten, alleen mogelijk als het model een predict_proba methode heeft
    if hasattr(tuned_model, "predict_proba"):
        y_score_test = tuned_model.predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, y_score_test)
        roc_auc = auc(fpr, tpr)
# ROC curve plotten
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
# Learning curve plotten op voorwaarde van aanroepen 
    if plot:
        train_sizes, train_scores, val_scores = learning_curve(
            estimator=tuned_model,
            X=X_train,
            y=y_train, 
            cv=inner_cv,
            scoring="accuracy",
            train_sizes=np.linspace(0.1, 1.0, 5),
            n_jobs=-1, #versnelt het rekenen
            shuffle=True, 
            random_state=42
        )
# Gemiddelde en standaarddeviatie van de scores berekenen voor de plot
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        val_scores_mean = np.mean(val_scores, axis=1)
        val_scores_std = np.std(val_scores, axis=1)
# Learning curve plotten
        plt.figure(figsize=(7, 5))
        plt.plot(train_sizes, train_scores_mean, marker="o", label="Training accuracy")
        plt.plot(train_sizes, val_scores_mean, marker="o", label="Validation accuracy")
        # std shading toevoegen aan train en val set 
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
# Random Forest hyperparameter optimisation. We doen eerst een grove search

def random_forest_coarse_search(X_train, y_train):
    rf = RandomForestClassifier(random_state=42, bootstrap=True)
# Afkaderen van het hyperparameter grid
    param_grid_coarse = {
        "n_estimators": [150, 200, 250],
        "max_depth": [6, 8, 10],
        "min_samples_split": [5, 7, 9],
        "min_samples_leaf": [2, 4, 6],
        "max_features": ["sqrt", "log2"]
    }
# De cors validatie opbouwen, mbv stratifiedKFold wordt de balans in de klasse gehouden
    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# Hyperparameter optimalisatie toepassen
    search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid_coarse,
        scoring="accuracy",
        cv=inner_cv,
        n_jobs=-1
    )
# Grid toepassen op de data
    search.fit(X_train, y_train)
    print(f"RF Coarse best CV accuracy: {search.best_score_ * 100:.2f}%")
    print(f"RF Coarse best params: {search.best_params_}")
    return search.best_params_

# De stappen in de grove search verkleinen
def random_forest_fine_search(X_train, y_train, coarse_params):
    rf = RandomForestClassifier(random_state=42, bootstrap=True)
    param_grid_fine = {
        # stappen van 50 rond de beste n_estimators 
        "n_estimators": [
            max(50, coarse_params["n_estimators"] - 50),
            coarse_params["n_estimators"],
            coarse_params["n_estimators"] + 50
        ],
        # stappen van 1 rond de max_depth
        "max_depth": [
            max(2, coarse_params["max_depth"] - 1),
            coarse_params["max_depth"],
            coarse_params["max_depth"] + 1
        ],
        # stappen van 2 rond min_sample_split
        "min_samples_split": [
            max(2, coarse_params["min_samples_split"] - 2),
            coarse_params["min_samples_split"],
            coarse_params["min_samples_split"] + 2
        ],
        # stappen van 2 rond min_samples_leaf
        "min_samples_leaf": [
            max(1, coarse_params["min_samples_leaf"] - 2),
            coarse_params["min_samples_leaf"],
            coarse_params["min_samples_leaf"] + 2
        ],
        "max_features": [coarse_params["max_features"]]
    }
    # cv setup blijft hetzelfde als voorgaande functie
    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    # Hyperparameter optimalisatie toepassen
    search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid_fine, 
        scoring="accuracy",
        cv=inner_cv,
        n_jobs=-1
    )
    # Grid toepassen op de data
    search.fit(X_train, y_train)
    print(f"RF Fine best CV accuracy: {search.best_score_ * 100:.2f}%")
    print(f"RF Fine best params: {search.best_params_}")
    return search.best_params_


#%%
# kNN
def knn_classifier(X_train, X_test, y_train, y_test, plot=False, title_suffix=""):
    knn = KNeighborsClassifier()
# Afkaderen van het hyperparameter grid
    param_grid = {
        "n_neighbors": [15, 17, 19, 21, 23, 25, 27],
        "weights": ["uniform", "distance"],
        "metric": ["minkowski"],
        "p": [1, 2]
    }
# De cros validatie opbouwen 
    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# Grid toepassen
    search = GridSearchCV(
        estimator=knn,
        param_grid=param_grid,
        scoring="accuracy",
        cv=inner_cv,
        n_jobs=-1
    )
    search.fit(X_train, y_train)
# Beste model selecteren
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

    y_score_test = None
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
# kNN hyperparameter optimisation

def knn_coarse_search(X_train, y_train):
    knn = KNeighborsClassifier()
    param_grid_coarse = {
        "n_neighbors": [3, 5, 7, 9, 11, 13, 15],
        "weights": ["uniform", "distance"],
        "metric": ["minkowski"],
        "p": [1, 2]
    }
    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    search = GridSearchCV(
        estimator=knn,
        param_grid=param_grid_coarse,
        scoring="accuracy",
        cv=inner_cv,
        n_jobs=-1
    )
    search.fit(X_train, y_train)
    print(f"kNN Coarse best CV accuracy: {search.best_score_ * 100:.2f}%")
    print(f"kNN Coarse best params: {search.best_params_}")
    return search.best_params_


def knn_fine_search(X_train, y_train, coarse_params):
    knn = KNeighborsClassifier()
    param_grid_fine = {
        "n_neighbors": [
            max(1, coarse_params["n_neighbors"] - 2),
            coarse_params["n_neighbors"],
            coarse_params["n_neighbors"] + 2
        ],
        "weights": [coarse_params["weights"]],
        "metric": ["minkowski"],
        "p": [coarse_params["p"]]
    }
    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    search = GridSearchCV(
        estimator=knn,
        param_grid=param_grid_fine,
        scoring="accuracy",
        cv=inner_cv,
        n_jobs=-1
    )
    search.fit(X_train, y_train)
    print(f"kNN Fine best CV accuracy: {search.best_score_ * 100:.2f}%")
    print(f"kNN Fine best params: {search.best_params_}")
    return search.best_params_


#%%
# SVM
def svm_classifier(X_train, X_test, y_train, y_test, plot=False, title_suffix=""):

    svm_model = LinearSVC(random_state=42, max_iter=2000)

    param_grid = {
        "C": [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.01]
    }

    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    search = GridSearchCV(
        estimator=svm_model,
        param_grid=param_grid,
        scoring="accuracy",
        cv=inner_cv,
        n_jobs=-1
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
    print(f"Validation accuracy SVM {title_suffix}: {test_acc * 100:.2f}%")
    print(f"Best parameters SVM {title_suffix}: {best_params}")

    y_score_test = None
    roc_auc = None

    if hasattr(tuned_model, "decision_function"):
        y_score_test = tuned_model.decision_function(X_test)
        fpr, tpr, _ = roc_curve(y_test, y_score_test)
        roc_auc = auc(fpr, tpr)

    elif hasattr(tuned_model, "predict_proba"):
        y_score_test = tuned_model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_score_test)
        roc_auc = auc(fpr, tpr)

    if plot and y_score_test is not None:
        plt.figure(figsize=(6, 6))
        plt.plot(fpr, tpr, linewidth=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
        plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="Random classifier")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve - SVM ({title_suffix})")
        plt.legend(loc="lower right")
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
# SVM hyperparameter optimisation

def svm_coarse_search(X_train, y_train):
    svm_model = LinearSVC(random_state=42, max_iter=2000)
    param_grid_coarse = {
        "C": [0.000001, 0.00001, 0.0001, 0.001, 0.01]
    }
    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    search = GridSearchCV(
        estimator=svm_model,
        param_grid=param_grid_coarse,
        scoring="accuracy",
        cv=inner_cv,
        n_jobs=-1
    )
    search.fit(X_train, y_train)
    print(f"SVM Coarse best CV accuracy: {search.best_score_ * 100:.2f}%")
    print(f"SVM Coarse best params: {search.best_params_}")
    return search.best_params_


def svm_fine_search(X_train, y_train, coarse_params):
    svm_model = LinearSVC(random_state=42, max_iter=2000)
    best_C = coarse_params["C"]
    param_grid_fine = {
        "C": [best_C / 5, best_C / 2, best_C, best_C * 2, best_C * 5]
    }
    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    search = GridSearchCV(
        estimator=svm_model,
        param_grid=param_grid_fine,
        scoring="accuracy",
        cv=inner_cv,
        n_jobs=-1
    )
    search.fit(X_train, y_train)
    print(f"SVM Fine best CV accuracy: {search.best_score_ * 100:.2f}%")
    print(f"SVM Fine best params: {search.best_params_}")
    return search.best_params_