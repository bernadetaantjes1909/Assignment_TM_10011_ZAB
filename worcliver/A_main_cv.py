#Main file:
#In the main file, the different feature selection methods are combined with the different classifiers and these combinations are tested on the test data to see which has the highest accuracy. This file contains:

#--> importing functions and other files
#--> Imports data via load_data and splits data via "preprocessing_data
#--> Determines the hyperparameter grid that is used for the grid search (this is manually changed in the classifier functions, this is already done!)
#--> "evaluate_combination" combines the feature_selection_methods with the classifiers, performes a cross validation
#--> "final_evaluation" performes the final evaluation of the best performing model on the testset
#--> "plot_learning_curve" plots a learning curve for this final model 

#%%
#--> importing functions and other files
import numpy as np
import pandas as pd
import importlib
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from scipy.stats import t

import A_Load_data_cv
import A_preprocessing_cv
import A_Feature_selection_cv
import A_Classifiers_cv

importlib.reload(A_Classifiers_cv)
importlib.reload(A_Feature_selection_cv)
importlib.reload(A_Load_data_cv)
importlib.reload(A_preprocessing_cv)

from A_Load_data_cv import load_data
from A_preprocessing_cv import preprocessing_data
from A_Feature_selection_cv import feature_filtering, feature_selection_PCA, feature_selection_RFE, feature_selection_L1
from A_Classifiers_cv import random_forest_classifier, knn_classifier, svm_classifier, \
    random_forest_coarse_search, random_forest_fine_search, \
    knn_coarse_search, knn_fine_search, \
    svm_coarse_search, svm_fine_search

#%%
#--> Imports data via load_data and splits data via "preprocessing_data"
X,y = load_data()
X_train_full, X_test, y_train_full, y_test = preprocessing_data(X,y)
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


 #%%
 #--> Determines the hyperparameter grid that is used for the grid search (this is manually changed in the classifier functions, this is already done!)

# This section is run once to find hyperparameter grid, that is changed in the classifier functions

scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train_full)  
X_test_scaled = scaler.transform(X_test)

#PCA----------------
X_train_search, _, y_train_search, _, _ = feature_selection_PCA(X_train_scaled, X_test_scaled, y_train_full, y_test)

#random forest
rf_coarse_params = random_forest_coarse_search(X_train_search, y_train_search)
rf_fine_params = random_forest_fine_search(X_train_search, y_train_search, rf_coarse_params)
print(f"\nFinal RF params to use: {rf_fine_params}")

#kNN
knn_coarse_params = knn_coarse_search(X_train_search, y_train_search)
knn_fine_params = knn_fine_search(X_train_search, y_train_search, knn_coarse_params)
print(f"\nFinal kNN params to use: {knn_fine_params}")

# SVM
svm_coarse_params = svm_coarse_search(X_train_search, y_train_search)
svm_fine_params = svm_fine_search(X_train_search, y_train_search, svm_coarse_params)
print(f"\nFinal SVM params to use: {svm_fine_params}")

#RFE----------------
X_train_search, _, y_train_search, _, _ = feature_selection_RFE(X_train_scaled, X_test_scaled, y_train_full, y_test)
#random forest
rf_coarse_params = random_forest_coarse_search(X_train_search, y_train_search)
rf_fine_params = random_forest_fine_search(X_train_search, y_train_search, rf_coarse_params)
print(f"\nFinal RF params to use: {rf_fine_params}")

#kNN
knn_coarse_params = knn_coarse_search(X_train_search, y_train_search)
knn_fine_params = knn_fine_search(X_train_search, y_train_search, knn_coarse_params)
print(f"\nFinal kNN params to use: {knn_fine_params}")
# SVM
svm_coarse_params = svm_coarse_search(X_train_search, y_train_search)
svm_fine_params = svm_fine_search(X_train_search, y_train_search, svm_coarse_params)
print(f"\nFinal SVM params to use: {svm_fine_params}")

#Lasso----------------
X_train_search, _, y_train_search, _, _ = feature_selection_L1(X_train_scaled, X_test_scaled, y_train_full, y_test  )
# Random Forest
rf_coarse_params = random_forest_coarse_search(X_train_search, y_train_search)
rf_fine_params = random_forest_fine_search(X_train_search, y_train_search, rf_coarse_params)
print(f"\nFinal RF params to use: {rf_fine_params}")

# kNN
knn_coarse_params = knn_coarse_search(X_train_search, y_train_search)
knn_fine_params = knn_fine_search(X_train_search, y_train_search, knn_coarse_params)
print(f"\nFinal kNN params to use: {knn_fine_params}")

# SVM
svm_coarse_params = svm_coarse_search(X_train_search, y_train_search)
svm_fine_params = svm_fine_search(X_train_search, y_train_search, svm_coarse_params)
print(f"\nFinal SVM params to use: {svm_fine_params}")

#%%
#--> "evaluate_combination" combines the feature_selection_methods with the classifiers, performes a cross validation
def evaluate_combination(X_train_full, y_train_full,
                         feature_selection, classifier, name):

    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    outer_scores = []
    fold_aucs = []

    plt.figure(figsize=(6, 6))

    for fold, (train_idx, val_idx) in enumerate(outer_cv.split(X_train_full, y_train_full), 1):

        X_train = X_train_full[train_idx]
        X_val   = X_train_full[val_idx]
        y_train = y_train_full[train_idx]
        y_val   = y_train_full[val_idx]

        # Scaler
        scaler = preprocessing.RobustScaler()
        X_train = scaler.fit_transform(X_train)
        X_val   = scaler.transform(X_val)

        #feature selection
        X_train_fs, X_val_fs, y_train_fs, y_val_fs, info = feature_selection(
            X_train, X_val, y_train, y_val
        )

        # Train classifier
        result = classifier(
            X_train_fs,
            X_val_fs,
            y_train_fs,
            y_val_fs,
            plot=False,
            title_suffix=f"{name} fold {fold}"
        )
        outer_scores.append(result["test_acc"])

        # ROC/AUC 
        if result["y_score_test"] is not None:
            fpr, tpr, _ = roc_curve(y_val_fs, result["y_score_test"])
            fold_auc = auc(fpr, tpr)
            fold_aucs.append(fold_auc)

            plt.plot(fpr, tpr, linewidth=1.5, label=f"Fold {fold} (AUC = {fold_auc:.3f})")

    mean_score = np.mean(outer_scores)
    std_score = np.std(outer_scores)
    n = len(outer_scores)
    ci95 = 1.96 * std_score / np.sqrt(n)

    # ROC plot
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Outer-CV ROC per fold - {name}")
    plt.legend()
    plt.grid(True)
    plt.show()

    # print mean accuracy for every combination
    print(f"\n{name} → CV accuracy: {mean_score:.3f} ± {ci95:.3f}")

    return {
        "name": name,
        "cv_mean": mean_score,
        "cv_std": std_score,
        "cv_ci95": ci95,
        "cv_mean_auc": np.mean(fold_aucs) if len(fold_aucs) > 0 else None,
        "cv_std_auc": np.std(fold_aucs) if len(fold_aucs) > 0 else None
    }
#%% 
# print results for all combinations
results = []

feature_selectors = [
    (feature_selection_PCA, "PCA"),
    (feature_selection_RFE, "RFE"),
    (feature_selection_L1, "Lasso")
]

classifiers = [
    (random_forest_classifier, "RF"),
    (knn_classifier, "kNN"),
    (svm_classifier, "SVM")
]

for feature_selection, fs_name in feature_selectors:
    for classifier, clf_name in classifiers:        
        results.append(
            evaluate_combination(
                X_train_full,
                y_train_full,
                feature_selection,
                classifier,
                f"{fs_name} + {clf_name}"
            )
        )

results_df = pd.DataFrame(results)
print("FINAL RESULTS")
print(results_df.sort_values(by="cv_mean", ascending=False))
# %%
#--> "final_evaluation" performes the final evaluation of the best performing model on the testset

def final_evaluation(X_train_full, y_train_full, X_test, y_test,
                         feature_selection, classifier, name):

    plt.figure(figsize=(6, 6))

    scaler = preprocessing.RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train_full)
    X_test_scaled = scaler.transform(X_test)

    X_train_fs, X_test_fs, y_train_fs, y_test_fs, info = feature_selection(
        X_train_scaled, X_test_scaled, y_train_full, y_test
    )

    result = classifier(
        X_train_fs,
        X_test_fs,
        y_train_fs,
        y_test_fs,
        plot=False
    )
    test_acc = result["test_acc"]
    # ROC + AUC
    test_auc = None
    if result["y_score_test"] is not None:
        fpr, tpr, _ = roc_curve(y_test_fs, result["y_score_test"])
        test_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, linewidth=1.5, label=f"AUC = {test_auc:.3f}")
        plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"Test ROC - {name}")
        plt.legend()
        plt.grid(True)
        plt.show()

    print(f"{name} → Test accuracy: {test_acc:.3f}± {ci95:.3f}")
    
    # confusion matrix
    y_pred_test = result["y_pred_test"]
    cm = confusion_matrix(y_test_fs, y_pred_test)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax)
    plt.title(f"Confusion Matrix - {name}")
    plt.grid(False)
    plt.show()

    return {
        "name": name,
        "test_acc": test_acc,
        "test_auc": test_auc
    }
# %%
result = final_evaluation(X_train_full, y_train_full, X_test, y_test,
    feature_selection_RFE, svm_classifier, "RFE + SVM")
# %%
#--> "plot_learning_curve" plots a learning curve for this final model 
def plot_learning_curve(X_train_full, y_train_full,
                               feature_selection, classifier, name,
                               train_sizes=np.linspace(0.1, 1.0, 8),
                               n_splits=5,
                               random_state=42):

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    train_sizes_abs = []
    train_scores_mean = []
    train_scores_std = []
    val_scores_mean = []
    val_scores_std = []

    for frac in train_sizes:
        fold_train_scores = []
        fold_val_scores = []
        current_train_size = []

        for train_idx, val_idx in cv.split(X_train_full, y_train_full):
            X_train = X_train_full[train_idx]
            X_val   = X_train_full[val_idx]
            y_train = y_train_full[train_idx]
            y_val   = y_train_full[val_idx]

            # take only a fraction of the training fold
            n_sub = max(2, int(len(X_train) * frac))
            X_train_sub = X_train[:n_sub]
            y_train_sub = y_train[:n_sub]

            current_train_size.append(len(X_train_sub))

            scaler = preprocessing.RobustScaler()
            X_train_sub = scaler.fit_transform(X_train_sub)
            X_val_scaled = scaler.transform(X_val)

            X_train_fs, X_val_fs, y_train_fs, y_val_fs, info = feature_selection(
                X_train_sub, X_val_scaled, y_train_sub, y_val
            )

            result = classifier(
                X_train_fs,
                X_val_fs,
                y_train_fs,
                y_val_fs,
                plot=False
            )

            val_acc = result["test_acc"]
            fold_val_scores.append(val_acc)

            if "y_pred_train" in result and result["y_pred_train"] is not None:
                train_acc = accuracy_score(y_train_fs, result["y_pred_train"])
                fold_train_scores.append(train_acc)

        train_sizes_abs.append(int(np.mean(current_train_size)))

        if len(fold_train_scores) > 0:
            train_scores_mean.append(np.mean(fold_train_scores))
            train_scores_std.append(np.std(fold_train_scores))
        else:
            train_scores_mean.append(np.nan)
            train_scores_std.append(np.nan)

        val_scores_mean.append(np.mean(fold_val_scores))
        val_scores_std.append(np.std(fold_val_scores))

    plt.figure(figsize=(7, 5))
    plt.plot(train_sizes_abs, train_scores_mean, marker='o', label='Training accuracy')
    plt.plot(train_sizes_abs, val_scores_mean, marker='o', label='Validation accuracy')
    plt.fill_between(train_sizes_abs,np.array(train_scores_mean) - np.array(train_scores_std),np.array(train_scores_mean) + np.array(train_scores_std),alpha=0.2)
    plt.fill_between(train_sizes_abs,np.array(val_scores_mean) - np.array(val_scores_std),np.array(val_scores_mean) + np.array(val_scores_std),alpha=0.2)
    plt.xlabel("Training set size")
    plt.ylabel("Accuracy")
    plt.title(f"Learning Curve - {name}")
    plt.legend()
    plt.grid(True)
    plt.show()

    #%%
    #learning curve plotten
    plot_learning_curve(X_train_full, y_train_full, feature_selection_RFE, random_forest_classifier,"RFE + Random Forest")
