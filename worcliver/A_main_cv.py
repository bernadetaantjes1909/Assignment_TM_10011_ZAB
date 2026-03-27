#%%
import numpy as np
import pandas as pd
import importlib
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, learning_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder

import A_Load_data_cv
import A_preprocessing_cv
import A_Feature_selection_cv
import A_Classifiers_cv
from sklearn import preprocessing
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_curve, auc

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

X,y = load_data()

# 1. Load + split (fixed test set)
X_train_full, X_test, y_train_full, y_test = preprocessing_data(X,y)

# 2. Outer CV alleen op train set
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)



 #%%
# === HYPERPARAMETER OPTIMISATION ===
# Run this section ONCE to find the best params.
# After finding them, you can comment this whole section out
# and paste the best params directly into the classifier functions.

# Apply feature selection once on full training data to get data for search
# We use PCA here as a representative — run separately for RFE/L1 if needed
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train_full)  # scale first
X_test_scaled = scaler.transform(X_test)

print("=== PCA Hyperparameter Search ===")


X_train_search, _, y_train_search, _, _ = feature_selection_PCA(
    X_train_scaled, X_test_scaled, y_train_full, y_test  # now scaled
)

# --- Random Forest ---
print("=== Random Forest coarse search ===")
rf_coarse_params = random_forest_coarse_search(X_train_search, y_train_search)

print("=== Random Forest fine search ===")
rf_fine_params = random_forest_fine_search(X_train_search, y_train_search, rf_coarse_params)

print(f"\nFinal RF params to use: {rf_fine_params}")

# --- kNN ---
print("=== kNN coarse search ===")
knn_coarse_params = knn_coarse_search(X_train_search, y_train_search)

print("=== kNN fine search ===")
knn_fine_params = knn_fine_search(X_train_search, y_train_search, knn_coarse_params)

print(f"\nFinal kNN params to use: {knn_fine_params}")

# --- SVM ---
print("=== SVM coarse search ===")
svm_coarse_params = svm_coarse_search(X_train_search, y_train_search)

print("=== SVM fine search ===")
svm_fine_params = svm_fine_search(X_train_search, y_train_search, svm_coarse_params)

print(f"\nFinal SVM params to use: {svm_fine_params}")

print("=== RFE Hyperparameter Search ===")


X_train_search, _, y_train_search, _, _ = feature_selection_RFE(
    X_train_scaled, X_test_scaled, y_train_full, y_test  # now scaled
)

# --- Random Forest ---
print("=== Random Forest coarse search ===")
rf_coarse_params = random_forest_coarse_search(X_train_search, y_train_search)

print("=== Random Forest fine search ===")
rf_fine_params = random_forest_fine_search(X_train_search, y_train_search, rf_coarse_params)

print(f"\nFinal RF params to use: {rf_fine_params}")

# --- kNN ---
print("=== kNN coarse search ===")
knn_coarse_params = knn_coarse_search(X_train_search, y_train_search)

print("=== kNN fine search ===")
knn_fine_params = knn_fine_search(X_train_search, y_train_search, knn_coarse_params)

print(f"\nFinal kNN params to use: {knn_fine_params}")

# --- SVM ---
print("=== SVM coarse search ===")
svm_coarse_params = svm_coarse_search(X_train_search, y_train_search)

print("=== SVM fine search ===")
svm_fine_params = svm_fine_search(X_train_search, y_train_search, svm_coarse_params)

print(f"\nFinal SVM params to use: {svm_fine_params}")

print("=== Lasso Hyperparameter Search ===")


X_train_search, _, y_train_search, _, _ = feature_selection_L1(
    X_train_scaled, X_test_scaled, y_train_full, y_test  # now scaled
)

# --- Random Forest ---
print("=== Random Forest coarse search ===")
rf_coarse_params = random_forest_coarse_search(X_train_search, y_train_search)

print("=== Random Forest fine search ===")
rf_fine_params = random_forest_fine_search(X_train_search, y_train_search, rf_coarse_params)

print(f"\nFinal RF params to use: {rf_fine_params}")

# --- kNN ---
print("=== kNN coarse search ===")
knn_coarse_params = knn_coarse_search(X_train_search, y_train_search)

print("=== kNN fine search ===")
knn_fine_params = knn_fine_search(X_train_search, y_train_search, knn_coarse_params)

print(f"\nFinal kNN params to use: {knn_fine_params}")

# --- SVM ---
print("=== SVM coarse search ===")
svm_coarse_params = svm_coarse_search(X_train_search, y_train_search)

print("=== SVM fine search ===")
svm_fine_params = svm_fine_search(X_train_search, y_train_search, svm_coarse_params)

print(f"\nFinal SVM params to use: {svm_fine_params}")







# === AFTER RUNNING THIS SECTION ONCE ===
# Comment it out and paste the printed best params as fixed values
# directly into the param_dist in each classifier function in A_Classifiers_cv.py
# This way the final run skips searching and uses known best params directly.

#%%

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

        # Scale only on outer training fold
        scaler = preprocessing.RobustScaler()
        X_train = scaler.fit_transform(X_train)
        X_val   = scaler.transform(X_val)

        # Feature selection only using outer training fold
        X_train_fs, X_val_fs, y_train_fs, y_val_fs, info = feature_selection(
            X_train, X_val, y_train, y_val
        )

        # Train + inner CV hyperparameter tuning
        result = classifier(
            X_train_fs,
            X_val_fs,
            y_train_fs,
            y_val_fs,
            plot=False,
            title_suffix=f"{name} fold {fold}"
        )

        outer_scores.append(result["test_acc"])

        # ROC/AUC on outer validation fold
        if result["y_score_test"] is not None:
            fpr, tpr, _ = roc_curve(y_val_fs, result["y_score_test"])
            fold_auc = auc(fpr, tpr)
            fold_aucs.append(fold_auc)

            plt.plot(fpr, tpr, linewidth=1.5, label=f"Fold {fold} (AUC = {fold_auc:.3f})")

    # Summary statistics
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
results = []

results.append(evaluate_combination(X_train_full, y_train_full,
    feature_selection_PCA, random_forest_classifier, "PCA + RF"))
print('ben hier 1')
results.append(evaluate_combination(X_train_full, y_train_full,
    feature_selection_RFE, random_forest_classifier, "RFE + RF"))
print('ben hier 2')
results.append(evaluate_combination(X_train_full, y_train_full,
    feature_selection_L1, random_forest_classifier, "Lasso + RF"))

results.append(evaluate_combination(X_train_full, y_train_full,
    feature_selection_PCA, knn_classifier, "PCA + kNN"))

results.append(evaluate_combination(X_train_full, y_train_full,
    feature_selection_RFE, knn_classifier, "RFE + kNN"))

results.append(evaluate_combination(X_train_full, y_train_full,
    feature_selection_L1, knn_classifier, "Lasso + kNN"))

results.append(evaluate_combination(X_train_full, y_train_full,
    feature_selection_PCA, svm_classifier, "PCA + SVM"))

results.append(evaluate_combination(X_train_full, y_train_full,
    feature_selection_RFE, svm_classifier, "RFE + SVM"))

results.append(evaluate_combination(X_train_full, y_train_full,
    feature_selection_L1, svm_classifier, "Lasso + SVM"))

results_df = pd.DataFrame(results)
print("\n=== FINAL RESULTS ===")
print(results_df.sort_values(by="cv_mean", ascending=False))
# %%
#Best model tested on testset
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import roc_curve, auc, accuracy_score
from scipy.stats import t

def evaluate_combination_testset(X_train_full, y_train_full, X_test, y_test,
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

    # Accuracy
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

    print(f"{name} → Test accuracy: {test_acc:.3f}")
    if test_auc is not None:
        print(f"{name} → Test AUC: {test_auc:.3f}")

    return {
        "name": name,
        "test_acc": test_acc,
        "test_auc": test_auc
    }
# %%

result = evaluate_combination_testset(X_train_full, y_train_full, X_test, y_test,
    feature_selection_RFE, random_forest_classifier, "RFE + RF")