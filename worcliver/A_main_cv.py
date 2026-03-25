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
from sklearn.metrics import roc_curve, auc

importlib.reload(A_Classifiers_cv)
importlib.reload(A_Feature_selection_cv)
importlib.reload(A_Load_data_cv)
importlib.reload(A_preprocessing_cv)

from A_Load_data_cv import load_data
from A_preprocessing_cv import preprocessing_data
from A_Feature_selection_cv import feature_filtering, feature_selection_PCA, feature_selection_RFE, feature_selection_L1

# ← PUT IT HERE with the other from imports
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

results = []
combinations = [
    ("PCA", "RF", feature_selection_PCA, random_forest_classifier),
    ("RFE", "RF", feature_selection_RFE, random_forest_classifier),
    ("PCA", "kNN", feature_selection_PCA, knn_classifier),
    ("RFE", "kNN", feature_selection_RFE, knn_classifier),
    ("PCA", "SVM", feature_selection_PCA, svm_classifier),
    ("RFE", "SVM", feature_selection_RFE, svm_classifier),
]


#%%
# === HYPERPARAMETER OPTIMISATION ===
# Run this section ONCE to find the best params.
# After finding them, you can comment this whole section out
# and paste the best params directly into the classifier functions.

# Apply feature selection once on full training data to get data for search
# We use PCA here as a representative — run separately for RFE/L1 if needed
X_train_search, _, y_train_search, _, _ = feature_selection_PCA(
    X_train_full, X_test, y_train_full, y_test
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

def evaluate_combination(X_train_full, y_train_full, X_test, y_test,
                        feature_selection, classifier, name):

    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    outer_scores = []
    fold_aucs = []
    all_train_scores = []
    all_val_scores = []
    train_sizes = None

    plt.figure(figsize=(6, 6))

    for fold, (train, val) in enumerate(outer_cv.split(X_train_full, y_train_full), 1):

        X_train = X_train_full[train]
        X_val   = X_train_full[val]
        y_train = y_train_full[train]
        y_val   = y_train_full[val]

        scaler = preprocessing.RobustScaler()
        X_train = scaler.fit_transform(X_train)
        X_val   = scaler.transform(X_val)

        X_train_fs, X_val_fs, y_train_fs, y_val_fs, info = feature_selection(
            X_train, X_val, y_train, y_val
        )

        result = classifier(
            X_train_fs,
            X_val_fs,
            y_train_fs,
            y_val_fs,
            plot=False,
            title_suffix=f"{name} fold {fold}"
        )

        outer_scores.append(result["test_acc"])

        # Gebruik scores/probabilities voor ROC, niet y_pred_test
        if result["y_score_test"] is not None:
            fpr, tpr, _ = roc_curve(y_val_fs, result["y_score_test"])
            fold_auc = auc(fpr, tpr)
            fold_aucs.append(fold_auc)

            plt.plot(fpr, tpr, linewidth=1.5, label=f"Fold {fold} (AUC = {fold_auc:.3f})")
        
        sizes, train_scores, val_scores = learning_curve(
        estimator=result["model"],   # jouw getrainde model
        X=X_train_fs,
        y=y_train_fs,
        cv=3,                        # inner CV (kleiner houden)
        scoring="accuracy",
        train_sizes=np.linspace(0.2, 1.0, 5),
        n_jobs=-1
        )

        all_train_scores.append(np.mean(train_scores, axis=1))
        all_val_scores.append(np.mean(val_scores, axis=1))

        if train_sizes is None:
            train_sizes = sizes

    mean_score = np.mean(outer_scores)
    std_score = np.std(outer_scores)
    mean_train = np.mean(all_train_scores, axis=0)
    std_train  = np.std(all_train_scores, axis=0)
    mean_val = np.mean(all_val_scores, axis=0)
    std_val  = np.std(all_val_scores, axis=0)

    # plot ROC curve
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Outer-CV ROC per fold - {name}")
    plt.legend()
    plt.grid(True)
    plt.show()

    # plot learning curve
    plt.figure(figsize=(7,5))
    plt.plot(train_sizes, mean_train, marker='o', label="Train (mean)")
    plt.plot(train_sizes, mean_val, marker='o', label="Validation (mean)")
    plt.fill_between(train_sizes,mean_train - std_train,mean_train + std_train,alpha=0.2)
    plt.fill_between(train_sizes,mean_val - std_val,mean_val + std_val,alpha=0.2)
    plt.xlabel("Training set size")
    plt.ylabel("Accuracy")
    plt.title(f"Learning Curve (outer-CV mean) - {name}")
    plt.legend()
    plt.grid(True)
    plt.show()

    print(f"\n{name} → CV accuracy: {mean_score:.3f} ± {std_score:.3f}")

    return {
        "name": name,
        "cv_mean": mean_score,
        "cv_std": std_score,
        "cv_mean_auc": np.mean(fold_aucs) if len(fold_aucs) > 0 else None,
        "cv_std_auc": np.std(fold_aucs) if len(fold_aucs) > 0 else None
    }
#%% 
results = []

results.append(evaluate_combination(X_train_full, y_train_full, X_test, y_test,
    feature_selection_PCA, random_forest_classifier, "PCA + RF"))
print('ben hier 1')
results.append(evaluate_combination(X_train_full, y_train_full, X_test, y_test,
    feature_selection_RFE, random_forest_classifier, "RFE + RF"))
print('ben hier 2')
results.append(evaluate_combination(X_train_full, y_train_full, X_test, y_test,
    feature_selection_L1, random_forest_classifier, "Lasso + RF"))

results.append(evaluate_combination(X_train_full, y_train_full, X_test, y_test,
    feature_selection_PCA, knn_classifier, "PCA + kNN"))

results.append(evaluate_combination(X_train_full, y_train_full, X_test, y_test,
    feature_selection_RFE, knn_classifier, "RFE + kNN"))

results.append(evaluate_combination(X_train_full, y_train_full, X_test, y_test,
    feature_selection_L1, knn_classifier, "Lasso + kNN"))

results.append(evaluate_combination(X_train_full, y_train_full, X_test, y_test,
    feature_selection_PCA, svm_classifier, "PCA + SVM"))

results.append(evaluate_combination(X_train_full, y_train_full, X_test, y_test,
    feature_selection_RFE, svm_classifier, "RFE + SVM"))

results.append(evaluate_combination(X_train_full, y_train_full, X_test, y_test,
    feature_selection_L1, svm_classifier, "Lasso + SVM"))

results_df = pd.DataFrame(results)
print("\n=== FINAL RESULTS ===")
print(results_df.sort_values(by="cv_mean", ascending=False))
# %%
