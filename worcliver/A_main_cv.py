#%%
import numpy as np
import pandas as pd
import importlib
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
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
from A_Feature_selection_cv import feature_filtering, feature_selection_PCA, feature_selection_RFE
from A_Classifiers_cv import random_forest_classifier, knn_classifier, svm_classifier

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

def evaluate_combination(X_train_full, y_train_full, X_test, y_test,
                        feature_selection, classifier, name):

    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    outer_scores = []
    fold_aucs = []

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

    mean_score = np.mean(outer_scores)
    std_score = np.std(outer_scores)

    plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Outer-CV ROC per fold - {name}")
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
    feature_selection_PCA, knn_classifier, "PCA + kNN"))

results.append(evaluate_combination(X_train_full, y_train_full, X_test, y_test,
    feature_selection_RFE, knn_classifier, "RFE + kNN"))

results.append(evaluate_combination(X_train_full, y_train_full, X_test, y_test,
    feature_selection_PCA, svm_classifier, "PCA + SVM"))

results.append(evaluate_combination(X_train_full, y_train_full, X_test, y_test,
    feature_selection_RFE, svm_classifier, "RFE + SVM"))

results_df = pd.DataFrame(results)
print("\n=== FINAL RESULTS ===")
print(results_df.sort_values(by="cv_mean", ascending=False))
# %%
