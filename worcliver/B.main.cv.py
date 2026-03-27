#%%
import numpy as np
import importlib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE, SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import A_Classifiers_cv
import A_Feature_selection_cv
import A_Load_data_cv
import A_preprocessing_cv
import B_Classifier
import B_Feature_selection
importlib.reload(A_Classifiers_cv)
importlib.reload(A_Feature_selection_cv)
importlib.reload(A_Load_data_cv)
importlib.reload(A_preprocessing_cv)
importlib.reload(B_Classifier)
importlib.reload(B_Feature_selection)

from A_Load_data_cv import load_data
from A_preprocessing_cv import preprocessing_data
from B_Classifier import classifiers  
from B_Feature_selection import feature_selectors

#%%
X,y = load_data()

# 1. Load + split (fixed test set)
X_train_full, X_test, y_train_full, y_test = preprocessing_data(X,y)

# 2. Outer CV alleen op train set
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

#%%



def evaluate_pipeline(X, y, fs_name, clf_name, cv_outer=5, cv_inner=3):

    fs_config = feature_selectors[fs_name]
    clf_config = classifiers[clf_name]
    
    # Bouw de pipeline
    pipeline = Pipeline([
        ("feature_selection", fs_config["selector"]),
        ("classifier", clf_config["clf"])
    ])

    # Combineer hyperparameters
    #param_grid = {**fs_config["params"], **clf_config["params"]}
    param_grid = {}
    for key, value in fs_config["params"].items():
        param_grid[f"feature_selection__{key}"] = value
        
    for key, value in clf_config["params"].items():
        param_grid[f"classifier__{key}"] = value

    # Outer CV setup
    outer_cv = StratifiedKFold(n_splits=cv_outer, shuffle=True, random_state=42)
    inner_cv = StratifiedKFold(n_splits=cv_inner, shuffle=True, random_state=42)
    
    outer_scores = []
    fold_aucs = []
    best_params_per_fold = []
    
    name = f"{fs_name} + {clf_name}"
    
    plt.figure(figsize=(6, 6))
    
    for fold, (train_idx, val_idx) in enumerate(outer_cv.split(X, y), 1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Inner CV voor hyperparameter tuning
        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=inner_cv,
            scoring="accuracy",
            n_jobs=-1,
            refit=True
        )
        
        grid_search.fit(X_train, y_train)
        best_params_per_fold.append(grid_search.best_params_)
        
        # Evalueer op outer validation fold
        y_pred = grid_search.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        outer_scores.append(acc)
        
        # ROC curve
        if hasattr(grid_search.best_estimator_.named_steps["classifier"], "predict_proba"):
            y_score = grid_search.predict_proba(X_val)[:, 1]
            fpr, tpr, _ = roc_curve(y_val, y_score)
            fold_auc = auc(fpr, tpr)
            fold_aucs.append(fold_auc)
            plt.plot(fpr, tpr, linewidth=1.5, label=f"Fold {fold} (AUC = {fold_auc:.3f})")
    
    # ROC plot
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Nested CV ROC - {name}")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # Summary statistics
    mean_acc = np.mean(outer_scores)
    std_acc = np.std(outer_scores)
    ci95 = 1.96 * std_acc / np.sqrt(len(outer_scores))
    
    mean_auc = np.mean(fold_aucs) if fold_aucs else None
    std_auc = np.std(fold_aucs) if fold_aucs else None
    
    print(f"{name} → Accuracy: {mean_acc:.3f} ± {ci95:.3f} | AUC: {mean_auc:.3f}" if mean_auc else "")
    
    return {
        "name": name,
        "cv_mean_acc": mean_acc,
        "cv_std_acc": std_acc,
        "cv_ci95": ci95,
        "cv_mean_auc": mean_auc,
        "cv_std_auc": std_auc,
        "best_params_per_fold": best_params_per_fold
    }


#%%
results = []

for fs_name in feature_selectors.keys():
    for clf_name in classifiers.keys():
        print(f"\n{'='*50}")
        print(f"Evaluating: {fs_name} + {clf_name}")
        print('='*50)
        
        result = evaluate_pipeline(X_train_full, y_train_full, fs_name, clf_name)
        results.append(result)

# Resultaten overzicht
results_df = pd.DataFrame(results)
print("\n" + "="*60)
print("FINAL RESULTS")
print("="*60)
print(results_df[["name", "cv_mean_acc", "cv_ci95", "cv_mean_auc"]]
      .sort_values(by="cv_mean_acc", ascending=False)
      .to_string(index=False))
