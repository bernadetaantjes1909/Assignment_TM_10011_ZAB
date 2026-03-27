#Feature_selection_cv

import numpy as np

from sklearn import decomposition
from sklearn import feature_selection
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_classif, RFE
from sklearn.model_selection import StratifiedKFold, cross_val_score

#%%
def feature_filtering(X_train, X_test, y_train, y_test, var_threshold=0.01, corr_threshold=0.995):
# remove zero variance
    variances = np.var(X_train, axis=0)
    keep_var_mask = variances >= var_threshold

    X_train_var = X_train[:, keep_var_mask]
    X_test_var = X_test[:, keep_var_mask]
# remove high correlation
    if X_train_var.shape[1] > 1:
        corr_matrix = np.abs(np.corrcoef(X_train_var, rowvar=False))
        to_drop = set()
        for i in range(corr_matrix.shape[1]):
            for j in range(i + 1, corr_matrix.shape[1]):
                if corr_matrix[i, j] > corr_threshold:
                    to_drop.add(j)
        keep_corr_mask = np.ones(X_train_var.shape[1], dtype=bool)
        if len(to_drop) > 0:
            keep_corr_mask[list(to_drop)] = False
    else:
        keep_corr_mask = np.ones(X_train_var.shape[1], dtype=bool)

    X_train_final = X_train_var[:, keep_corr_mask]
    X_test_final = X_test_var[:, keep_corr_mask]

    filter_info = {
        "n_features_after_filtering": X_train_final.shape[1]
    }
    print(f'amount of features after filtering: {X_train_final.shape[1]}')

    return X_train_final, X_test_final, y_train, y_test, filter_info

#%%
# Feature selection PCA
def feature_selection_PCA(X_train,X_test,y_train,y_test,n_components=20,var_threshold=0.01,corr_threshold=0.995):
    X_train_filt, X_test_filt, y_train, y_test, filter_info = feature_filtering(X_train, X_test, y_train, y_test,var_threshold=var_threshold,corr_threshold=corr_threshold)

    max_components = min(n_components, X_train_filt.shape[1], X_train_filt.shape[0])
    pca = decomposition.PCA(n_components=max_components) 
    X_train_sel = pca.fit_transform(X_train_filt)
    X_test_sel = pca.transform(X_test_filt)

    info = {
        "n_components_used": max_components
    }

    print(f'amount of features after selection: {X_train_sel.shape[1]}')

    return X_train_sel, X_test_sel, y_train, y_test, info

#%%
# feature selection RFE

def feature_selection_RFE(X_train,X_test,y_train,y_test,n_features_to_select=20,step=1,var_threshold=0.01,corr_threshold=0.995): 
    X_train_filt, X_test_filt, y_train, y_test, filter_info = feature_filtering(X_train, X_test, y_train, y_test,var_threshold=var_threshold,corr_threshold=corr_threshold)

    n_features_to_select = min(n_features_to_select, X_train_filt.shape[1])
    estimator = svm.SVC(kernel="linear")
    rfe = RFE(
        estimator=estimator,
        n_features_to_select=n_features_to_select,
        step=step)
    
    rfe.fit(X_train_filt, y_train)
    X_train_sel = rfe.transform(X_train_filt)
    X_test_sel = rfe.transform(X_test_filt)

    info = {
        "n_features_selected": np.sum(rfe.support_)
    }

    print(f'amount of features after selection: {X_train_sel.shape[1]}')
    return X_train_sel, X_test_sel, y_train, y_test, info

#%%
#
def feature_selection_L1(X_train, X_test, y_train, y_test, C=0.1, max_features=20,
                         var_threshold=0.01, corr_threshold=0.995):

    X_train_filt, X_test_filt, y_train, y_test, filter_info = feature_filtering(
        X_train, X_test, y_train, y_test,
        var_threshold=var_threshold, corr_threshold=corr_threshold
    )

    # Fix warnings 1 & 2: use l1_ratio instead of penalty="l1"
    l1_model = LogisticRegression(
        solver="saga",          # supports l1_ratio
        l1_ratio=1,             # 1 = pure L1, same behaviour as penalty="l1"
        C=C,
        random_state=42,
        max_iter=10000           # fix warning 3: increase iterations
    )

    # Fix: use max_features properly
    selector = SelectFromModel(
        estimator=l1_model,
        max_features=max_features,  # exactly 20 features
        threshold=-np.inf           # required to enforce max_features
    )

    selector.fit(X_train_filt, y_train)
    X_train_sel = selector.transform(X_train_filt)
    X_test_sel = selector.transform(X_test_filt)

    info = {
        "n_features_selected": np.sum(selector.get_support())
    }
    print(f'amount of features after filtering: {X_train_filt.shape[1]}')
    print(f'amount of features after selection: {X_train_sel.shape[1]}')

    return X_train_sel, X_test_sel, y_train, y_test, info

#%%
# univariate testing
def feature_selection_univariate(X_train,X_test,y_train,y_test,k=20,var_threshold=0.01,corr_threshold=0.995):
    X_train_filt, X_test_filt, y_train, y_test, filter_info = feature_filtering(X_train, X_test, y_train, y_test,var_threshold=var_threshold,corr_threshold=corr_threshold)

    k = min(k, X_train_filt.shape[1])

    selector = SelectKBest(score_func=f_classif, k=k)
    X_train_sel = selector.fit_transform(X_train_filt, y_train)
    X_test_sel = selector.transform(X_test_filt)

    info = {
        "n_features_selected": np.sum(selector.get_support())
    }
    print(f'amount of features after selection: {X_train_sel.shape[1]}')
    return X_train_sel, X_test_sel, y_train, y_test, info

#%%
import pandas as pd
import numpy as np                 # <--- Extra import
from scipy.stats import normaltest
from A_Load_data_cv import load_data

# 1. Data inladen
result = load_data()

# 2. Omzetten naar DataFrame
if isinstance(result, tuple):
    # Als het een tuple is, pak het eerste element (meestal de features X)
    data_array = result[0]
else:
    data_array = result

# Zet de NumPy array om naar een Pandas DataFrame
# We maken namen zoals 'Feature_0', 'Feature_1', etc.
df = pd.DataFrame(data_array, columns=[f'Feature_{i}' for i in range(data_array.shape[1])])

print(f"Data succesvol geladen. Aantal kolommen gevonden: {df.shape[1]}")

# 3. De functie voor normaliteit (ongewijzigd)
def check_normality(df, alpha=0.05):
    results = []
    
    # Nu werkt select_dtypes wel, want df is een DataFrame
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        data = df[col].dropna()
        if len(data) >= 8:
            stat, p = normaltest(data)
            is_normal = p > alpha
            results.append({
                'Feature': col,
                'P-waarde': round(p, 4),
                'Normaal verdeeld?': 'JA' if is_normal else 'NEE'
            })
    return pd.DataFrame(results)

# 4. Voer de test uit
normality_summary = check_normality(df)

# Toon de eerste 10 resultaten
print("\n--- Resultaten (Eerste 10 kolommen) ---")
print(normality_summary.head(10))

# Toon alleen de kolommen die NIET normaal zijn
niet_normaal = normality_summary[normality_summary['Normaal verdeeld?'] == 'NEE']
print(f"\nAantal features dat NIET normaal verdeeld is: {len(niet_normaal)}")