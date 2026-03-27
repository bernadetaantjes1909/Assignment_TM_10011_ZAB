#%%
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_classif, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier # Optioneel, werd in je voorbeeld genoemd
#%%

feature_selectors = {
    "PCA": {
        "selector": PCA(),
        "params": {
            "n_components": [5, 10, 20, 30, 50]
        }   
    },
    "RFE": {
        "selector": RFE(estimator=SVC(kernel="linear")),
        "params": {
            "n_features_to_select": [5, 10, 20, 30],
            "step": [1, 5]
        }
    },
    "Lasso_L1": {
        "selector": SelectFromModel(
            LogisticRegression(penalty="l1", solver="saga", max_iter=10000, random_state=42)
        ),
        "params": {
            "estimator__C": [0.01, 0.1, 1.0, 10.0],
            "max_features": [20] 
        }
    },
    "Univariate": {
        "selector": SelectKBest(score_func=f_classif),
        "params": {
            "k": [5, 10, 20, 30]
        }
    }
}
