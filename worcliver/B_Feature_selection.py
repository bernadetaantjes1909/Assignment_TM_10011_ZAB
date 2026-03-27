#%%
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_classif, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
#%%

feature_selectors = {
    "PCA": {
        "selector": PCA(),
        "params": {
            "n_components": 20
        }   
    },
    "RFE": {
        "selector": RFE(estimator=SVC(kernel="linear")),
        "params": {
            "n_features_to_select": 20,
            "step": 1
        }
    },
    "Lasso_L1": {
        "selector": SelectFromModel(
            LogisticRegression(penalty="l1", solver="saga", max_iter=10000, random_state=42)
        ),
        "params": {
            "estimator__C": 0.1,
            "max_features": 20 
        }
    },
}
