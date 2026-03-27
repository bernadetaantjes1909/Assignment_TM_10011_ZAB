#%%
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay

#%%
# 1. Definieer de Feature Selector
feature_selector = SelectFromModel(
    LogisticRegression(solver='saga', penalty='l1', l1_ratio=1, random_state=42),
    max_features=20
)
# 2. De Classifiers Dictionary
# LET OP: De parameters hebben nu 'classifier__' ervoor staan!
classifiers = {
    "RF": {
        "clf": RandomForestClassifier(random_state=42),
        "params": {
            "n_estimators": [150, 200, 250],
            "max_depth": [6, 8, 10],
            "min_samples_split": [5, 7, 9],
            "min_samples_leaf": [2, 4, 6],
            "max_features": ["sqrt", "log2"]
        }
    }, 
    "kNN": {
        "clf": RandomForestClassifier(random_state=42),
        "params": {
            "n_neighbors": [15, 17, 19, 21, 23, 25, 27],
            "weights": ["uniform", "distance"],
            "metric": ["minkowski"],
            "p": [1, 2]
        }
    },
    "SVM": {
        "clf": RandomForestClassifier(random_state=42),
        "params": {
            "C": [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.01]
        }
    }
}

