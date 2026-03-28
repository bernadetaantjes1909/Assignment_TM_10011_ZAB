#%%
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
 
from scipy import stats
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
 
#%%  Load data
print("Loading data...")
 
this_directory = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(this_directory, "Liver_radiomicFeatures.csv")
 
data = pd.read_csv(file_path, index_col=0)
 
X = data.iloc[:, 2:]
y = data["label"]
 
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
 

scaler   = RobustScaler()
X_scaled = scaler.fit_transform(X)
 
selector  = VarianceThreshold()
X_cleaned = selector.fit_transform(X_scaled)
 
print(f"Samples  : {X_cleaned.shape[0]}")
print(f"Features : {X_cleaned.shape[1]}")
print(f"Classes  : {list(label_encoder.classes_)}")
 
# Split by class
classes     = label_encoder.classes_           # ['benign', 'malignant']
X_per_class = {cls: X_cleaned[y_encoded == i]
               for i, cls in enumerate(classes)}
 
#%%  Shapiro-Wilk normality test 

 
print("\n" + "="*60)
print("TEST 1 – Shapiro-Wilk normality test (per feature per class)")
print("="*60)
 
shapiro_results = {}
 
for cls, X_cls in X_per_class.items():
    n_features   = X_cls.shape[1]
    n_normal     = 0
    p_values     = []
 
    for f in range(n_features):
        stat, p = stats.shapiro(X_cls[:, f])
        p_values.append(p)
        if p >= 0.05:
            n_normal += 1
 
    pct_normal = 100 * n_normal / n_features
    shapiro_results[cls] = p_values
 
    print(f"\n  Class: {cls}")
    print(f"  Features passing normality (p >= 0.05) : "
          f"{n_normal} / {n_features} ({pct_normal:.1f}%)")
    print(f"  Features FAILING normality (p < 0.05)  : "
          f"{n_features - n_normal} / {n_features} "
          f"({100 - pct_normal:.1f}%)")
 
    if pct_normal < 50:
        print(f"  → CONCLUSION: Majority of features are not normally distributed.")
        print(f"    Gaussian assumption for LDA/QDA does not hold for class '{cls}'.")
    else:
        print(f"  → CONCLUSION: Majority of features appear normally distributed.")
        print(f"    Gaussian assumption may hold for class '{cls}'.")
# %%
