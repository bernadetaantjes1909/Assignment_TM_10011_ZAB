
#Check normality contains the function:
#"check_normality": To check which features were normally distributed.

import pandas as pd
import numpy as np
import A_Load_data_cv
from scipy.stats import normaltest
from A_Load_data_cv import load_data

#%%
X,y = load_data()

X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])

def check_normality(dataframe, alpha=0.05):
    results = []
    numeric_cols = dataframe.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        data = dataframe[col].dropna()
        if len(data) >= 8:
            stat, p_value = normaltest(data)
            is_normal = p_value > alpha

            results.append({
                "feature": col,
                "p_value": round(p_value, 4),
                "is_normal": is_normal
            })

    return pd.DataFrame(results)


normality = check_normality(X)

non_normal = normality[normality["is_normal"] == False]

print(f"\nNumber of non-normal features: {len(non_normal)}")