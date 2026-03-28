#Load_data contains the function:
#"Load_data": In this function the data is loaded and the column containing the classification is changed from 'benign'and 'malignant' to 0 and 1, respectively. 
# The function returns the features (X) and the labels (y).

import os
import pandas as pd
from sklearn.calibration import LabelEncoder

def load_data():
    this_directory = os.path.dirname(os.path.abspath(__file__))
    data = pd.read_csv(
        os.path.join(this_directory, 'Liver_radiomicFeatures.csv'),
        index_col=0
    )
    target_column = "label"

    X = data.drop(columns=[target_column]).values
    y = data[target_column].values

    le = LabelEncoder()
    y = le.fit_transform(y)   

    return X, y
