#Load_data_cv

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
