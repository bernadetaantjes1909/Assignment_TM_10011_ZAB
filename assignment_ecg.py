#!/usr/bin/env python
# coding: utf-8

# # TM10011 Assignment template -- ECG data

# ## Data loading and cleaning
# 
# Below are functions to load the dataset of your choice. After that, it is all up to you to create and evaluate a classification method. Beware, there may be missing values in these datasets. Good luck!


import os
import pandas as pd

data = pd.read_csv('/content/tm10011_ml/ecg/ecg_data/ecg_data.csv', index_col=0)

print(f'The number of samples: {len(data.index)}')
print(f'The number of columns: {len(data.columns)}')