# TM10011_PROJECT
The different steps of the machine learning algorithm are split in different documents."Load_data", "preprocessing_data", "feature_selection_data" and "classifiers" all contain functions that will be called on by functions in the following steps or in "Main file".

preprocessing contains the function:
"preprocessing_data":This function scales the data and splits the data in a training set and a test set. 

Feature_selection_data contains the functions:
"deleting_zero_variance": This function deletes features with a variance of below 0.01 in the preprocessed data. 
"feature_selection_RFE": This function uses this scaled and filtered data and performes recursive feature elimination.
"feature_selection_PCA": This function uses this scaled and filtered data and performes principle component analysis 

Classifiers contains the functions: 
"random_forest_classifier": This function uses the selected features from one of the feature selection methods and performes hyperparameter tuning within a random forest classifier
"logistic_regression_classifier": This function uses the selected features from one of the feature selection methods and performes hyperparameter tuning within a logistic regression classifier

Main file:
In the main file, the different feature selection methods are combined with the different classifiers and these combinations are tested on the test data to see which has the highest accuracy. 
