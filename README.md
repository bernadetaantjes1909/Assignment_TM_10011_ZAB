# TM10011_PROJECT
The different steps of the machine learning algorithm are split in different documents."Load_data", "preprocessing_data", "feature_selection_data" and "classifiers" all contain functions that will be called on by functions in the following steps or in "Main file".
When runnning "Main file" all results will be shown. 

Load_data contains the function:
"Load_data": In this function the data is loaded and the column containing the classification is changed from 'benign'and 'malignant' to 0 and 1, respectively. The function returns the features (X) and the labels (y).

Check normality contains the function:
"check_normality": To check which features were normally distributed.

preprocessing contains the function:
"preprocessing_data":This function splits the data in a training set and a test set. 

Feature_selection_data contains the functions:
"feature_filtering": This function deletes features with a variance of below 0.01 or a correlation of above 0.995.
"feature_selection_RFE": This function performes recursive feature elimination.
"feature_selection_PCA": This function performes principle component analysis 
"feature_selection_L1": This function performes L1-Regularised Logistic Regression 

Classifiers contains the functions: 
"random_forest_classifier": This function uses the selected features from one of the feature selection methods and performes a gridsearh on a hyperparameter grid within a random forest classifier
"kNN_classifier": This function uses the selected features from one of the feature selection methods and performes a gridsearh on a hyperparameter grid within a kNN classifier
"SVM_classifier": This function uses the selected features from one of the feature selection methods and performes a gridsearh on a hyperparameter grid within a SVM classifier


Main file:
In the main file, the different feature selection methods are combined with the different classifiers and these combinations are tested on the test data to see which has the highest accuracy. This file contains:

--> importing functions and other files
--> Imports data via load_data and splits data via "preprocessing_data
--> Determines the hyperparameter grid that is used for the grid search (this is manually changed in the classifier functions, this is already done!)
--> "evaluate_combination" combines the feature_selection_methods with the classifiers, performes a cross validation
--> "final_evaluation" performes the final evaluation of the best performing model on the testset
--> "plot_learning_curve" plots a learning curve for this final model 

