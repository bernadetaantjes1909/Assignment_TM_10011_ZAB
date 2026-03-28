#preprocessing contains the function:
#"preprocessing_data":This function splits the data in a training set and a test set. 
from sklearn import model_selection

def preprocessing_data(X,y):

    data_train, data_test, y_train, y_test = model_selection.train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    return data_train, data_test, y_train, y_test