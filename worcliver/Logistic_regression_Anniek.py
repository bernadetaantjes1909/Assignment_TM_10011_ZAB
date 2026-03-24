#%%
import os
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.exceptions import ConvergenceWarning


#%%
# Optional: suppress only convergence warnings from saga
warnings.filterwarnings("ignore", category=ConvergenceWarning)


#%%
# DATA LOADING
def load_data():
    print("\n[1/6] Loading data...")

    this_directory = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(this_directory, "Liver_radiomicFeatures.csv")

    data = pd.read_csv(file_path, index_col=0)

    print(f"      Loaded {data.shape[0]} samples and {data.shape[1]} columns.")
    return data


#%%
# PREPROCESSING
def preprocessing_data(load_data_func):
    print("\n[2/6] Preprocessing data...")

    data = load_data_func()

    # Features and labels
    X = data.iloc[:, 2:]
    y = data["label"]

    print(f"      Raw feature matrix shape: {X.shape}")
    print(f"      Unique labels found: {sorted(y.unique())}")

    # Encode benign/malignant to 0/1
    print("      Encoding labels to numeric values...")
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    class_mapping = {
        cls_name: int(idx) for idx, cls_name in enumerate(label_encoder.classes_)
    }
    print(f"      Label mapping: {class_mapping}")

    # Train-test split
    print("      Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.20,
        random_state=42,
        stratify=y_encoded
    )

    print(f"      Train shape: {X_train.shape}")
    print(f"      Test shape:  {X_test.shape}")

    # Scale
    print("      Scaling features with RobustScaler...")
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("      Preprocessing finished.")

    return X_train_scaled, X_test_scaled, y_train, y_test, label_encoder


#%%
# REMOVE ZERO-VARIANCE FEATURES
def deleting_zero_variance(X_train, X_test):
    print("\n[3/6] Removing zero-variance features...")

    selector = VarianceThreshold()
    X_train_filtered = selector.fit_transform(X_train)
    X_test_filtered = selector.transform(X_test)

    removed = X_train.shape[1] - X_train_filtered.shape[1]

    print(f"      Removed {removed} zero-variance features.")
    print(f"      Remaining features: {X_train_filtered.shape[1]}")

    return X_train_filtered, X_test_filtered


#%%
# LOGISTIC REGRESSION PIPELINE
def logistic_regression_classifier():
    print("\n================ LOGISTIC REGRESSION PIPELINE START ================")

    # -------------------------
    # Data preprocessing
    # -------------------------
    X_train, X_test, y_train, y_test, label_encoder = preprocessing_data(load_data)
    X_train, X_test = deleting_zero_variance(X_train, X_test)

    # -------------------------
    # Base model
    # -------------------------
    print("\n[4/6] Building Logistic Regression model...")
    print("      Using solver='saga' with penalty='l1'.")

    base_model = LogisticRegression(
        solver="saga",
        l1_ratio=1.0,
        max_iter=20000,
        tol=1e-4,
        random_state=42
    )



    # Hyperparameter search space
    param_dist = {
        "C": np.logspace(0, 5, 40)
    }

    print("      Hyperparameter search space for C:")
    print(f"      From {param_dist['C'][0]:.5f} to {param_dist['C'][-1]:.5f}")
    print(f"      Number of candidate values: {len(param_dist['C'])}")

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        n_iter=40,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
        random_state=42,
        refit=True,
        verbose=1
    )

    # -------------------------
    # Hyperparameter tuning
    # -------------------------
    print("\n[5/6] Searching for best hyperparameter C...")
    search.fit(X_train, y_train)

    best_C = search.best_params_["C"]

    print("      Hyperparameter search finished.")
    print(f"      Best C found: {best_C}")
    print(f"      Best cross-validation accuracy: {search.best_score_:.4f}")

    # -------------------------
    # Final model training
    # -------------------------
    print("      Training final model using best C...")

    final_model = LogisticRegression(
        solver="saga",
        l1_ratio=1.0,
        C=best_C,
        max_iter=20000,
        tol=1e-4,
        random_state=42
    )

    final_model.fit(X_train, y_train)

    # Count nonzero coefficients just to see sparsity
    nonzero_coef = np.sum(final_model.coef_ != 0)
    total_coef = final_model.coef_.size
    print(f"      Nonzero coefficients after L1 regularization: {nonzero_coef} / {total_coef}")

    # -------------------------
    # Threshold tuning
    # -------------------------
    print("\n[6/6] Finding best decision threshold...")

    prob_train = final_model.predict_proba(X_train)[:, 1]
    prob_test = final_model.predict_proba(X_test)[:, 1]

    thresholds = np.linspace(0.2, 0.8, 10)

    best_threshold = 0.5
    best_score = -np.inf

    for t in thresholds:
        preds_train_tmp = (prob_train >= t).astype(int)
        score = accuracy_score(y_train, preds_train_tmp)

        if score > best_score:
            best_score = score
            best_threshold = t

    print(f"      Best threshold found: {best_threshold:.4f}")
    print(f"      Training accuracy at best threshold: {best_score:.4f}")

    # Final predictions
    print("      Creating final train and test predictions...")
    y_pred_train = (prob_train >= best_threshold).astype(int)
    y_pred_test = (prob_test >= best_threshold).astype(int)

    # Performance
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)

    print("\n================ RESULTS ================")
    print(f"Train accuracy: {train_acc:.4f}")
    print(f"Test accuracy:  {test_acc:.4f}")

    train_misclassified = np.sum(y_train != y_pred_train)
    test_misclassified = np.sum(y_test != y_pred_test)

    print(f"Misclassified train: {train_misclassified} / {len(y_train)}")
    print(f"Misclassified test:  {test_misclassified} / {len(y_test)}")

    # Convert back to original class names
    y_train_labels = label_encoder.inverse_transform(y_train)
    y_test_labels = label_encoder.inverse_transform(y_test)
    y_pred_train_labels = label_encoder.inverse_transform(y_pred_train)
    y_pred_test_labels = label_encoder.inverse_transform(y_pred_test)

    print("\nClass check:")
    print(f"      Original classes: {list(label_encoder.classes_)}")

    print("\n================ PIPELINE FINISHED ================\n")

    return (
        best_C,
        best_threshold,
        final_model,
        y_pred_train,
        y_pred_test,
        y_train,
        y_test,
        y_pred_train_labels,
        y_pred_test_labels,
        y_train_labels,
        y_test_labels,
        label_encoder
    )


#%%
# RUN MODEL
if __name__ == "__main__":
    (
        best_C,
        best_threshold,
        model,
        train_pred,
        test_pred,
        y_train,
        y_test,
        train_pred_labels,
        test_pred_labels,
        y_train_labels,
        y_test_labels,
        label_encoder
    ) = logistic_regression_classifier()