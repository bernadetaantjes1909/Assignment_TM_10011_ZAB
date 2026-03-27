import numpy as np


def feature_filtering(X_train, X_test, y_train, y_test, var_threshold=0.01, corr_threshold=0.995):
# remove zero variance
    variances = np.var(X_train, axis=0)
    keep_var_mask = variances >= var_threshold

    X_train_var = X_train[:, keep_var_mask]
    X_test_var = X_test[:, keep_var_mask]
# remove high correlation
    if X_train_var.shape[1] > 1:
        corr_matrix = np.abs(np.corrcoef(X_train_var, rowvar=False))
        to_drop = set()
        for i in range(corr_matrix.shape[1]):
            for j in range(i + 1, corr_matrix.shape[1]):
                if corr_matrix[i, j] > corr_threshold:
                    to_drop.add(j)
        keep_corr_mask = np.ones(X_train_var.shape[1], dtype=bool)
        if len(to_drop) > 0:
            keep_corr_mask[list(to_drop)] = False
    else:
        keep_corr_mask = np.ones(X_train_var.shape[1], dtype=bool)

    X_train_final = X_train_var[:, keep_corr_mask]
    X_test_final = X_test_var[:, keep_corr_mask]

    filter_info = {
        "n_features_after_filtering": X_train_final.shape[1]
    }
    print(f'amount of features after filtering: {X_train_final.shape[1]}')

    return X_train_final, X_test_final, y_train, y_test
