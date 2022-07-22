import numpy as np
from sklearn.svm import SVC


def train_svm_and_predict(train_features, train_target, test_features):
    """
    train_features: np.array, (num_elements_train x num_features) - train data description, the same features and the same order as in train data
    train_target: np.array, (num_elements_train) - train data target
    test_features: np.array, (num_elements_test x num_features) -- some test data, features are in the same order as train features

    return: np.array, (num_elements_test) - test data predicted target, 1d array
    """
    best_c = 2.5714285714285716
    model = SVC(kernel='rbf', C=best_c)
    X_train = train_features
    X_test = test_features
    X_3344_train = np.concatenate([(X_train[:, 3] ** 2).reshape((-1, 1)), (X_train[:, 4] ** 2).reshape((-1, 1))], axis=1)
    X_3344_test = np.concatenate([(X_test[:, 3] ** 2).reshape((-1, 1)), (X_test[:, 4] ** 2).reshape((-1, 1))], axis=1)
    model.fit(X_3344_train, train_target)
    
    return model.predict(X_3344_test)
