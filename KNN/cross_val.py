import numpy as np
from collections import defaultdict

def kfold_split(num_objects, num_folds):
    """Split [0, 1, ..., num_objects - 1] into equal num_folds folds (last fold can be longer) and returns num_folds train-val 
       pairs of indexes.

    Parameters:
    num_objects (int): number of objects in train set
    num_folds (int): number of folds for cross-validation split

    Returns:
    list((tuple(np.array, np.array))): list of length num_folds, where i-th element of list contains tuple of 2 numpy arrays,
                                       the 1st numpy array contains all indexes without i-th fold while the 2nd one contains
                                       i-th fold
    """
    indexes = np.arange(num_objects)
    delims = np.arange(0, num_objects // num_folds * num_folds, num_objects // num_folds)[1:]
    folds = np.split(indexes, delims)
    
    l = []
    for i in range(num_folds):
        l.append((np.delete(indexes, folds[i]), folds[i]))

    return l


def knn_cv_score(X, y, parameters, score_function, folds, knn_class):
    """Takes train data, counts cross-validation score over grid of parameters (all possible parameters combinations) 

    Parameters:
    X (2d np.array): train set
    y (1d np.array): train labels
    parameters (dict): dict with keys from {n_neighbors, metrics, weights, normalizers}, values of type list,
                       parameters['normalizers'] contains tuples (normalizer, normalizer_name), see parameters
                       example in your jupyter notebook
    score_function (callable): function with input (y_predict, y_true) which outputs score metric
    folds (list): output of kfold_split
    knn_class (obj): class of knn model to fit

    Returns:
    dict: key - tuple of (normalizer_name, n_neighbors, metric, weight), value - mean score over all folds
    """
    
    parameters = defaultdict(lambda: [None], parameters)

    models_score = dict()
    
    for t in parameters['normalizers']:
        X_scaled = X
        scal_name = 'None'
        
        if t != None:
            scaler, scal_name = t
            if scaler != None:
                scaler.fit(X)
                X_scaled = scaler.transform(X)
        
        for k in parameters['n_neighbors']:
            model_params = dict()

            if k != None:
                model_params['n_neighbors'] = k
            
            for m in parameters['metrics']:
                if m != None:
                    model_params['metric'] = m
                
                for w in parameters['weights']:
                    if w != None:
                        model_params['weights'] = w

                    knn_model = knn_class()
                    knn_model.set_params(**model_params)
                    
                    folds_score = []
                    
                    for fold in folds:
                        train_X, test_X = X_scaled[fold[0]], X_scaled[fold[1]]
                        train_y, test_y = y[fold[0]], y[fold[1]]
                        
                        knn_model.fit(train_X, train_y)
                        pred_y = knn_model.predict(test_X)
                        folds_score.append(score_function(test_y, pred_y))

                    models_score[(scal_name, 
                                  knn_model.get_params()['n_neighbors'], 
                                  knn_model.get_params()['metric'], 
                                  knn_model.get_params()['weights'])] = np.mean(folds_score)
    return models_score
