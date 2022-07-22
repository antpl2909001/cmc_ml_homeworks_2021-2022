import numpy as np


class Preprocesser:

    def __init__(self):
        pass
    
    def fit(self, X, Y=None):
        pass
    
    def transform(self, X):
        pass
    
    def fit_transform(self, X, Y=None):
        pass
    
    
class MyOneHotEncoder(Preprocesser):
    
    def __init__(self, dtype=np.float64):
        super(Preprocesser).__init__()
        self.feature_list = []
        self.dtype = dtype
        
    def fit(self, X, Y=None):
        """
        param X: training objects, pandas-dataframe, shape [n_objects, n_features]
        param Y: unused
        """
        # необходимо вынести некоторые переменные в члены класса (возможно Preprocesser)
        # информация, необходимая для transform:
        # - названия всех признаков, упорядоченных по количеству уникальных значений
        # - для каждого признака - список уникальных значений, отсортированных в порядке убывания (видимо он задан по умолчанию)
        
        # список кортежей вида (feature_name, unique_values_num, unique_values_array)
        for feature in X.columns:
            unique_values_array = np.unique(X[feature])
            unique_values_num = unique_values_array.shape[0]
            
            self.feature_list.append((feature, unique_values_num, unique_values_array))
        
        # self.feature_list.sort(key=lambda x: x[1])
    
    def transform(self, X):
        """
        param X: objects to transform, pandas-dataframe, shape [n_objects, n_features]
        returns: transformed objects, numpy-array, shape [n_objects, |f1| + |f2| + ...]
        """
        n_obj = X.shape[0]
        
        # список массивов формы n_obj * |f_i|, где |f_i| - количество уникальных значений i-ого признака
        array_list = []
        
        for name, num, values in self.feature_list:
            # если признака с именем name нет в Х, пропускаем его
            # if name not in X.columns:
            #     print('Warning: there is no column with name {name} in X')
            #    continue
            
            # результат one-hot-encoding для одного признака
            arr = np.zeros((n_obj, num), dtype=self.dtype)
            
            # for obj_num in range(X[name].shape[0]):
            #     index = np.where(values == X.loc[obj_num, name])[0][0]
            #     arr[obj_num, index] = 1
            
            for i in range(num):
                arr[:, i][np.where(X[name] == values[i])] = 1
                
            array_list.append(arr)
        return np.concatenate(array_list, axis=1)
    
    def fit_transform(self, X, Y=None):
        self.fit(X)
        return self.transform(X)
    
    def get_params(self, deep=True):
        return {"dtype": self.dtype}

    
class SimpleCounterEncoder:
    
    def __init__(self, dtype=np.float64):
        self.feature_dict = dict()
        self.dtype=dtype
        
    def fit(self, X, Y):
        """
        param X: training objects, pandas-dataframe, shape [n_objects, n_features]
        param Y: target for training objects, pandas-series, shape [n_objects,]
        """
        
        for feature in X.columns:
            uniq_vals = np.unique(X[feature])
            stat_list = []
            
            for val in uniq_vals:
                counters = (X[feature] == val).sum() / X[feature].shape[0]
                successes = Y[X[feature] == val].mean()
                stat_list.append((val, successes, counters))
            
            self.feature_dict[feature] = stat_list
        
            
    def transform(self, X, a=1e-5, b=1e-5):
        """
        param X: objects to transform, pandas-dataframe, shape [n_objects, n_features]
        param a: constant for counters, float
        param b: constant for counters, float
        returns: transformed objects, numpy-array, shape [n_objects, 3]
        """
        array_list = []
        
        for feature in X.columns:
            arr = np.zeros((X.shape[0], 3), dtype=self.dtype)
            
            for val, successes, counters in self.feature_dict[feature]:
                arr[:, 0][np.where(X[feature] == val)] = successes
                arr[:, 1][np.where(X[feature] == val)] = counters
                arr[:, 2][np.where(X[feature] == val)] = (successes + a) / (counters + b)
                
            array_list.append(arr)
        
        return np.concatenate(array_list, axis=1)
    
    def fit_transform(self, X, Y, a=1e-5, b=1e-5):
        self.fit(X, Y)
        return self.transform(X, a, b)
    
    def get_params(self, deep=True):
        return {"dtype": self.dtype}

    
def group_k_fold(size, n_splits=3, seed=1):
    idx = np.arange(size)
    np.random.seed(seed)
    idx = np.random.permutation(idx)
    n_ = size // n_splits
    for i in range(n_splits - 1):
        yield idx[i * n_ : (i + 1) * n_], np.hstack((idx[:i * n_], idx[(i + 1) * n_:]))
    yield idx[(n_splits - 1) * n_ :], idx[:(n_splits - 1) * n_]

    
class FoldCounters:
    
    def __init__(self, n_folds=3, dtype=np.float64):
        self.dtype = dtype
        self.n_folds = n_folds
        self.feature_dict = dict()

    def fit(self, X, Y, seed=1):
        """
        param X: training objects, pandas-dataframe, shape [n_objects, n_features]
        param Y: target for training objects, pandas-series, shape [n_objects,]
        param seed: random seed, int
        """
        folds = list(group_k_fold(X.shape[0], n_splits=self.n_folds, seed=seed))
        
        for feature in X.columns:
            array_list = []
            uniq_vals = np.unique(X[feature])
            x = X[feature].to_numpy()
            
            for target_fold, train_folds in folds:
                arr = np.hstack([target_fold.reshape((-1, 1)), np.ones((target_fold.shape[0], 3))])
                
                for val in uniq_vals:
                    # successes
                    arr[np.where(x[target_fold] == val), 1] = Y.iloc[train_folds][x[train_folds] == val].mean()
                    # arr[np.where(x[target_fold] == val), 1] = Y[train_folds][x[train_folds] == val].mean()
                    # counters
                    arr[np.where(x[target_fold] == val), 2] = (x[train_folds] == val).sum() / x[train_folds].shape[0]
                array_list.append(arr)
            
            full_encode_array = np.concatenate(array_list, axis=0)
            self.feature_dict[feature] = full_encode_array[full_encode_array[:, 0].argsort(axis=0), 1:]
            
    def transform(self, X, a=1e-5, b=1e-5):
        """
        param X: objects to transform, pandas-dataframe, shape [n_objects, n_features]
        param a: constant for counters, float
        param b: constant for counters, float
        returns: transformed objects, numpy-array, shape [n_objects, 3]
        """
        for feature, arr in self.feature_dict.items():
            arr[:, 2] = (arr[:, 0] + a) / (arr[:, 1] + b)
        
        array_list = []
        for feature in X.columns:
            array_list.append(self.feature_dict[feature])
        
        return np.concatenate(array_list, axis=1)
        
    def fit_transform(self, X, Y, a=1e-5, b=1e-5):
        self.fit(X, Y)
        return self.transform(X, a, b)
 
       
def weights(x, y):
    """
    param x: training set of one feature, numpy-array, shape [n_objects,]
    param y: target for training objects, numpy-array, shape [n_objects,]
    returns: optimal weights, numpy-array, shape [|x unique values|,]
    """
    # размер выборки
    n = x.shape[0]
    
    # Уникальные значения {a_1, ..., a_m} признака x
    unique_values = np.unique(x)
    m = unique_values.shape[0]
    
    # P(y_i == 1)
    y_positive_prob = y.sum() / n
    
    # массив вероятностей P(X == a_j)
    x_prob = np.zeros((m, ))
    for i in range(m):
        x_prob[i] = (x == unique_values[i]).sum() / n
        
    # массив условных вероятностей P(X_i == a_j | y_i == 1)
    x_cond_prob = np.zeros((m, ))
    for i in range(m):
        x_cond_prob[i] = (x[y == 1] == unique_values[i]).sum() / (y == 1).sum()
        
    return y_positive_prob * x_cond_prob / x_prob