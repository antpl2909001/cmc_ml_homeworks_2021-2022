import numpy as np

class MinMaxScaler:
    min_v = np.nan
    max_v = np.nan
    def fit(self, data):
        """Store calculated statistics

        Parameters:
        data (np.array): train set, size (num_obj, num_features)
        """
        self.min_v = data.min(axis=0)
        self.max_v = data.max(axis=0)
        
        
    def transform(self, data):
        """
        Parameters:
        data (np.array): train set, size (num_obj, num_features)

        Return:
        np.array: scaled data, size (num_obj, num_features)
        """
        return (data - self.min_v) / (self.max_v - self.min_v)


class StandardScaler:
    mean_v = np.nan
    sd_v = np.nan
    
    def fit(self, data):
        """Store calculated statistics
        
        Parameters:
        data (np.array): train set, size (num_obj, num_features)
        """
        self.mean_v = data.mean(axis=0)
        disp_v = ((data - self.mean_v) ** 2).sum(axis=0) / data.shape[0]
        self.sd_v = disp_v ** 0.5
        
    def transform(self, data):
        """
        Parameters:
        data (np.array): train set, size (num_obj, num_features)

        Return:
        np.array: scaled data, size (num_obj, num_features)
        """
        return (data - self.mean_v) / self.sd_v
