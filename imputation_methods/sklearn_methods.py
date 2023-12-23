import sklearn
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer

available_imputers = {'knn': KNNImputer, 
                      'simple': SimpleImputer, 
                      'iterative': IterativeImputer}

def get_sklearn_imputer(method, **kwargs):
    """
      Method to get imputer from sklearn
    """
    if method not in available_imputers.keys():
      raise ValueError("Invalid method. Please use one of 'knn', 'simple', or 'iterative'.")
    return available_imputers[method](**kwargs)
