import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
np.random.seed(42)

def _drop_by_id_(df, column, ids):
    """
      Function to replace values in column with NaNs by indeces
       - df: pd.DataFrame - original dataframe
       - column: str - name of column to drop values
       - ids: list or array - indeces to drop out
    """
    modified_df = df.copy()
    modified_df.loc[ids, column] = np.nan
    return modified_df

def MCAR(df, columns, percentages):
    """
      Function to implement Missing Completely at Random type of missingness.
      In this case P(NaN) doesn't depend on any values in dataset.
      - df: pd.DataFrame - original dataframe
      - columns: List[str] - name of columns to drop values
      - percentages: List[float] - percentages of missing values in each column

      Returns:
         modified_df - pd.DataFrame - dataframe with replaced values
    """
    modified_df = df.copy()
    for i, col in enumerate(columns):
        n_drop = int(percentages[i] * len(df))
        drop_indices = np.random.choice(df[col].index, size=n_drop, replace=False)
        modified_df = _drop_by_id_(modified_df, col, drop_indices)
    return modified_df

def MNAR(df, columns, percentages, mode='greater'):
    """
      Function to implement Missing Not at Random type of missingness.
      In this case P(NaN_xi) = P(NaN_xi|xi) where xi is some particular column.
      - df: pd.DataFrame - original dataframe
      - columns: List[str] - name of columns to drop values
      - percentages: List[float] - percentages of missing values in each column
      - mode: "greater" or "less" - if "greater", probability of NaN is greater when value of Xi is greater. If "less", vise versa

      Returns:
         modified_df - pd.DataFrame - dataframe with replaced values
    """
    assert mode in ['greater', 'less'], 'Can be only "greater" or "less"'
    sc = MinMaxScaler((0, 1))
    modified_df = df.copy()
    for i, col in enumerate(columns):
        if mode == 'greater':
            probas = sc.fit_transform(df[col].values.reshape(-1, 1))
            probas /= np.sum(probas)
        else:
            probas = sc.fit_transform(-df[col].values.reshape(-1, 1))
            probas /= np.sum(probas)            
        n_drop = int(percentages[i] * len(df))
        drop_indices = np.random.choice(df[col].index, size=n_drop, p=probas.reshape(-1), replace=False)
        modified_df = _drop_by_id_(modified_df, col, drop_indices)
    return modified_df

def MAR(df, columns, percentages, mode='greater'):
    """
      Function to implement Missing Not at Random type of missingness.
      In this case P(NaN_xi) = P(NaN_xi|x1 ... x(i-1), x(i+1) ... xn) where xi is some particular column.
      - df: pd.DataFrame - original dataframe
      - columns: List[str] - name of columns to drop values
      - percentages: List[float] - percentages of missing values in each column
      - mode: "greater" or "less" - if "greater", probability of NaN is greater when mean value of X1 ... x(i-1) x(i+1) ... xn is greater. If "less", vise versa

      Returns:
         modified_df - pd.DataFrame - dataframe with replaced values
    """
    assert mode in ['greater', 'less'], 'Can be only "greater" or "less"'
    sc = MinMaxScaler((0, 1))
    modified_df = df.copy()

    for i, col in enumerate(columns):
        independent_columns = list(set(df.columns) - set([col]))
        if mode == 'greater':
            probas = sc.fit_transform(df[independent_columns]).mean(axis=1)
            probas /= np.sum(probas)
        else:
            probas = sc.fit_transform(-df[independent_columns]).mean(axis=1)
            probas /= np.sum(probas)

        n_drop = int(percentages[i] * len(df))
        drop_indices = np.random.choice(df[col].index, size=n_drop, p=probas.reshape(-1), replace=False)
        modified_df = _drop_by_id_(modified_df, col, drop_indices)
    return modified_df
