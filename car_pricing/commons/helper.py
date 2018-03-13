import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd


def data_split(percentSplit, df_features, df_target):
    train_indexes = np.random.rand(len(df_features)) < percentSplit
    train_x, train_y = df_features[train_indexes], df_target[train_indexes]
    test_x, test_y = df_features[~train_indexes], df_target[~train_indexes]
    return train_x, train_y, test_x, test_y


def scale(df, scaler = None):
    if isinstance(df, np.ndarray) and len(df.shape) == 1:
        df = df.reshape(-1, 1)

    if scaler is None:
        scaler = MinMaxScaler()
        scaler.fit(df)

    if isinstance(df, pd.DataFrame):
        df_scaled = pd.DataFrame(scaler.transform(df), columns=df.columns)
    else:
        df_scaled = scaler.transform(df)

    return df_scaled, scaler


def reverse_scale(df, scaler):
    if len(df.shape) == 1:
        df = df.reshape(-1, 1)

    return scaler.inverse_transform(df)


def average(arr):
    try:
        return np.average(arr)
    except ZeroDivisionError:
        print("Could not compute average on %s, ZeroDivisionError caught." % str(arr))
        return None
    except TypeError:
        print("Could not compute average on %s of size %s, TypeError caught." % (str(arr), str(arr.size)))
        return None

