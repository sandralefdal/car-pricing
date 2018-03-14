import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd


def rescale_outliers(predictions, min_value, max_value):
    """
    Rescale prediction outliers where values in index i are such that predictions[i]>max_value or predictions[i]<min_value
    If prediction[i] < min_value then => prediction[i] = min_value
    If prediction[i] > max_value then => prediction[i] = max_value
    :param target: target values in test set
    :param min_value: min value in train target set
    :param max_value: max value in train target set
    :return: predictions without outliers
    """
    if len(predictions.shape) == 2:
        predictions = predictions.flatten()

    for i in range(len(predictions)):
        if predictions[i] > max_value:
            predictions[i] = max_value
        elif predictions[i] < min_value:
            predictions[i] = min_value

    return predictions


def data_split(percentSplit, df_features, df_target):
    """
    Split X and y into test and train
    :param percentSplit: persentage of data points to go into the training set
    :param df_features: feature set
    :param df_target: target vector
    :return: train_x, train_y, test_x, test_y
    """
    train_indexes = np.random.rand(len(df_features)) < percentSplit
    train_x, train_y = df_features[train_indexes], df_target[train_indexes]
    test_x, test_y = df_features[~train_indexes], df_target[~train_indexes]
    return train_x, train_y, test_x, test_y


def scale(df, scaler = None):
    """
    MinMax scale attributes in df.
    Values are scaled to be between 0 and 1
    :param df: data frame or vector to scale
    :param scaler: if scaler is passed in as argument, scaler is used. Otherwise a new scaler is created.
    :return: df_scaled, scaler
    """
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


def inverse_scale(df, scaler):
    """
    Inverse scale attributes that has been scaled
    :param df: df or vector to inverse scale
    :param scaler: scaler that was originally used to scale df
    :return: inverse scale of df
    """
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

