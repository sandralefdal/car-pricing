from math import sqrt

from sklearn import neighbors
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as mse
from sklearn.tree import DecisionTreeRegressor

from car_pricing.commons.helper import rescale_outliers

from car_pricing.commons.helper import *

INF_VALUE = 100000000


def mean_value(train, test, test_mode=False):
    """
    Return mean value of train as estimate for test
    :param train: training array of values
    :param test: test array of values
    :param test_mode: if test_mode, return RMSE
    :return: if not test mode, return list of mean values of len(test)
    """
    mean = average(train)

    if mean is None:
        mean = INF_VALUE

    if test_mode:
        return sqrt(mse(test, [mean for i in range(len(test))]))
    else:
        return np.array([mean] * len(test))


def linear_regression(train_x, train_y, test_x, test_y, scaler = None, lr=None, test_mode=False):
    """
    Build linear regression model
    :param train_x: training features
    :param train_y: training target
    :param test_x: test features
    :param test_y: test target
    :param scaler: target scaler, if argument is given, predictions and target are scaled before returned
    :param lr: trained linear regression model
    :param test_mode: if test_mode, RMSE is returned
    :return:
    """
    if lr is None:
        lr = LinearRegression()
        lr.fit(train_x, train_y)

    predictions = lr.predict(test_x).flatten()

    train_y_reverse_scaled = train_y
    if scaler is not None:
        train_y_reverse_scaled = inverse_scale(train_y, scaler)

    if test_mode:
        if scaler is not None:
            test_y = inverse_scale(test_y, scaler)
            predictions = inverse_scale(predictions, scaler)
        preds = rescale_outliers(predictions, np.argmin(train_y_reverse_scaled), np.argmax(train_y_reverse_scaled))
        return sqrt(mse(test_y, preds))
    else:
        if scaler is not None:
            predictions = inverse_scale(predictions, scaler)
        return predictions, lr


def knn(train_x, train_y, test_x, test_y, scaler = None, test_mode=False, n_neighbors = None, distance = None):
    """
    Build kNN model, or make predictions on kNN model if test mode and model given
    :param train_x: training features
    :param train_y: training target
    :param test_x: test features
    :param test_y: test target
    :param scaler: target scaler, if argument is given, predictions and target are scaled before returned
    :param test_mode: if test_mode =True, (smallest_error, k_neighbors, distance_weights) are returned
    :param n_neighbors: number of neighbors to use in model
    :param n_neighbors: distance metric to use in model
    :return: if not test_mode, predictions on test set and kNN model is returned
    """
    if n_neighbors is not None and distance is not None and not test_mode:
        knn = neighbors.KNeighborsRegressor(n_neighbors, weights=distance)
        knn.fit(train_x, train_y)
        preds = knn.predict(test_x).flatten()

        if scaler is not None:
            preds = inverse_scale(preds, scaler)

        return preds, knn

    smallest_error = INF_VALUE
    predictions = []
    knn = None

    if scaler is not None:
        test_y = inverse_scale(test_y, scaler)

    weights = 'distance'
    for k in range(2, 20):
        knn_local = neighbors.KNeighborsRegressor(k, weights=weights)
        knn_local.fit(train_x, train_y)

        preds = knn_local.predict(test_x).flatten()

        if scaler is not None:
            preds = inverse_scale(preds, scaler)

        error = sqrt(mse(test_y, preds))
        if error < smallest_error:
            smallest_error = error
            knn = knn_local
            predictions = preds

    if test_mode:
        return smallest_error, knn.n_neighbors, knn.weights
    else:
        return predictions, knn


def decision_tree(train_x, train_y, test_x, test_y, scaler = None, dt=None, test_mode=False):
    """
    Build regression decision tree model
    :param train_x: training features
    :param train_y: training target
    :param test_x: test features
    :param test_y: test target
    :param scaler: target scaler, if argument is given, predictions and target are scaled before returned
    :param dt: trained decision tree
    :param test_mode: if test_mode, RMSE is returned
    :return:
    """
    if dt is None:
        dt = DecisionTreeRegressor()
        dt.fit(train_x, train_y)

    predictions = dt.predict(test_x).flatten()

    train_y_reverse_scaled = train_y
    if scaler is not None:
        train_y_reverse_scaled = inverse_scale(train_y, scaler)

    if test_mode:
        if scaler is not None:
            test_y = inverse_scale(test_y, scaler)
            predictions = inverse_scale(predictions, scaler)
        preds = rescale_outliers(predictions, np.argmin(train_y_reverse_scaled), np.argmax(train_y_reverse_scaled))
        return sqrt(mse(test_y, preds))
    else:
        if scaler is not None:
            predictions = inverse_scale(predictions, scaler)
        return predictions, dt
