import numpy as np
from sklearn import neighbors
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as mse

from car_pricing.commons.helper import *

INF_VALUE = 100000000


def mean_value(train, test, test_mode=False):
    """

    :param train:
    :param test:
    :param test_mode:
    :return:
    """
    mean = average(train)

    if mean is None:
        mean = INF_VALUE

    if test_mode:
        return mse(test, [mean for i in range(len(test))])
    else:
        return np.array([mean] * len(test))


def linear_regression(train_x, train_y, test_x, test_y, lr=None, test_mode=False):
    """

    :param lr:
    :param train_x:
    :param train_y:
    :param test_x:
    :param test_y:
    :param test_mode:
    :return:
    """
    if lr is None:
        lr = LinearRegression()
        lr.fit(train_x, train_y)

    predictions = lr.predict(test_x).flatten()

    if test_mode:
        return mse(test_y, predictions)
    else:
        return predictions, lr


def similar_items(train_x, train_y, test_x, test_y, test_mode=False, n_neighbors = None, distance = None):
    """

    :param train_x:
    :param train_y:
    :param test_x:
    :param test_y:
    :param test_mode:
    :return:
    """
    if n_neighbors is not None and distance is not None and not test_mode:
        knn = neighbors.KNeighborsRegressor(n_neighbors, weights=distance)
        knn.fit(train_x, train_y)
        return knn.predict(test_x).flatten(), knn

    smallest_error = INF_VALUE
    predictions = []
    knn = None
    for n in range(2, 20):
        for weights in ['uniform', 'distance']:
            knn_local = neighbors.KNeighborsRegressor(n, weights=weights)
            knn_local.fit(train_x, train_y)

            preds = knn_local.predict(test_x).flatten()

            error = mse(test_y, preds)
            if error < smallest_error:
                smallest_error = error
                knn = knn_local
                predictions = preds

    if test_mode:
        return smallest_error, knn.n_neighbors, knn.weights
    else:
        return predictions, knn
