import pandas as pd
import numpy as np
from math import sqrt

import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score as r2
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor

import car_pricing.commons.helper as helper
import car_pricing.commons.models as models


class AdaBoost:
    """
    Ada Boost consists of a set of weak learners
     * linear regression
     * knn
     * decision tree

    Prediction is made by taking the weighted average of the weak learners output

    The weights of each weak learner are determined by using a genetic algorithm
    """
    def __init__(self, df, target_column):
        self.df = df
        self.target_column = target_column

        self.lr = None
        self.knn = None
        self.dt = None

    def save_to_file(self, path):
        self.df.to_csv(path)

    def build_model(self):
        train_x, train_y, test_x, test_y = helper.data_split(0.8,
                                                             self.df.loc[:,
                                                             [col for col in self.df.columns if col!=self.target_column]].values,
                                                             self.df.loc[:, self.target_column].values)


        train_x_scaled, x_scaler = helper.scale(train_x)
        test_x_scaled, _ = helper.scale(test_x, x_scaler)

        train_y_scaled, y_scaler = helper.scale(train_y)
        test_y_scaled, _ = helper.scale(test_y, y_scaler)

        predictions_lr, self.lr = models.linear_regression(train_x_scaled, train_y_scaled, test_x_scaled, test_y_scaled, y_scaler)

        predictions_knn, self.knn = models.knn(train_x_scaled, train_y_scaled, test_x_scaled, test_y_scaled, y_scaler)

        predictions_dt, self.dt = models.decision_tree(train_x, train_y, test_x, test_y)

        predictions_ada = [np.average([pred for pred in [predictions_lr[i],predictions_knn[i], predictions_dt[i]] if pred < max(train_y) and pred > min(train_y)]) for i in range(len(predictions_lr))]

        rmse = sqrt(mse(test_y, predictions_ada))

        return rmse
