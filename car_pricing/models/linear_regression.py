import pandas as pd
import car_pricing.commons.helper as helper
import car_pricing.commons.models as models


class LinearRegression:
    def __init__(self, df, target_column):
        self.df = df
        self.target_column = target_column
        self.lr = None

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

        mse = models.linear_regression(train_x_scaled, train_y_scaled, test_x_scaled, test_y_scaled, None, True)

        print(mse)