import numpy as np

import car_pricing.commons.helper as helper
import car_pricing.commons.models as models
from statistics import mode


class FeatureEngineer:
    def __init__(self, df, target_column):
        self.df = df
        self.hasNan = self.df.columns[self.df.isna().any()].tolist()
        self.target_column = target_column

        self.lr = None
        self.knn = None

    def save_to_file(self, path):
        self.df.to_csv(path)

    def impute_all_nan_columns(self):
        """
        Impute missing values on all columns that contain NaN values
        :return: boolean
        """
        for column in self.hasNan:
            self.impute_missing_values(column, 11)

        return True

    def impute_missing_values(self, column, iterations):
        errors_mean = []
        errors_lr = []
        errors_knn = []
        knn_neighbors = []
        knn_distance = []

        # remove rows with NaN values in column to impute
        df_training = self.df.loc[np.isfinite(self.df[column]), [col for col in self.df.columns if col not in self.hasNan+[self.target_column]]+[column]]

        # Scale values in each feature to be between 0 and 1
        # df_training, scaler = helper.scale(df_training)

        train_columns = [col for col in df_training.columns if col != column]

        for i in range(iterations):
            train_x, train_y, test_x, test_y = helper.data_split(0.8,
                                                                 df_training.loc[:, train_columns].values,
                                                                 df_training.loc[:, column].values)

            train_x_scaled, scaler_x = helper.scale(train_x)
            test_x_scaled, _ = helper.scale(test_x, scaler_x)

            train_y_scaled, scaler_target_col = helper.scale(train_y)
            test_y_scaled, _ = helper.scale(test_y, scaler_target_col)

            errors_mean += [models.mean_value(train_y.flatten(), test_y.flatten(), True)]
            errors_lr += [models.linear_regression(train_x_scaled, train_y_scaled, test_x_scaled, test_y_scaled, scaler_target_col, None, True)]

            error_knn, k_neighbors, distance = models.knn(train_x_scaled, train_y_scaled, test_x_scaled, test_y_scaled, scaler_target_col, True)
            knn_neighbors += [k_neighbors]
            knn_distance += [distance]

            errors_knn += [error_knn]

        mean_avg_error = helper.average(errors_mean)
        lr_avg_error = helper.average(errors_lr)
        knn_avg_error = helper.average(errors_knn)

        train_x_full = self.df.loc[np.isfinite(self.df[column]), train_columns]
        train_x_full, scaler_train_cols = helper.scale(train_x_full)
        test_x_full = self.df.loc[self.df[column].isnull(), train_columns]
        test_x_full, _ = helper.scale(test_x_full, scaler_train_cols)

        train_y_full = self.df.loc[np.isfinite(self.df[column]), [column]]
        train_y_full, scaler_target_col = helper.scale(train_y_full)

        if mean_avg_error <= lr_avg_error and mean_avg_error <= knn_avg_error:
            self.log_model(column, 'mean', mean_avg_error)
            mean_vals_scaled = models.mean_value(train_y_full.values, ['0' for i in range(len(test_x_full))], False)
            self.df.loc[self.df[column].isnull(), column] = helper.inverse_scale(mean_vals_scaled, scaler_target_col).flatten()
        elif lr_avg_error <= knn_avg_error:
            lr_preds, self.lr = models.linear_regression(train_x_full, train_y_full, test_x_full, None, scaler_target_col, None, False)
            self.df.loc[self.df[column].isnull(), column] = lr_preds.flatten()
            self.log_model(column, 'regressor', lr_avg_error)

        else:
            knn_preds, self.knn = models.knn(train_x_full, train_y_full, test_x_full,
                                                   None, scaler_target_col, False, int(np.average(knn_neighbors)),
                                                   mode(knn_distance))
            self.df.loc[self.df[column].isnull(), column] = knn_preds.flatten()
            print("Column %s imputed with knn value (%d neighbors, %s distance metric). RMSE: %f \n"
                                % (column, self.knn.n_neighbors, self.knn.weights, knn_avg_error))

    def log_model(self, column, method, mean_avg_error):
        print("Column %s imputed with %s value. RMSE: %f \n" % (column, method, mean_avg_error))
