import pandas as pd
import car_pricing.commons.helper as helper
import car_pricing.commons.models as models


class HybridModel:
    """
    kNN. Distance metric: weighted distance.
    weights random grid search.

    Cluster similar items for kNN, for faster lookup.

    In production: Save centroids
    Check against centroids.
    find closest in cluster (with distance metric)
    Average value from them

    Find relationship between predicted and real as valued by distance
    Create confidence interval
    """
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

        ## Build model
