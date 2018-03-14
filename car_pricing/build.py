import pandas as pd
import numpy as np
import os

from car_pricing.feature_engineering.feature_engineer import FeatureEngineer
from car_pricing.models.linear_regression import LinearRegression
from car_pricing.models.ada_boost import AdaBoost

inputfile = "data/caddy_jz.csv"
feature_engineer_output = "out/df_fe.csv"

df = pd.read_csv(inputfile)

#  Feature Engineer / Impute missing values
print("Imputing missing values in feature space using optimal of Mean, Linear Regression and kNN imputation per column")
fe = FeatureEngineer(df, 'ObjectPrice')
fe.impute_all_nan_columns()
fe.save_to_file(feature_engineer_output)

# Build Linear Regression
df = pd.read_csv(feature_engineer_output)
lr = LinearRegression(df, 'ObjectPrice')

print("Building 10 Linear regression models.. ")
rmse_lr = []
for i in range(10):
    rmse_lr += [lr.build_model()]

print("Average RMSE for linear regression models: " + str(np.average(rmse_lr)))

# Build Ada-Boost Model
ada = AdaBoost(df, 'ObjectPrice')
print("Building 10 Ada boost regression models.. ")
rmse_ada = []
for i in range(10):
    rmse_ada += [ada.build_model()]

print("Average RMSE for Ada regression models: " + str(np.average(rmse_ada)))




