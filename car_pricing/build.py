import pandas as pd
import os

from car_pricing.feature_engineering.feature_engineer import FeatureEngineer
from car_pricing.models.linear_regression import LinearRegression

inputfile = "data/caddy_jz.csv"
feature_engineer_output = "out/df_fe.csv"

df = pd.read_csv(inputfile)
log_file = open('out/log', 'w')

#  Feature Engineer / Impute missing values
fe = FeatureEngineer(df, log_file, 'ObjectPrice')
fe.impute_all_nan_columns()
fe.save_to_file(feature_engineer_output)

# Build Linear Regression
df = pd.read_csv(feature_engineer_output)
lr = LinearRegression(df, 'ObjectPrice')
lr.build_model()




