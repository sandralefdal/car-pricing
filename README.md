# Car Pricing

Project to predict car prices.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. 

### Prerequisites

You will need Python 3 to run the project. To check which python is on your machine, run in the command line:

```
python --version
```

### Installing

First, you should create a virtualenvironment to install required dependencies.

To create a virtualenvironment with name {name_of_env}, make sure you have virtualenv installed, and run:

```
virtualenv {name_of_env}
```

Now, activate the virtual environment

```
source {name_of_env}/bin/activate
```
Make sure you are in the car-pricing directory, otherwise cd into it.


Install the necessary dependencies
```
pip install -r requirements.txt
```

Steps of operations in module:
 * Feature scaling (All features are scaled to values between 0 and 1 for models that require scaling)
 * Feature Engineering (NaN values are imputed with a predicted value)
 * Model building and assesment


Feature Engineering on input data: \
For each feature that contain NaN values, the best predictor for the given feature is used to impute the missing values.
Predictors are chosen from best performers on: 
 * Mean value of non-NaN values
 * Regressor value built on the other features in the feature space
 * k-Nearest Neighbor determined by the other features in the feature space
 
Best performers are determined by minimising RMSE on 11-fold cross validation on part of feature space not containing NaN values. 
 
When the feature space no longer contain NaN values, two seperate models are built with 'ObjectPrice' as the dependent variable: 
 * Linear Regression model
 * Ada boost model (Average of weak learnerd kNN, LinearRegression and Regression Decision Tree)

Average RMSE of each model is printed to standard out (Each model is built ten times, and the average RMSE of the ten models is reported)

Module can be run by command


```commandline
python car_pricing/build.py {input_file_path}
```

## Future Work

Incorporate more weak learners into the Ada Boosting model, as well as incorporating a weighted average of the different weak learners based on performance.
Evolutionary algorithms or random grid search should work well for determining the different weights for the different weak learners, and these are considered hyper parameters that will change over time.

All operations can be translated down to simple mathematical formulas, and services that pull the metadata of the algorithms (ie, weights) can be set up to expose the models.
This way, the weights and metadata of the models can be changed allowing for re-training of models with no down time on model exposing services. 

## Authors

* **Sandra Lefdal**

