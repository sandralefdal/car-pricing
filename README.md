# Car Pricing

Project to predict car prices.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

You will need Python 3 to run the project. To check which python is on your machine, run in the command line:

```
python --version
```

### Installing

First, you should create a virtualenvironment to install required dependencies in

To create a virtualenvironment with name {name_of_env}, make sure you have virtualenv installed, and run:

```
virtualenv {name_of_env}
```

Now, activate the virtual environment

```
source {name_of_env}/bin/activate
```

And install the necessary dependencies
```
pip install -r requirements.txt
```

Run the command 

```commandline
python car_pricing/build.py {input_file_path}
```

to perform feature engineering if necessary and build two models;
 * Linear Regression model
 * Ada boost model
Average RMSE of 10 built models of each model is printed to standard out
Feature engineering performed is also printed to standard out

## Deployment

This code is written to show a proof-of-concept. 
Further development must be made to make the models deployment ready.
Further discussion of productionalising and deployment can be found in separate write-up.

## Authors

* **Sandra Lefdal**

