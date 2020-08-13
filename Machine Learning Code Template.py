# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 14:18:19 2020

@author: Paul
"""

""""""
#Import all required libraries
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error

""""""
# Define filepath and specify the data you want to train/test
datadata = pd.read_csv(#Filepath here)
data.head(3)
target = data.#TargetData
inputs = data.drop(#Column to drop, axis=1)
results = train_test_split(inputs, target, test_size = 0.2, random_state =1)
input_train, input_test, target_train, target_test = results

""""""
# Create pipelines and define hyperparameters
pipelines = {
    'lasso': make_pipeline(StandardScaler(), Lasso(random_state=1)),
    'ridge': make_pipeline(StandardScaler(), Ridge(random_state=1)),
    'elasticnet': make_pipeline(StandardScaler(), ElasticNet(random_state=1)),
    'rf': make_pipeline(StandardScaler(), RandomForestRegressor(random_state=1)),
    'gb' : make_pipeline(StandardScaler(), GradientBoostingRegressor(random_state=1))
    }

lasso_hyperparameters = {
    'lasso__alpha' : [0.01, 0.05, 0.1, 0.5, 1, 5]
    }
ridge_hyperparameters = {
    'ridge__alpha' : [0.01, 0.05, 0.1, 0.5, 1, 5]
    }
elasticnet_hyperparameters = {
    'elasticnet__alpha' : [0.01, 0.05, 0.1, 0.5, 1, 5],
    'elasticnet__l1_ratio' : [0.1, 0.3, 0.5, 0.7, 0.9]
    }
rf_hyperparameters = {
    'randomforestregressor__n_estimators' : [100, 200],
    'randomforestregressor__max_features' : ['auto', 0.3, 0.6]
    }
gb_hyperparameters = {
    'gradientboostingregressor__n_estimators' : [100, 200],
    'gradientboostingregressor__learning_rate' : [0.05, 0.1, 0.2],
    'gradientboostingregressor__max_depth' : [1, 3, 5]
    }
hyperparameter_grids = {
    'lasso' : lasso_hyperparameters,
    'ridge' : ridge_hyperparameters,
    'elasticnet' : elasticnet_hyperparameters,
    'rf' : rf_hyperparameters,
    'gb' : gb_hyperparameters
    }

""""""
# Create predictive model outputs
models = {}
for key in pipelines.keys():
    models[key] = GridSearchCV(pipelines[key], hyperparameter_grids[key], cv=5)

for key in models.keys():
    models[key].fit(input_train, target_train)
    print(key, 'is trained and tuned')

for key in models:
    preds = models[key].predict(input_test)
    plt.scatter(preds, target_test)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    print(key)
    print('R-Squared: ', round(r2_score(target_test, preds), 3))
    print('MAE: ', round(mean_absolute_error(target_test, preds), 3))
    print('---')