"""
Created on Thu Nov 18 17:38:19 2021

@author: jameslanser
"""


# Basic Data Exploration

import pandas as pd

# set path of data
melbourne_file_path = 'Data/melb_data.csv'

# read csv and assign to variable
melbourne_data = pd.read_csv(melbourne_file_path)

# select prediction target
y = melbourne_data.Price

# choose features
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
x = melbourne_data[melbourne_features]

# display data description
melbourne_data.columns
x.describe()
x.head()


# Decision Tree - Predict Home Price

from sklearn.tree import DecisionTreeRegressor

# define model
melbourne_model = DecisionTreeRegressor(random_state=1)

# fit model
melbourne_model.fit(x,y)

# output preictions
print("Predictions for the following 5 houses...")
print(x.head())
print("Predictions...")
print(melbourne_model.predict(x.head()))


# Decision Tree - Model Validation

from sklearn.metrics import mean_absolute_error

# predict home prices
predicted_home_prices = melbourne_model.predict(x)

# mean absolute error (error = actual - predicted)
mean_absolute_error(y, predicted_home_prices)


# Decision Tree - Train/Test Split (and Re-run Model)

from sklearn.model_selection import train_test_split

# split data into train/test for both features and target
train_x, val_x, train_y, val_y = train_test_split(x, y, random_state=0)

# define model
melbourne_model = DecisionTreeRegressor()

# fit model
melbourne_model.fit(train_x, train_y)

# get predicted prices on validation data
val_predictions = melbourne_model.predict(val_x)
print(mean_absolute_error(val_y, val_predictions))


# Decision Tree - Number of Leaf Nodes

from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor

# define function to to run model with number of leaf nodes as input
def get_mae(max_leaf_nodes, train_x, val_x, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_x, train_y)
    pred_val = model.predict(val_x)
    mae = mean_absolute_error(val_y, pred_val)
    return(mae)

# iterate through number of leaf nodes
for max_leaf_nodes in [5, 50, 500, 5000]:
    run_mae = get_mae(max_leaf_nodes, train_x, val_x, train_y, val_y)
    print("Max Leaf Nodes: %d \t\t Mean Absolute Error: %d"
          %(max_leaf_nodes, run_mae))

# Random Forest - Predict Home Price

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# define model
forest_model = RandomForestRegressor(random_state=1)

# fit model
forest_model.fit(train_x, train_y)
melb_pred = forest_model.predict(val_x)
print(mean_absolute_error(val_y, melb_pred))
