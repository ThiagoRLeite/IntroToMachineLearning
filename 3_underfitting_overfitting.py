import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor

def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0) # making a tree regressor using a variable to define different max_leaf_nodes value
    model.fit(train_X, train_y) # fit the model with the train variables
    preds_val = model.predict(val_X) # predict the validation features variable
    mae = mean_absolute_error(val_y, preds_val) # calculating the MAE between predicted and validation data
    return(mae) # return MAE value

melbourne_file_path = 'data/melb_data.csv' # filepath to the data
melbourne_data = pd.read_csv(melbourne_file_path) # read data and put it into a DataFrame

y = melbourne_data.Price # select the prediction target
features = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea', 'YearBuilt', 'Lattitude', 'Longtitude'] # select the features...
X = melbourne_data[features] # ... and store them in a variable

from sklearn.model_selection import train_test_split

train_X, val_X, train_y, val_y = train_test_split(X, y,random_state = 0) # split data into training and validation data

for max_leaf_nodes in [5, 50, 500, 5000]: # show MAE using different max_leaf_nodes values
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y) # call function to get MAE value for differente max_leaf_nodes values
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))

# overfitting: capturing spurious patterns that won't recur in the future, leading to less accurate predictions
# underfitting: failing to capture relevant patterns, again leading to less accurate predictions
