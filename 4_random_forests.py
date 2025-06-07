import pandas as pd

melbourne_file_path = 'data/melb_data.csv' # filepath to the data
melbourne_data = pd.read_csv(melbourne_file_path) # read data and put it into a DataFrame

y = melbourne_data.Price # select the prediction target
features = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea', 'YearBuilt', 'Lattitude', 'Longtitude'] # select the features...
X = melbourne_data[features] # ... and store them in a variable

from sklearn.model_selection import train_test_split

train_X, val_X, train_y, val_y = train_test_split(X, y,random_state = 0) # split data into training and validation data

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

forest_model = RandomForestRegressor(random_state=1) # using RandomForestRegressor as model instead of DecisionTreeRegressor
forest_model.fit(train_X, train_y) # fit the model with the train variables
preds_val = model.predict(val_X) # predict the validation features variable
mae = mean_absolute_error(val_y, preds_val) # calculating the MAE between predicted and validation data
print(mae)
