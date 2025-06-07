import pandas as pd

melbourne_file_path = 'data/melb_data.csv' # filepath to the data
melbourne_data = pd.read_csv(melbourne_file_path) # read data and put it into a DataFrame

y = melbourne_data.Price # select the prediction target
features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude'] # select the features...
X = melbourne_data[features] # ... and store them in a variable

#------------- BUILDING MODEL -------------#

from sklearn.tree import DecisionTreeRegressor

melbourne_model_1 = DecisionTreeRegressor(random_state=1) # define model and specify a random_state to maintain the same results every run
melbourne_model_1.fit(X, y) # fit model (capture patterns from provided data)

#----------- MAKING PREDICTIONS -----------#

predicted_values_1 = melbourne_model_1.predict(X)

#---------- MEAN ABSOLUTE ERROR -----------#
# the error is defined as: error = actual_value - predicted, the mean absolute error method takes the average of the absolute values of all errors

from sklearn.metrics import mean_absolute_error

mean_absolute_error_1 = mean_absolute_error(y, predicted_values_1)
print("MEAN ABSOLUTE ERROR: " + str(mean_absolute_error_1))

#------------ VALIDATION DATA -------------#
# use some of the data previously used to train the model to validate it instead

from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state=1) # split into training and validation data, it's based on a random number
melbourne_model_2 = DecisionTreeRegressor(random_state=1) # define model
melbourne_model_2.fit(X_train, y_train) # fit the model with the train variables
predicted_values_2 = melbourne_model_2.predict(X_valid) # predict the validation features variable
mean_absolute_error_2 = mean_absolute_error(y_valid, predict_values_2) # valid the predicted values with the validation data

print("MEAN ABSOLUTE ERROR: " + str(mean_absolute_error_2))
