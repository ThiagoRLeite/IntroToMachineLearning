import pandas as pd

melbourne_file_path = 'data/melb_data.csv' #filepath to the data
melbourne_data = pd.read_csv(melbourne_file_path) #read data and put it into a DataFrame

y = melbourne_data.Price #select the prediction target
features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude'] #select the features...
X = melbourne_data.features #...and store it in a variable

#------------- BUILDING MODEL -------------#

from sklearn.tree import DecisionTreeRegressor

melbourne_model = DecisionTreeRegressor(random_state=1) #define model and specify a random_state to maintain the same results every run
melbourne_model.fit(X, y) #fit model (capture patterns from provided data)

#----------- MAKING PREDICTIONS -----------#

predictions = melbourne_model.predict(X)
print(predictions)
