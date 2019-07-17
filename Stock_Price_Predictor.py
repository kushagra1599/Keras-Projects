# Importing header files
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Importing the data set
training_set = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = training_set.iloc[:,1:2].values

# Scaling the dataset
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
training_set = scaler.fit_transform(training_set)

# Creating input and label dataset
X_train = training_set[0:1257]
Y_train = training_set[1:1258]
X_train = np.reshape(X_train,(1257,1,1))

# importing header files for making RNN
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

regressor = Sequential()
regressor.add(LSTM(units=4,activation='sigmoid',input_shape=(None,1)))
regressor.add(Dense(units=1))
regressor.compile(optimizer='adam',loss='mean_squared_error')

# Fitting the dataset to the regressor
regressor.fit(X_train,Y_train,batch_size=32,epochs=200)

# Inputting the test set
test_set = pd.read_csv('Google_Stock_Price_Test.csv')
real_set = test_set.iloc[:,1:2].values

inputa = real_set
inputs = scaler.transform(inputa)
inputs = np.reshape(inputs,(20,1,1))

# Predicting the test set
predict = regressor.predict(inputs)
predict = scaler.inverse_transform(predict)

# Plotting the real output vs predicted output
plt.plot(real_set, color = 'red', label = 'Real Google Stock Price')
plt.plot(predict, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()