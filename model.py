#import necessary libraries (some of these are not used in the final code, but were used in experimentation with other models)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

# import cleaned dataset with date, sentiment score, stock price, and future price. Also convert date to date structure
dataset = pd.read_csv('cleanData.csv')
dataset['dateMod'] = pd.to_datetime(dataset['dateMod'])

# average and collapse sentiment score values for dates that have multiple scores(multiple tweets)
data = dataset.groupby('dateMod').mean('compound_score').reset_index()


# Separate input features (X-sentiment scores and stock price) and target variable (y-future stock price)
X = dataset[['compound_score', 'Open']]
y = dataset['delta']

# Divide into training and testing sets using 80/20 split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# scale input features to between 0 and 1 so model will converge more quickly
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build layers of the network, incorporating dropout
model = keras.models.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    keras.layers.Dropout(0.15),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dropout(0.15),
    keras.layers.Dense(1)
])

# Compile with MSE loss/cost function
model.compile(optimizer='adam', loss='mean_squared_error')

# Training the model with training data
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.1)


# Make predictions using testing data
predictions = model.predict(X_test)

# Show loss from testing
test_loss = model.evaluate(X_test, y_test)
print(f'Test Loss: {test_loss}')

# Print predicted stock price values next to actual values
for i, pred in enumerate(predictions):
    print(f"Predicted: {pred[0]}, Actual: {y_test.iloc[i]}")

# print mean absolute error
mae = mean_absolute_error(y_test, predictions)
print(f'Mean Absolute Error: {mae}')