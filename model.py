import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras

# dataset
dataset = pd.read_csv('cleanData.csv')

# Separate input features (X) and target variable (y)
X = dataset[['compound_score', 'Open']]
y = dataset['delta']

# Divide into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# input features
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Set up layers (relu activation)
model = keras.models.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(1)
])

# Compile with loss/cost function
model.compile(optimizer='adam', loss='mean_squared_error')

# Training
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.1)

# Show loss from testing
test_loss = model.evaluate(X_test, y_test)
print(f'Test Loss: {test_loss}')

# predictions
predictions = model.predict(X_test)