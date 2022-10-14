import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, LSTM, Dense
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam

import helpers

class LSTM_Regression:
    def __init__(self, days_ahead = 1, window_size = 7, look_back = 10):
        """
        parameters:
            days_ahead: the number of days ahead the model predicts the temperature
            window_size: size of window of number of previous temperatures to use for the sequence
            look_back: the length of the look_back window intended for calcualtions (not used for training)
        """
        assert window_size <= look_back

        self.days_ahead = days_ahead
        self.window_size = window_size
        self.look_back = look_back
        self.model = None

    def prepare_data(self):
        """Loads data, adds necessary features, and convert data into appropriate numpy arrays."""
        # loading data
        df = helpers.load_data()

        # adding seasonal variables as described in helpers module
        df = helpers.add_seasonal_variables(df)

        # arranging samples in necessary format for model fitting
        all_vals = df.values
        X = []
        y = []
        
        for i in range(self.look_back, len(all_vals) - self.days_ahead):
            X.append([v for v in all_vals[i-self.window_size+1:i+1]])
            y.append(all_vals[i+self.days_ahead][0])
        
        # getting last data point which will be used for the final prediction
        last_X = np.array([[v for v in all_vals[len(all_vals)-self.window_size + 2:len(all_vals) - 1]]])

        return np.array(X), np.array(y), last_X

    def fit_model(self, X, y, X_val, y_val, recurrent_units = 64, dense_units = 8, lr = 0.001, epochs = 20):
        """Fit model to training data"""
        print('Fiting LSTM Regressor...')

        # defining the model
        self.model = Sequential()
        self.model.add(InputLayer((self.window_size, 3)))
        self.model.add(LSTM(recurrent_units))
        self.model.add(Dense(dense_units, 'relu'))
        self.model.add(Dense(1, 'linear'))

        # defining checkpoint to save best model in memory
        checkpoint = ModelCheckpoint('model/', save_best_only=True)
        self.model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=lr), metrics=[RootMeanSquaredError()])

        # fitting model, using validation data to determine which epoch of weights to use
        self.model.fit(X, y, validation_data=(X_val, y_val), epochs=epochs, callbacks=[checkpoint], verbose = 1)

    def predict(self, X):
        """Predict temperature values given input values X"""
        if self.model is None:
            raise ValueError("No model found. Please run fit_model to fit a new model.")
        
        return self.model.predict(X).flatten()