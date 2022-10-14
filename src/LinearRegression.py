import numpy as np
from sklearn.linear_model import LinearRegression

import helpers

class Linear_Regression:
    def __init__(self, days_ahead = 1, av_window = 5, look_back = 10):
        """
        parameters:
            days_ahead: the number of days ahead the model predicts the temperature
            av_window: the number of previous temperatures to average across
            look_back: the length of the look_back window intended for calcualtions (not used for training)
        """
        self.days_ahead = days_ahead
        self.av_window = av_window
        self.look_back = look_back
        self.model = None

    def prepare_data(self, return_index = False):
        """Loads data, adds necessary features, and convert data into appropriate numpy arrays."""
        # loading data
        df = helpers.load_data()

        # computing average temperature for last 'av_window' days
        df['av'] = df['temp_f'].rolling(self.av_window).mean()

        # adding seasonal variables as described in helpers module
        df = helpers.add_seasonal_variables(df)

        # arranging samples in necessary format for model fitting
        all_vals = df.values
        X = []
        y = []

        for i in range(self.look_back, len(all_vals) - self.days_ahead):
            X.append(all_vals[i])
            y.append(all_vals[i+self.days_ahead][0])

        # getting last data point which will be used for the final prediction
        last_X = np.array([all_vals[-1]])

        if return_index:
            return np.array(X), np.array(y), last_X, df.index[self.look_back:len(df) - self.days_ahead]

        return np.array(X), np.array(y), last_X

    def fit_model(self, X, y):
        """Fit model to training data"""
        print('Fiting Linear Regressor...')

        self.model = LinearRegression()
        self.model.fit(X,y)

    def predict(self, X):
        """Predict temperature values given input values X"""
        if self.model is None:
            raise ValueError("No model found. Please run fit_model to fit a new model.")
        
        return self.model.predict(X)

        