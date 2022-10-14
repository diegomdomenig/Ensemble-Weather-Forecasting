from socket import AI_CANONNAME
import pandas as pd
import numpy as np
from itertools import product
from statsmodels.tsa.statespace.sarimax import SARIMAX

import warnings
warnings.filterwarnings("ignore")

import helpers

class ARIMA_Regression:
    def __init__(self, days_ahead = 1, look_back = 10):
        """
        parameters:
            days_ahead: the number of days ahead the model predicts the temperature
            look_back: the length of the look_back window intended for calcualtions (not used for training)
        """
        self.days_ahead = days_ahead
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
            X.append(all_vals[i])
            y.append(all_vals[i+self.days_ahead][0])
        
        # getting last data point which will be used for the final prediction
        last_X = np.array([all_vals[-1]])

        return np.array(X), np.array(y), last_X

    def fit_model(self, X, y):
        """Fit model to training data"""
        print('Fiting ARIMA Regressor...')

        # defining all possible values for p, d, s. We use zero-differentiated ARIMA models
        ps = range(0,5)
        ds = [0]
        qs = range(0,5)

        parameters = list(product(ps, ds, qs))

        best_model = None
        best_aic = np.inf
        best_params = None

        # grid search through all possible parameters of (p, d, q) and choose the best model
        for order in helpers.progressbar(parameters, prefix = "Computing: "):
            model = SARIMAX(exog = X, endog = y, order=order).fit(disp = False)

            # If MLE does not converge, mle_retvals['warnflag'] will be set to 1 which we do
            # not want to consider. Thus we check whether the MLE converges and whether the
            # model's AIC is less than the current best AIC and decide on the best model
            if model.mle_retvals['warnflag'] == 0 and model.aic < best_aic:
                best_aic = model.aic
                best_model = model
                best_params = order

        self.model = best_model
        print(f'Optimal Parameters for ARIMA: {best_params}')

    def predict(self, X):
        """Predict temperature values given input values X"""
        if self.model is None:
            raise ValueError("No model found. Please run fit_model to fit a new model.")

        return self.model.get_forecast(steps = len(X), exog = X).predicted_mean