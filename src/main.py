import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

from LinearRegression import Linear_Regression
from ARIMA import ARIMA_Regression
from LSTM import LSTM_Regression

from helpers import train_val_test_split, normalize, evaluate_predictions, plot_predictions, plot_histogram

def train_models(days_ahead = 1, plot = False):
    """
    First, we train 3 separate models on the training dataset. These models are a linear regression,
    an ARIMA regression model, and an LSTM recurrent neural network. We then use the validation dataset
    as a training dataset to optimize weights for how to combine these three models into one. The testing
    dataset is then used to train a linear predictor of the standard deviation of returns of our combined
    model, which we use to compute our prediction interval
    """
    # Defining three regression models
    linear_regression = Linear_Regression(days_ahead = days_ahead)
    arima_regression = ARIMA_Regression(days_ahead = days_ahead)
    lstm_regression = LSTM_Regression(days_ahead = days_ahead, window_size = 10)

    # Getting and preparing data for the three regression models
    X1, y1, _last_X1, index = linear_regression.prepare_data(return_index=True)
    X2, y2, _last_X2 = arima_regression.prepare_data()
    X3, y3, _last_X3 = lstm_regression.prepare_data()

    # First 60% (3 years) of dataset will be used as training, next 
    # 20% (1 year) will be used for validation, and last 20% (1 year)
    # will be used for testing.
    q1 = 0.6
    q2 = 0.8

    # Using same train-val-test split on the index of the dataframe
    # which will be used for plotting
    train_index, _, val_index, _, test_index, _ = train_val_test_split(index, index, q1, q2)

    # Splitting data into training, validation, and testing sets.
    # We will use the prefix "_" to denote the fact that these values
    # are not normalized.
    _X1_train, _y1_train, _X1_val, _y1_val, _X1_test, _y1_test = train_val_test_split(X1, y1, q1, q2)
    _X2_train, _y2_train, _X2_val, _y2_val, _X2_test, _y2_test = train_val_test_split(X2, y2, q1, q2)
    _X3_train, _y3_train, _X3_val, _y3_val, _X3_test, _y3_test = train_val_test_split(X3, y3, q1, q2)

    # Calculating mean and std for the datasets of the three models.
    # We do this separately for each model since some of the datasets
    # have different dimensions (_X3_train is a 3d-tensor).
    # We use the training dataset to compute the means and standard
    # deviation to avoid look-ahead bias.
    X1_mean, X1_std = np.mean(_X1_train), np.std(_X1_train)
    X2_mean, X2_std = np.mean(_X2_train), np.std(_X2_train)
    X3_mean, X3_std = np.mean(_X3_train), np.std(_X3_train)

    y1_mean, y1_std = np.mean(_y1_train), np.std(_y1_train)
    y2_mean, y2_std = np.mean(_y2_train), np.std(_y2_train)
    y3_mean, y3_std = np.mean(_y3_train), np.std(_y3_train)

    # Normlizing data.
    X1_train, X1_val, X1_test = normalize(_X1_train, _X1_val, _X1_test, X1_mean, X1_std)
    y1_train, y1_val, y1_test = normalize(_y1_train, _y1_val, _y1_test, y1_mean, y1_std)

    X2_train, X2_val, X2_test = normalize(_X2_train, _X2_val, _X2_test, X2_mean, X2_std)
    y2_train, y2_val, y2_test = normalize(_y2_train, _y2_val, _y2_test, y2_mean, y2_std)

    X3_train, X3_val, X3_test = normalize(_X3_train, _X3_val, _X3_test, X3_mean, X3_std)
    y3_train, y3_val, y3_test = normalize(_y3_train, _y3_val, _y3_test, y3_mean, y3_std)

    # Fitting models to training dataset.
    linear_regression.fit_model(X1_train, y1_train)
    arima_regression.fit_model(X2_train, y2_train)
    lstm_regression.fit_model(X3_train, y3_train, X3_val, y3_val)

    # Using fitted models to predict temperatures of the validation set.
    y1_val_pred = linear_regression.predict(X1_val)
    y2_val_pred = arima_regression.predict(X2_val)
    y3_val_pred = lstm_regression.predict(X3_val)

    # Defining function we want to minimize over the validation set.
    def fun(w):
        """Computes the linear combination of predictions of the 3 models and then
        calcualtes the mean squared error, which we will optimize"""
        # linear combination of predictions from the 3 models
        y_val_pred = w[0] * y1_val_pred + w[1] * y2_val_pred + w[2] * y3_val_pred

        # calculae mean squared error
        mse = ((y_val_pred - y1_val)**2).mean()

        return mse

    # optimizing the weights w1, w2, w3 by minimizing the loss of 
    # the function defined above.
    res = minimize(fun, [0.3, 0.3, 0.4])
    
    if not res.success:
        raise ValueError("Model Optimization did not converge.")

    [w1, w2, w3] = res.x
    print(f"Weights of combined model: ({w1}, {w2}, {w3})")

    # Predicting the test set using 3 models.
    y1_test_pred = linear_regression.predict(X1_test)
    y2_test_pred = arima_regression.predict(X2_test)
    y3_test_pred = lstm_regression.predict(X3_test)

    # using weights minimized above to compute the prediction of
    # our combined model.
    test_pred = w1 * y1_test_pred + w2 * y2_test_pred + w3 * y3_test_pred

    # un-normalizing the test prediction by reversing the normalizaion
    # process.
    _test_pred = test_pred * y1_std + y1_mean

    # calculate residuals of test predictions using un_normalized data
    residuals = _y1_test - _test_pred

    # We want to create a dataset to train our standard-deviation predictor.
    # The features of the X-matrix contain all the features of the data used
    # for the linear regression above as well as two additional feature which is
    # the standard deviation of the past 5 residuals and the predicted temperature
    # for the data point we want to predict the std for. Our linear predictor will
    # try to predict the standard deviation of the 10 residuals "around" the current
    # data point.
    X_residual = []
    y_residual = []
    n_of_res = 10 # number of residuals that make up the std value of the y values
    half_res = int(n_of_res/2)  # the number of residuals that will be used to calculate
                                # the std for the X-matrix

    # TODO: Include predicted temperature in the X variables
    for i in range(half_res, len(residuals) - half_res):
        temp = list(_X1_test[i])
        temp.append(_test_pred[i+1]) # including temperature prediction in model
        temp.append(np.std(residuals[i-half_res:i])) # including std of past 5 residuals in model

        X_residual.append(temp)
        y_residual.append(np.std(residuals[i-half_res:i+half_res]))

    _X_residual = np.array(X_residual)
    _y_residual = np.array(y_residual)

    # calculating means and standard deviations for normalization
    mean_X_res, std_X_res = np.mean(X_residual), np.std(X_residual)
    mean_y_res, std_y_res = np.mean(y_residual), np.std(y_residual)

    # noramlizing data
    X_residual = (_X_residual - mean_X_res) / std_X_res
    y_residual = (_y_residual - mean_y_res) / std_y_res

    # defining linear regression for predicting std of residuals
    # and fitting to data
    print('Fitting residual model...')
    residual_model = Linear_Regression(days_ahead = days_ahead)
    residual_model.fit_model(X_residual, y_residual)

    # normalizing data tensor we will use for making the final prediction
    last_X1 = (_last_X1 - X1_mean) / X1_std
    last_X2 = (_last_X2 - X2_mean) / X2_std
    last_X3 = (_last_X3 - X3_mean) / X3_std

    # predicting final temperature
    predicted_temperature_normalized = w1 * linear_regression.predict(last_X1) + w2 * arima_regression.predict(last_X2) + w3 * lstm_regression.predict(last_X3)
    
    # un-normalizing predicted temperature
    [predicted_temperature] = predicted_temperature_normalized * y1_std + y1_mean

    # creating data tensor for std prediction
    temp = list(_last_X1[0])
    temp.append(predicted_temperature)
    temp.append(np.std(residuals[-half_res]))
    X_res = np.array([temp])

    # normalizing data tensor for std prediction
    X_res = (X_res - mean_X_res) / std_X_res
    
    # predicting the std of residuals for the predicted temperature
    predicted_std_normalized = residual_model.predict(X_res)

    # un-normalizing predicted std
    [predicted_std] = predicted_std_normalized * std_y_res + mean_y_res

    if plot:
        # predicting normalized temperatures for train, val, and test datasets
        y1_train_pred = linear_regression.predict(X1_train)
        y2_train_pred = arima_regression.predict(X2_train)
        y3_train_pred = lstm_regression.predict(X3_train)

        y1_val_pred = linear_regression.predict(X1_val)
        y2_val_pred = arima_regression.predict(X2_val)
        y3_val_pred = lstm_regression.predict(X3_val)

        y1_test_pred = linear_regression.predict(X1_test)
        y2_test_pred = arima_regression.predict(X2_test)
        y3_test_pred = lstm_regression.predict(X3_test)

        all_train_pred = w1 * y1_train_pred + w2 * y2_train_pred + w3 * y3_train_pred
        all_val_pred = w1 * y1_val_pred + w2 * y2_val_pred + w3 * y3_val_pred
        all_test_pred = w1 * y1_test_pred + w2 * y2_test_pred + w3 * y3_test_pred

        res_preds = residual_model.predict(X_residual)

        # computing un-normalized temperatures for train, val, and test datasets
        _y1_train_pred = y1_train_pred * y1_std + y1_mean
        _y2_train_pred = y2_train_pred * y2_std + y2_mean
        _y3_train_pred = y3_train_pred * y3_std + y3_mean

        _y1_val_pred = y1_val_pred * y1_std + y1_mean
        _y2_val_pred = y2_val_pred * y2_std + y2_mean
        _y3_val_pred = y3_val_pred * y3_std + y3_mean

        _y1_test_pred = y1_test_pred * y1_std + y1_mean
        _y2_test_pred = y2_test_pred * y2_std + y2_mean
        _y3_test_pred = y3_test_pred * y3_std + y3_mean

        _all_train_pred = all_train_pred * y1_std + y1_mean
        _all_val_pred = all_val_pred * y1_std + y1_mean
        _all_test_pred = all_test_pred * y1_std + y1_mean

        _res_preds = res_preds * std_y_res + mean_y_res

        d_ahead_string = "1 Day Ahead" if days_ahead == 1 else f"{days_ahead} Days Ahead"

        print('Evaluating models...')

        evaluate_predictions(_y1_train_pred, _y1_train, f"Results for Linear Regression on Training Set - {d_ahead_string}")
        evaluate_predictions(_y2_train_pred, _y2_train, f"Results for ARIMA Regression on Training Set - {d_ahead_string}")
        evaluate_predictions(_y3_train_pred, _y3_train, f"Results for LSTM Regression on Training Set - {d_ahead_string}")
        evaluate_predictions(_y1_val_pred, _y1_val, f"Results for Linear Regression on Validation Set - {d_ahead_string}")
        evaluate_predictions(_y2_val_pred, _y2_val, f"Results for ARIMA Regression on Validation Set - {d_ahead_string}")
        evaluate_predictions(_y3_val_pred, _y3_val, f"Results for LSTM Regression on Validation Set - {d_ahead_string}")
        evaluate_predictions(_y1_test_pred, _y1_test, f"Results for Linear Regression on Test Set - {d_ahead_string}")
        evaluate_predictions(_y2_test_pred, _y2_test, f"Results for ARIMA Regression on Test Set - {d_ahead_string}")
        evaluate_predictions(_y3_test_pred, _y3_test, f"Results for LSTM Regression on Test Set - {d_ahead_string}")
        evaluate_predictions(_all_train_pred, _y1_train, f"Results for Combined Regression on Train Set - {d_ahead_string}")
        evaluate_predictions(_all_val_pred, _y1_val, f"Results for Combined Regression on Validation Set - {d_ahead_string}")
        evaluate_predictions(_all_test_pred, _y1_test, f"Results for Combined Regression on Test Set - {d_ahead_string}")
        evaluate_predictions(_res_preds, _y_residual, f"Results for Residual Linear Model on Test Set - {d_ahead_string}")

        print('Plotting Normalized predictions...')
        plot_predictions(train_index, y1_train_pred, y1_train, f"Normalized Temperatures for Linear Regression Model - Training Data - {d_ahead_string}", f"LR-TR-NORM-{days_ahead}")
        plot_predictions(train_index, y2_train_pred, y2_train, f"Normalized Temperatures for SARIMAX Regression Model - Training Data - {d_ahead_string}", f"SA-TR-NORM-{days_ahead}")
        plot_predictions(train_index, y3_train_pred, y3_train, f"Normalized Temperatures for LSTM Regression Model - Training Data - {d_ahead_string}", f"LS-TR-NORM-{days_ahead}")
        plot_predictions(val_index, y1_val_pred, y1_val, f"Normalized Temperatures for Linear Regression Model - Validation Data - {d_ahead_string}", f"LR-VA-NORM-{days_ahead}")
        plot_predictions(val_index, y2_val_pred, y2_val, f"Normalized Temperatures for SARIMAX Regression Model - Validation Data - {d_ahead_string}", f"SA-VA-NORM-{days_ahead}")
        plot_predictions(val_index, y3_val_pred, y3_val, f"Normalized Temperatures for LSTM Regression Model - Validation Data - {d_ahead_string}", f"LS-VA-NORM-{days_ahead}")
        plot_predictions(test_index, y1_test_pred, y1_test, f"Normalized Temperatures for Linear Regression Model - Test Data - {d_ahead_string}", f"LR-TE-NORM-{days_ahead}")
        plot_predictions(test_index, y2_test_pred, y2_test, f"Normalized Temperatures for SARIMAX Regression Model - Test Data - {d_ahead_string}", f"SA-TE-NORM-{days_ahead}")
        plot_predictions(test_index, y3_test_pred, y3_test, f"Normalized Temperatures for LSTM Regression Model - Test Data - {d_ahead_string}", f"LS-TE-NORM-{days_ahead}")
        plot_predictions(train_index, all_train_pred, y1_train, f"Normalized Temperatures for Combined Regression Model - Train Data - {d_ahead_string}", f"CR-TR-NORM-{days_ahead}")
        plot_predictions(val_index, all_val_pred, y1_val, f"Normalized Temperatures for Combined Regression Model - Validation Data - {d_ahead_string}", f"CR-VA-NORM-{days_ahead}")
        plot_predictions(test_index, all_test_pred, y1_test, f"Normalized Temperatures for Combined Regression Model - Test Data - {d_ahead_string}", f"CR-TE-NORM-{days_ahead}")
        plot_predictions(test_index[5:-5], res_preds, y_residual, "Normalized Stdevs for Residual Regression Model - Test Data - {d_ahead_string}", f"RR-TE-NORM-{days_ahead}")
        
        print('Plotting Real Predictions...')
        plot_predictions(train_index, _y1_train_pred, _y1_train, f"Predicted Temperatures for Linear Regression Model - Training Data - {d_ahead_string}", f"LR-TR-{days_ahead}")
        plot_predictions(train_index, _y2_train_pred, _y2_train, f"Predicted Temperatures for SARIMAX Regression Model - Training Data - {d_ahead_string}", f"SA-TR-{days_ahead}")
        plot_predictions(train_index, _y3_train_pred, _y3_train, f"Predicted Temperatures for LSTM Regression Model - Training Data - {d_ahead_string}", f"LS-TR-{days_ahead}")
        plot_predictions(val_index, _y1_val_pred, _y1_val, f"Predicted Temperatures for Linear Regression Model - Validation Data - {d_ahead_string}", f"LR-VA-{days_ahead}")
        plot_predictions(val_index, _y2_val_pred, _y2_val, f"Predicted Temperatures for SARIMAX Regression Model - Validation Data - {d_ahead_string}", f"SA-VA-{days_ahead}")
        plot_predictions(val_index, _y3_val_pred, _y3_val, f"Predicted Temperatures for LSTM Regression Model - Validation Data - {d_ahead_string}", f"LS-VA-{days_ahead}")
        plot_predictions(test_index, _y1_test_pred, _y1_test, f"Predicted Temperatures for Linear Regression Model - Test Data - {d_ahead_string}", f"LR-TE-{days_ahead}")
        plot_predictions(test_index, _y2_test_pred, _y2_test, f"Predicted Temperatures for SARIMAX Regression Model - Test Data - {d_ahead_string}", f"SA-TE-{days_ahead}")
        plot_predictions(test_index, _y3_test_pred, _y3_test, f"Predicted Temperatures for LSTM Regression Model - Test Data - {d_ahead_string}", f"LS-TE-{days_ahead}")
        plot_predictions(train_index, _all_train_pred, _y1_train, f"Predicted Temperatures for Combined Regression Model - Train Data - {d_ahead_string}", f"CR-TR-{days_ahead}")
        plot_predictions(val_index, _all_val_pred, _y1_val, f"Predicted Temperatures for Combined Regression Model - Validation Data - {d_ahead_string}", f"CR-VA-{days_ahead}")
        plot_predictions(test_index, _all_test_pred, _y1_test, f"Predicted Temperatures for Combined Regression Model - Test Data - {d_ahead_string}", f"CR-TE-{days_ahead}")
        plot_predictions(test_index[5:-5], _res_preds, _y_residual, "Predicted Stdevs for Residual Regression Model - Test Data - {d_ahead_string}", f"RR-TE-{days_ahead}")

        plot_histogram(residuals, 30, "Residuals", "Probability", f"Residuals of Combined Regression Model on Test Data - {d_ahead_string}", f"Res-TE-{days_ahead}")


    return predicted_temperature, predicted_std

if __name__ == '__main__':
    temp, std = train_models(days_ahead = 30, plot=True)

    # The prediction interval goes from temp - 2*std to temp + 2*std
    pi = [temp - 2*std, temp + 2*std]

    print(f"Predicted Temperature: {temp}, 95% Prediction Interval: ({pi[0]}, {pi[1]})")