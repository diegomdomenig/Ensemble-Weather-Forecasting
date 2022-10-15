# Ensemble-Weather-Forecasting

This is a regression model that combines predictions from a linear regression model, ARIMA regression model, and LSTM recurrent neural network 
to predict the temperature as well as a 95% prediction interval n days into the future.

# Prerequisites
python 3 <br>
numpy <br>
pandas <br>
matplotlib <br>
scipy <br>
sklearn <br>
tensorflow <br>
statsmodels <br>

# Running the code
To run the code, choose the days_ahead parameter in the `main.py` file (which is found in the `src` folder) which represents how many days into the future
the model will predict and then run ```python3 main.py``` to train the models and make the predictions.


