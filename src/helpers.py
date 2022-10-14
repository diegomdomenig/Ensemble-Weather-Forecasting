import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt

def load_data():
    """Loads dataframe, reading the 'date' column as the index"""
    df = pd.read_csv('nyc_temperature.csv', header = 0, index_col = 'date', parse_dates = True)

    # checking for nan values
    assert df.isnull().values.any
    
    # making sure dataframe contains all days between 2015-5-14 to 2020-5-13
    assert (df.index == pd.date_range('2015-05-14', '2020-5-13')).all()

    return df

def add_seasonal_variables(df):
    """Adds two features to data which are sin and cosine curves with periodicity of 1 year"""
    # convert index of dataframe to seconds since 1970-01-01 00:00:00
    seconds = df.index.map(pd.Timestamp.timestamp)

    # compute number of seconds in a year
    seconds_per_day = 24 * 60 * 60
    seconds_per_year = 365.2425 * seconds_per_day

    # adding two columns to dataframe containing sin and cosine periodicity values
    df['sin year'] = np.sin(seconds * (2 * np.pi / seconds_per_year))
    df['cos year'] = np.cos(seconds * (2 * np.pi / seconds_per_year))
    
    return df

def train_val_test_split(X, y, q1, q2):
    """Splits dataset into a train, validation, and test dataset based on q1 < q2 in (0,1)."""
    train_split = int(q1 * len(X))
    val_split = int(q2 * len(X))
    
    X_train = X[:train_split]
    y_train = y[:train_split]
    X_val = X[train_split:val_split]
    y_val = y[train_split:val_split]
    X_test = X[val_split:]
    y_test = y[val_split:]
    
    return X_train, y_train, X_val, y_val, X_test, y_test

def normalize(train, val, test, mean, std):
    """Perform normalization on training, validation, and test data using training mean and std."""
    # We use same mean and std for all datasets to prevent look-ahead bias.
    train = (train - mean) / std
    val = (val - mean) / std
    test = (test - mean) / std

    return train, val, test

def plot_predictions(index, pred, actual, title, img_name):
    """Plot predictions vs actuals and save figure in plots folder."""
    fig, ax = plt.subplots(figsize = (20,10))
    ax.plot(index, actual, label = "actual")
    ax.plot(index, pred, label = "predicted")
    ax.legend(loc="best")
    ax.set_title(title)
    fig.savefig(f'plots/{img_name}.png')
    plt.close(fig)

def plot_histogram(x, bins, xlabel, ylabel, title, img_name):
    """Plot a histogram of values. Used for residuals of testing dataset."""
    fig, ax = plt.subplots(figsize = (20,10))
    ax.hist(x, bins = bins)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.savefig(f'plots/{img_name}.png')
    plt.close(fig)

def evaluate_predictions(pred, actual, title):
    """Print MSE and RMSE of predictions."""
    mse = ((actual - pred)**2).mean()
    rmse = np.sqrt(mse)

    print("############################################################")
    print(title)
    print()
    print(f'Mean Squared Error: {mse}')
    print(f'Root Mean Squared Error: {rmse}')
    print("############################################################")
    print()


def progressbar(it, prefix="", size=60, out=sys.stdout):
    """Display progress bar on screen"""
    count = len(it)
    def show(j):
        x = int(size*j/count)
        print("{}[{}{}] {}/{}".format(prefix, "#"*x, "."*(size-x), j, count), 
                end='\r', file=out, flush=True)
    show(0)
    for i, item in enumerate(it):
        yield item
        show(i+1)
    print("\n", flush=True, file=out)