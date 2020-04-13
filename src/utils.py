import os
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer

DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

FILE_DATASET_FULL = "energy_model_dataset_anonymized.csv"
FILE_DATASET_TEST = "energy_model_dataset_anonymized_test.csv"

LOAD_TO_PREDICT = "C4"
FREQ = "1H" # frequency of the time series

FCAST_STEPS = 48

def get_dataset(load_to_predict=LOAD_TO_PREDICT, is_test=False):

    fname = FILE_DATASET_TEST if is_test else FILE_DATASET_FULL
    data = pd.read_csv(os.path.join(os.getcwd(), "..", "data", fname))
    data["date"] = data["date"].apply(lambda dates: datetime.strptime(dates, DATE_FORMAT))
    data.set_index("date", drop=True, inplace=True)

    # define which load to use for forecasting and other general constants
    c_data = pd.DataFrame()
    c_data["energy"] = data[load_to_predict].dropna()
    c_data["temperature"] = data["temperature"].dropna()
    c_data = c_data.asfreq(FREQ,"bfill")

    return c_data

def forecast_split(data, n_steps=FCAST_STEPS, freq="1H"):
    """
    Divide the dataset in two parts, one for training the model and the
    other for assessing forecasts

    Parameters
    ----------
    data: pd.Series with the original time series
    n_steps: number of forecasting steps
    freq: the time series period given in standard Pandas format

    Returns
    -------
    t_target: pd.Series with the training data
    f_target: pd.Series with the forecast data
    fcast_range: pd.DateRange with the forecasting timestamps
    """

    # the complete time series
    c_target = data.copy()

    # data used for training
    date = c_target.index[-1] - pd.Timedelta(hours=n_steps)
    t_target = c_target[c_target.index <= date]

    # data used for forecasting
    f_target = c_target[c_target.index > date]
    fcast_initial_date = f_target.index[0]
    fcast_range = pd.date_range(fcast_initial_date, periods=n_steps, freq=freq)

    print(f"Full available time range: from {c_target.index[0]} to {c_target.index[-1]}")
    print(f"Training time range: from {t_target.index[0]} to {t_target.index[-1]}")
    print(f"Forecasting time range: from {fcast_range[0]} to {fcast_range[-1]}")

    return t_target, f_target, fcast_range

def mape(y, yhat, perc=True):
    """
    Safe computation of the Mean Average Percentage Error

    Parameters
    ----------
    y: pd.Series or np.array holding the actual values
    yhat: pd.Series or np.array holding the predicted values
    perc: if True return the value in percentage

    Returns
    -------
    the MAPE value
    """
    err = -1.
    try:
        m = len(y.index) if type(y) == pd.Series else len(y) 
        n = len(yhat.index) if type(yhat) == pd.Series else len(yhat)
        assert m == n
        mape = []
        for a, f in zip(y, yhat):
            # avoid division by 0
            if f > 1e-6:
                mape.append(np.abs((a - f)/f))
        mape = np.mean(np.array(mape))
        return mape * 100. if perc else mape
    except AssertionError:
        print(f"Wrong dimension for MAPE calculation: y = {m}, yhat = {n}")
        return -1.
        
mape_scorer = make_scorer(mape, greater_is_better=False)