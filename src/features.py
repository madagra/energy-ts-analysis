import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, scale
from statsmodels.tsa.stattools import pacf
from sklearn.linear_model import LinearRegression


class TargetTransformer:
    """
    Perform some transformation on the time series
    data in order to make the model more performant and
    avoid non-stationary effects.
    """
        
    def __init__(self, log=False, detrend=False, diff=False):
        
        self.trf_log = log
        self.trf_detrend = detrend
        self.trend = pd.Series(dtype=np.float64)
    
    def transform(self, index, values):
        """
        Perform log transformation to the target time series

        :param index: the index for the resulting series
        :param values: the values of the initial series

        Return:
            transformed pd.Series
        """
        res = pd.Series(index=index, data=values)

        if self.trf_detrend:
            self.trend = TargetTransformer.get_trend(res) - np.mean(res.values)
            res = res.subtract(self.trend)
            
        if self.trf_log:
            res = pd.Series(index=index, data=np.log(res.values))
        
        return res
    
    def inverse(self, index, values):
        """
        Go back to the original time series values

        :param index: the index for the resulting series
        :param values: the values of series to be transformed back

        Return:
            inverse transformed pd.Series
        """        
        res = pd.Series(index=index, data=values)
        
        if self.trf_log:
            res = pd.Series(index=index, data=np.exp(values))
        try:
            if self.trf_detrend:
                assert len(res.index) == len(self.trend.index)                
                res = res + self.trend
                
        except AssertionError:
            print("Use a different transformer for each target to transform")
            
        return res
    
    @staticmethod
    def get_trend(data):
        """
        Get the linear trend on the data which makes the time
        series not stationary
        """
        n = len(data.index)
        X = np.reshape(np.arange(0, n), (n, 1))
        y = np.array(data)
        model = LinearRegression()
        model.fit(X, y)
        trend = model.predict(X)
        return pd.Series(index=data.index, data=trend)


def create_lag_features(target, lags=None, thres=0.2):
    
    scaler = StandardScaler()
    features = pd.DataFrame()
                
    if lags is None:
        partial = pd.Series(data=pacf(target, nlags=48))
        lags = list(partial[np.abs(partial) >= thres].index)

    df = pd.DataFrame()
    if 0 in lags:
        lags.remove(0) # do not consider itself as lag feature
    for l in lags:
        df[f"lag_{l}"] = target.shift(l)
        
    features = pd.DataFrame(scaler.fit_transform(df[df.columns]), 
                            columns=df.columns)

    features = df
    features.index = target.index
    
    return features

def create_ts_features(data):
    
    def get_shift(row):
        """
        Factory working shift: 3 shifts per day of 8 hours
        """
        if 6 <= row.hour <= 14:
            return 2
        elif 15 <= row.hour <= 22:
            return 3
        else:
            return 1
    
    features = pd.DataFrame()
    
    features["hour"] = data.index.hour
    features["weekday"] = data.index.weekday
    features["dayofyear"] = data.index.dayofyear
    features["is_weekend"] = data.index.weekday.isin([5, 6]).astype(np.int32)
    features["weekofyear"] = data.index.weekofyear
    features["month"] = data.index.month
    features["season"] = (data.index.month%12 + 3)//3
    features["shift"] = pd.Series(data.index.map(get_shift))
    
    features.index = data.index
        
    return features

def create_endog_features(data, extracted=None):
    
    features = pd.DataFrame()
    
    # energy consuption of the turbine
    features["turbine"] = np.where(data["turbine"] > 500, 1, 0)
    extracted.remove("turbine")   

    # all the rest of the features
    extracted = list(data.columns)
    extracted.remove("turbine")
    for f in extracted:
        features[f] = scale(data[f].values)

    features.index = data.index
    
    return features

def get_features(y, f_lags=True, f_endog=False, lags=None):

    """
    Create the feature set for the time series

    Parameters
    ----------
    y: pd.Series with the target time series
    f_lags: boolean switch for turning on lags features
    f_endog: boolean switch for turning on endogenous features
    lags: optional list of lags to create the lag features for

    Returns
    -------
    features: pd.DataFrame with the feature set
    target: pd.Series holding the target time series with the same index as the features
    """

    features = pd.DataFrame()

    ts = create_ts_features(y)
    features = features.join(ts, how="outer").dropna()

    if f_lags:
        lags = create_lag_features(y, lags=lags, thres=0.2)
        features = features.join(lags, how="outer").dropna()

    if f_endog:
        endog = create_endog_features(y)
        features = features.join(endog, how="outer").dropna()
    
    target = y[y.index >= features.index[0]]

    return features, target