import pandas as pd
from features import create_ts_features, create_lag_features, TargetTransformer
from sklearn.model_selection import train_test_split

class Forecaster:

    def __init__(self, n_steps, step="1H", lags=None):

        """Constructor for the Forecaster object

        Parameters
        ----------
        lags: list of lags used for training the model
        n_steps: number of time periods in the forecasting horizon
        step: forecasting time period given as Pandas time series frequencies
        """

        self.n_steps = n_steps
        self.step = step
        self.lags = lags

    def recursive(self, y, model):
    
        """Multi-step recursive forecasting using the input time 
        series data and a pre-trained machine learning model
        
        Parameters
        ----------
        y: pd.Series holding the input time-series to forecast
        model: an already trained machine learning model implementing the scikit-learn interface
        
        Returns
        -------
        fcast_values: pd.Series with forecasted values indexed by forecast horizon dates 
        """
        
        # get the dates to forecast
        last_date = y.index[-1] + pd.Timedelta(hours=1)
        fcast_range = pd.date_range(last_date, periods=self.n_steps, freq=self.step)

        fcasted_values = []
        target = y.copy()

        for date in fcast_range:
            
            # FIXME: This is not correct, use the latest prediction value from the model instead
            new_point = fcasted_values[-1] if len(fcasted_values) > 0 else 0.0   
            target = target.append(pd.Series(index=[date], data=new_point))

            # forecast
            ts_features = create_ts_features(target)
            if len(self.lags) > 0:
                lags_features = create_lag_features(target, lags=self.lags)
                features = ts_features.join(lags_features, how="outer").dropna()
            else:
                features = ts_features
                
            predictions = model.predict(features)
            fcasted_values.append(predictions[-1])

        return pd.Series(index=fcast_range, data=fcasted_values)

    def direct(self, y, train_fn, params=None):
        
        """Multi-step direct forecasting using a machine learning model
        to forecast each time period ahead
        
        Parameters
        ----------
        y: pd.Series holding the input time-series to forecast
        train_fn: a function for training the model which returns as output the trained model
                  cross-validation score and test score
        params: additional parameters for the training function
        
        Returns
        -------
        fcast_values: pd.Series with forecasted values indexed by forecast horizon dates    
        """
        
        def one_step_features(date, step):

            # features must be obtained using data lagged 
            # by the desired number of steps (the for loop index)
            tmp = y[y.index <= date]       
            lags_features = create_lag_features(tmp, lags=self.lags)
            ts_features = create_ts_features(tmp)
            features = ts_features.join(lags_features, how="outer").dropna()
            
            # build target to be ahead of the features built 
            # by the desired number of steps (the for loop index)
            target = y[y.index >= features.index[0] + pd.Timedelta(hours=step)]
            assert len(features.index) == len(target.index)
            
            return features, target
            
        params = {} if params is None else params
        fcast_values = []
        fcast_range = pd.date_range(y.index[-1] + pd.Timedelta(hours=1), 
                                    periods=self.n_steps, freq=self.step)
        fcast_features, _ = one_step_features(y.index[-1], 0)
                
        for s in range(1, self.n_steps+1):
            
            last_date = y.index[-1] - pd.Timedelta(hours=s)
            features, target = one_step_features(last_date, s)
            
            model, cv_score, test_score = train_fn(features, target, **params) 
                
            # use the model to predict s steps ahead
            predictions = model.predict(fcast_features)        
            fcast_values.append(predictions[-1])
                    
        return pd.Series(index=fcast_range, data=fcast_values)
