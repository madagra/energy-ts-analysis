
## ML time series forecasting the right way

## Introduction

The application of machine learning (ML) techniques to time series forecasting is not [straightforward](https://towardsdatascience.com/how-not-to-use-machine-learning-for-time-series-forecasting-avoiding-the-pitfalls-19f9d7adf424). One of the main challenges is to use the ML model for actually predicting the future in what is commonly referred to as *forecasting*. Without forecasting, time series analysis becomes irrelevant.

![Photo by [Aron Visuals](https://unsplash.com/@aronvisuals?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)](https://cdn-images-1.medium.com/max/12000/0*IIy6L9G8f8wQ2JxT)

This issue stems from the temporal structure of the data since, at variance with standard ML projects, it is not enough to apply a pretrained model on new data points to get the forecasts but, as we will see in this post, additional steps are required. **Very** **few examples of time series forecasting with ML available online are really end-to-end**, since they keep the focus on testing the model on available data and overlook the forecasting part.

This post shows instead a complete example — from feature engineering to forecasting the future using different strategies — of a common problem such as the prediction of the energy consumption of a utility (for instance a transformer) within a factory. Before diving in, let us briefly clarify some terminology.

### Time series analysis

Time series analysis is a broad domain which has been applied to many different problems, ranging from econometric to earthquakes and weather predictions. Broadly speaking, time series methods can be divided into two categories depending on the desired outcome:

* **Time series forecasting**: forecasting is the most common practice in time series analysis. Given a time series these techniques aim at predicting future values with a certain confidence interval using a computational model.

* **Time series classification**: given a time series, these techniques aim at extracting relevant aggregated features from the time series in order to determine whether it belongs to a certain class. An example of time series classification is the analysis of ECG recordings to understand whether are taken from a healthy or sick patient. These methods are more akin to standard ML classification.

In this post we are dealing only with time series forecasting. We restrict our problem to forecast a *univariate time series *performing a *multi-step forecasting* of several steps ahead in the future.

## The dataset

The dataset we consider consists of *hourly energy consumption rates* in kWh for an industrial utility over a period of around 7 months, from July 2019 to January 2020. The energy measurements have been taken from a sensor installed in a real factory in Italy. Data are anonymized by random constant multiplication. Let us plot different views of our data.

![Different views of the dataset. From top to bottom clockwise we have the original time series, the distribution of its values, autocorrelation and partial autocorrelation functions.](https://cdn-images-1.medium.com/max/2914/1*Omg5gSdtkdZ-K5guwBYdkw.png)

In the top left figure above we see the original behavior over time of the energy consumption. The p-value of the augmented Dick-Fuller (ADF) test, shown in the figure title, is a good indication that the time series is stationary.

The top right panel shows the value distribution of the energy consumption. The data values seems to follow a slow-decaying Poisson distribution where high values of the energy consumption above 50 KWh are much less likely than lower ones (10 to 40 KWh). This is an expected behavior since the energy consumption of industrial utilities operating continuously has usually a fairly constant average with sporadic peaks due to production increase or maintenance windows.

The bottom figures show the autocorrelation (ACF) and partial autocorrelation (PACF) functions. See [here](https://dzone.com/articles/autocorrelation-in-time-series-data) for more details on their importance for time series analysis. The ACF shows that the time series has a seasonal component at 24 hours (grey horizontal lines) with autocorrelation peaks which do not decrease over time, indicating the strength of this component. This is confirmed by the PACF.

## Machine learning forecast with XGBoost

For our forecasting problem we choose the XGBoost algorithm using [this](https://xgboost.readthedocs.io/en/latest/get_started.html) popular Python implementation. XGBoost is fast and accurate compared to other tree-based ML methods for time series problems, as shown by several Kaggle competitions and other works available online (see for instance [here](https://filip-wojcik.com/talks/xgboost_forecasting_eng.pdf) or [here](https://towardsdatascience.com/forecasting-stock-prices-using-xgboost-a-detailed-walk-through-7817c1ff536a)).

In order to transform a time series forecasting task into a supervised machine learning problem we need to generate the features. The actual time series values are instead used as target for the ML model. The types of features we can consider for a time series are divided into 3 categories:

* **lag features**: they use the original time series itself as feature with a certain shift usually called *lag*. Lags can be chosen automatically looking at the values of the partial autocorrelation function. In particular, we take as features only the lags where the PACF is greater than 0.2, equivalent to a 5% relevance for the lag. Here the Python snippet to create these features.

```python

import pandas as pd
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import pacf

def create_lag_features(y):
   
    scaler = StandardScaler()
    features = pd.DataFrame()
    
    partial = pd.Series(data=pacf(y, nlags=48))
    lags = list(partial[np.abs(partial) >= 0.2].index)
    
    df = pd.DataFrame()
    
    # avoid to insert the time series itself
    lags.remove(0)
    
    for l in lags:
        df[f"lag_{l}"] = y.shift(l)
    
    features = pd.DataFrame(scaler.fit_transform(df[df.columns]),
                            columns=df.columns)
    features.index = y.index
    
    return features
```

* **standard time series features**: standard time series features such as hour, weekend, season and so on. Using some domain knowledge on the data we can also build additional time series features such as, in our case, the current worker shift per each timestamp.

* **endogenous features**: endogenous features are themselves time series and are external, i.e. they do not depend on the time series to be forecast. In our case an endogenous variable could be the external temperature. In this post we do not consider them.

We use a standard pipeline for training and testing the XGBoost model with 80/20 train test split, [cross-validation with rolling time window](https://stats.stackexchange.com/a/268847) and hyperparameters optimization using the [hyperopt](https://hyperopt.github.io/hyperopt/) library. The best model yields a mean average percentage error (MAPE) of 14.302 from cross-validation and 8.378 when predicting on the much smaller test set. The predicted values on the test set and corresponding importance coefficients for the top features are shown below.

![XGBoost model predictions on the test with correspondent feature importance histogram.](https://cdn-images-1.medium.com/max/2000/1*PlQtvgaAyLrbkwUJipm69A.png)

## Multi-step ahead forecasting with ML models

The ML model results are acceptable and our job is done. Not quite! We mentioned in the introduction that getting satisfactory results from the model is not enough to obtain an actionable outcome. Similarly to standard supervised ML problems, one wants to use the model on unknown data points which in our case are *future time periods*: **ultimately we want to predict the future. The number of time periods to forecast into the future is usually referred to as *forecasting horizon*.

At variance with standard ML where the model can be directly applied to new data, the temporal structure of time series problems makes forecasting more complicated. There are several strategies to achieve multi-step ahead forecasting, a good introduction to these methods can be found [here](https://dzone.com/articles/lessons-learnt-while-solving-time-series-forecasti-1). In this post we compare two of the most widely used strategies showing also their implementation for the ML model discussed above. We choose a forecasting horizon of 48 time periods which corresponds to a *2 days ahead forecasting*.

### Recursive strategy

The recursive forecasting strategy uses only a single model pre-trained for one-step ahead forecasting. *At each forecasting step, this model is used to predict one-step ahead and the value obtained from the forecasting is then fed into the same model to predict the following step* (similarly to a recursive function, hence the name) and so on and so forth until the desired forecasting horizon is reached. The recursive strategy is the least expensive method for forecasting since it uses the same model. However, it is not the best strategy for long time horizons since it accumulates forecasting errors at each step. This problem arises because previous forecast values are used to build the feature space for the current step.

The following snippet implements recursive strategy for our ML model.

 ```python
 def recursive_forecast(y, model, lags, 
                       n_steps=FCAST_STEPS, step="1H"):
    
    """
    Parameters
    ----------
    y: pd.Series holding the input time-series to forecast
    model: pre-trained machine learning model
    lags: list of lags used for training the model
    n_steps: number of time periods in the forecasting horizon
    step: forecasting time period
    
    Returns
    -------
    fcast_values: pd.Series with forecasted values 
    """
    
    # get the dates to forecast
    last_date = y.index[-1] + pd.Timedelta(hours=1)
    fcast_range = pd.date_range(last_date, 
                                periods=n_steps, 
                                freq=step)
    fcasted_values = []
    target = y.copy()
    
    for date in fcast_range:
      
        # build target time series using previously forecast value
        new_point = fcasted_values[-1] if len(fcasted_values) > 0 else 0.0   
        target = target.append(pd.Series(index=[date], data=new_point))
        
        # build feature vector using previous forecast values
        features = create_ts_features(target, lags=lags)
        
        # forecast
        predictions = model.predict(features)
        fcasted_values.append(predictions[-1])
        
    return pd.Series(index=fcast_range, data=fcasted_values)
 ```

### Direct strategy

The direct forecasting strategy uses instead a different ML model for each forecasting step. More specifically, *each model is trained using as target the time series shifted of the desired number of time periods into the future*. Imagine for example that one wants to train a 4 steps ahead model. In this case each timestamp in the target time series is chosen 4 steps ahead with respect to the corresponding timestamp in the feature set. In this way we create a model trained to predict 4 steps ahead into the future. The same procedure is repeated for all forecasting steps. The size of the training/cross-validation set slightly decreases for each additional step ahead to perform.

Direct method does not generate error accumulation at each forecast, but it has larger computational cost making it not suitable for large forecasting horizons. Moreover it cannot model statistical relationships among predictions since the models used for each time step are independent.

And here the implementation for our ML model.

```python
def direct_forecast(y, model, lags, 
                    n_steps=FCAST_STEPS, step="1H"):
    
    """    
    Parameters
    ----------
    y: pd.Series holding the input time-series to forecast
    model: a ML model not trained
    lags: list of lags used for training the model
    n_steps: how many steps forecast into the future
    step: the period of forecasting
    Returns
    -------
    fcast_values: pd.Series with forecasted values 
    fcast_scores: list of the MAPE errors per each time step   
    """
    
    def one_step_features(date, step):
      
        # build standard features
        tmp = y[y.index <= date]
        features = create_ts_features(tmp, lags=lags)
        
        # build target to be ahead of the features by the desired number of steps
        target = y[y.index >= features.index[0] + pd.Timedelta(hours=step)]
        assert len(features.index) == len(target.index)
        
        return features, target
        
    fcast_values = []
    fcast_scores = []
    fcast_range = pd.date_range(y.index[-1] + pd.Timedelta(hours=1), 
                                periods=n_steps, freq=step)
    
    fcast_features, _ = one_step_features(y.index[-1], 0, y)        
    for s in range(1, n_steps+1):
        
        last_date = y.index[-1] - pd.Timedelta(hours=s)
        features, target = one_step_features(last_date, s)
        
        # train XGBoost model for the current forecasting step
        X_train, X_test, y_train, y_test = \
        train_test_split(features, 
                         target, 
                         test_size=0.2,
                         shuffle=False) 
                
        # train the model and use it to predict the current time period
        model.train(X_train, y_train)
        test_predictions = model.predict(X_test)
        test_score = mape(y_test, test_predictions)
        fcast_scores.append(test_score)
        
        # forecast
        predictions = model.predict(fcast_features)        
        fcast_values.append(predictions[-1])
      
    result = pd.Series(index=fcast_range, data=fcast_values)          
    return result, fcast_scores
```
 
## Forecasting Results

And finally let us look at how the two strategies above compare in forecasting the future. We consider the last 48 hours of our dataset as the future so that we have real data to compare the forecasting with. Beware that in real-world problems the future is not known in advance!

By looking at the MAPE score for the two methods, the direct one performs better than the recursive strategy as expected, with an error around 20% lower.

| Strategy          | Cross\-validation set | Test set | Forecast |
|-------------------|:----------------------------:|:---------------:|:---------------:|
| Direct    | 14.302                     | 8.378         | 13.124       |
| Recursive | 14.302                     | 8.378         | 16.471       |


Just looking at these values is not enough to assess forecasting quality. Let us plot the two forecasts.

![](https://cdn-images-1.medium.com/max/2000/1*l-6GbySz51RiQki38QpJZw.png)

For this relatively short horizon both strategies perform similarly for the initial forecasting steps. After around 24 hours, however, the recursive strategy starts to diverge from the direct one. This is most likely due to the error accumulation starting to affect recursive method results. Error accumulation likely generates also the stronger fluctuations displayed by the recursive forecast. The improvement yielded by the direct strategy is however not so significant to always justify its use. In production deployments of time series models it is often better to use a less accurate method requiring much lower computational effort such as the recursive strategy.

In conclusion, remember that the present methods are just basic forecasting strategies. Extensive research effort has been put into devising more precise and controlled forecasting techniques, one example being the Rectify strategy described [here](https://robjhyndman.com/papers/rectify.pdf).

If you enjoyed this post and you have any comment, suggestion, critics just connect with us [here](http://www.linkedin.com/in/mariodagrada) or [here](https://www.linkedin.com/in/lorenzo-ghiringhello-a5235414b). The full code used for this post is available on [Github](https://github.com/madagra/energy-ts-analysis).

