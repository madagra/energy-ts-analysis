import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import (TimeSeriesSplit, train_test_split, 
                                     cross_val_score)
from hyperopt import STATUS_OK, STATUS_FAIL

from utils import mape_scorer, mape


def test_model(model, X_test, y_test):
    """
    Get the RMSE for a given model on a test dataset

    Parameters
    ----------
    model: a model implementing the standard scikit-learn interface
    X_test: pd.DataFrame holding the features of the test set
    y_test: pd.Series holding the test set target

    Returns
    -------
    test_score: the RMSE on the test dataset
    """
    
    predictions = model.predict(X_test)
    test_score = mape(y_test.values, predictions)
    return test_score


def train_model(model, X_train, y_train, **fit_parameters):
    """
    Train a model with time series cross-validation by returning
    the right dictionary to be used by hyperopt for optimization

    Parameters
    ----------
    model: a model implementing the standard scikit-learn interface
    X_train: pd.DataFrame holding the training features
    y_train: pd.Series holding the training target
    fit_parameters: dict with parameters to pass to the model fit function
    Returns
    -------
    Dictionary holding the resulting model, RMSE and final status of the training
    as required by hyperopt interface
    """

    try:

        model.fit(X_train, y_train, **fit_parameters)

        # cross validate using the right iterator for time series
        cv_space = TimeSeriesSplit(n_splits=5)
        cv_score = cross_val_score(model, 
                                   X_train, 
                                   y_train.values.ravel(), 
                                   cv=cv_space, 
                                   scoring=mape_scorer)

        mape = np.mean(np.abs(cv_score))
        return {
            "loss": mape,
            "status": STATUS_OK,
            "model": model
        }
        
    except ValueError as ex:
        return {
            "error": ex,
            "status": STATUS_FAIL
        }

def linear_model(features, target, model_cls=LinearRegression, test_size=0.2, params=None):
    """
    Full training and testing pipeline for linear regression model
    without hyperparameter optimization

    Parameters
    ----------
    features: pd.DataFrame holding the model features
    target: pd.Series holding the target values
    model_cls: model constructor from scikit-learn    
    test_size: the train/test split size
    params: additional parameters to pass to model constructor

    Returns
    -------
    model: the optimized linear regression model
    cv_score: the average RMSE coming from cross-validation
    test_score: the RMSE on the test set
    """

    X_train, X_test, y_train, y_test = train_test_split(features, 
                                                        target, 
                                                        test_size=test_size,
                                                        shuffle=False)
    params = params if params is not None else {}
    model = model_cls(**params)
    res = train_model(model, X_train, y_train)
    cv_score = res["loss"]
    test_score = test_model(res["model"], X_test, y_test)
    return model, cv_score, test_score
    