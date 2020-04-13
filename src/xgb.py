from functools import partial
import xgboost as xgb
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK, STATUS_FAIL
from sklearn.model_selection import train_test_split

from linear import train_model, test_model


def train_xbg_model(params, X_train, y_train):
    """
    Train a XGBoost model with given input parameters
    
    Parameters
    ----------
    params: dict with parameters to be passed to model constructor
    X_train: pd.DataFrame holding the training features
    y_train: pd.Series holding the training target
    
    Returns
    -------
    Dictionary holding the resulting model, MAPE score and final status of the training
    as required by hyperopt interface    
    """

    n_estimators = int(params["n_estimators"])
    max_depth= int(params["max_depth"])

    model = xgb.XGBRegressor(n_estimators=n_estimators, 
                             max_depth=max_depth, 
                             learning_rate=params["learning_rate"],
                             subsample=params["subsample"])

    res = train_model(model, X_train, y_train,
                      eval_set=[(X_train, y_train.values.ravel())],
                      early_stopping_rounds=50,
                      verbose=False)
    return res

def optimize_xgb_model(X_train, y_train, max_evals=50, verbose=False):
    """
    Run Bayesan optimization to find the optimal XGBoost algorithm
    hyperparameters.
    
    Parameters
    ----------
    X_train: pd.DataFrame with the training set features
    y_train: pd.Series with the training set targets
    max_evals: the maximum number of iterations in the Bayesian optimization method
    verbose: if True print the best output parameters

    Returns
    -------
    best: dict with the best parameters obtained
    trials: a list of hyperopt Trials objects with the history of the optimization
    """
    
    space = {
        "n_estimators": hp.quniform("n_estimators", 100, 1000, 10),
        "max_depth": hp.quniform("max_depth", 1, 8, 1),
        "learning_rate": hp.loguniform("learning_rate", -5, 1),
        "subsample": hp.uniform("subsample", 0.8, 1),
        "gamma": hp.quniform("gamma", 0, 100, 1)
    }

    objective_fn = partial(train_xbg_model, 
                           X_train=X_train, 
                           y_train=y_train)
    
    trials = Trials()
    best = fmin(fn=objective_fn,
                space=space,
                algo=tpe.suggest,
                max_evals=max_evals,
                trials=trials)

    if verbose:
        print(f"""
        Best parameters:
            learning_rate: {best["learning_rate"]} 
            n_estimators: {best["n_estimators"]}
            max_depth: {best["max_depth"]}
            sub_sample: {best["subsample"]}
            gamma: {best["gamma"]}
        """)

    return best, trials


def xgboost_model(features, target, test_size=0.2, max_evals=10):
    """
    Full training and testing pipeline for XGBoost ML model with
    hyperparameter optimization using Bayesian method

    Parameters
    ----------
    features: pd.DataFrame holding the model features
    target: pd.Series holding the target values
    max_evals: maximum number of iterations in the optimization procedure

    Returns
    -------
    model: the optimized XGBoost model
    cv_score: the average RMSE coming from cross-validation
    test_score: the RMSE on the test set
    """

    X_train, X_test, y_train, y_test = train_test_split(features, 
                                                        target, 
                                                        test_size=test_size,
                                                        shuffle=False)

    best, trials = optimize_xgb_model(X_train, y_train, max_evals=max_evals)
    res = train_xbg_model(best, X_train, y_train)
    model = res["model"]
    cv_score = min([f["loss"] for f in trials.results if f["status"] == STATUS_OK])
    test_score = test_model(model, X_test, y_test)
    return model, cv_score, test_score