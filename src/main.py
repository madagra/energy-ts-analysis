import sys

from utils import get_dataset, forecast_split, mape
from forecaster import Forecaster
from features import get_features
from xgb import xgboost_model
from linear import linear_model
import argparse


def input_cmd():
    parser = argparse.ArgumentParser()

    parser.add_argument("-s", "--steps",
                        type=int,
                        default=48,
                        help="The number of forecasting steps")

    parser.add_argument("-f", "--fcast", 
                        type=str,
                        default="recursive", choices=["direct", "recursive"],
                        help="The type of forecasting")

    parser.add_argument("-l", "--load", 
                        type=str,
                        default="C4", choices=["C1", "C2", "C3", "C4"],
                        help="The load to predict")

    args = parser.parse_args()
    return args

def ts_forecasting():

    args = input_cmd()

    # get energy consumption data
    load = args.load
    f_steps = args.steps

    data = get_dataset(load_to_predict=load)
    
    c_target = data["energy"]
    t_target, f_target, fcast_range = forecast_split(c_target, n_steps=f_steps)

    # ML methods
    features, target = get_features(t_target)
    lags = [int(f.split("_")[1]) for f in features if "lag" in f]
    forecaster = Forecaster(f_steps, lags=lags)

    print("Forecast with Linear Regression model")
    model, cv_score, test_score = linear_model(features, target)

    if args.fcast == "direct":
        fcast_linear = forecaster.direct(t_target, linear_model)
    elif args.fcast == "recursive":
        fcast_linear = forecaster.recursive(t_target, model)

    fcast_score = mape(f_target, fcast_linear)
    print(f"""
Linear Regression scores
--------------
Cross-validation MAPE: {round(cv_score, 2)}%
Test MAPE: {round(test_score, 2)}%
Direct Forecast MAPE: {round(fcast_score, 2)}%
    """)

    print("Forecast with XGBoost model")
    model, cv_score, test_score = xgboost_model(features, target, max_evals=25)

    if args.fcast == "direct":
        fcast_xgb = forecaster.direct(t_target, xgboost_model)
    elif args.fcast == "recursive":
        fcast_xgb = forecaster.recursive(t_target, model)

    fcast_score = mape(f_target, fcast_xgb)
    print(f"""
XGBoost scores
--------------
Cross-validation MAPE: {round(cv_score, 2)}%
Test MAPE: {round(test_score, 2)}%
Recursive Forecast MAPE: {round(fcast_score, 2)}%
    """)


if __name__ == "__main__":
    sys.exit(ts_forecasting())