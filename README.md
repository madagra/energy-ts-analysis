# Time series analysis on energy consumption data
 
Jupyter notebook implementing time series forecasting of energy consumption data with different techniques:
* Simple and double exponential smoothing
* Holt-Winters
* Facebook Prophet model
* Simple linear regression
* XGBoost model with Bayesan hyperparameters optimization

### Usage

At first make sure to have Python 3.6 or higher and Jupyter notebook installed. 
To run the notebook execute the following commands:

```bash
python -m pip install pipenv
python -m pipenv shell
pipenv install
python -m ipykernel install --user --name=<folder_name>
jupyter notebook
```

Now you should be able to execute the code in the notebook with all related dependencies installed.
