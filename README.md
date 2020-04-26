# Time series forecasting of energy data
 
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

You can also play with the ML forecasting models using the CLI script 
contained in the `src` folder. After having install the dependencies you can execute 
the script with the following command inside the `src` folder:

```
# available commands
python main.py --help

# 48 steps ahead forecast using recursive technique
python main.py --steps 48 --fcast recursive
```
