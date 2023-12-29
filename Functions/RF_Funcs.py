import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
import time
import warnings
warnings.filterwarnings('ignore')
from math import ceil, floor

import sys
sys.path.append('/Users/benwolfe/Desktop/OneDrive/Python')
from OptionsData.Functions.polygon_scrape import *


def RollingWindowRF(X,Y,dates,w=300):
    if len(X) != len(Y):
        raise ValueError("X and Y are not the same length")
    
    if len(X) < w:
        raise ValueError("Window size is larger than dataset")
    
    start = time.time()

    feature_importance = pd.DataFrame(index=X.columns, columns=dates[w:])
    predictions = pd.DataFrame(index=['values'],columns=dates[w:])
    
    for t in range(w,len(X)):
        x_train = X.iloc[t-w:t]
        y_train = Y.iloc[t-w:t]

        rf = RandomForestRegressor(n_estimators=200, random_state=10,min_samples_leaf=4, max_features=len(X.columns), 
                                   max_depth=7,min_samples_split=4, n_jobs=-1).fit(x_train,y_train)

        predictions[dates[t]] = rf.predict([X.iloc[t]])[0]
        feature_importance[dates[t]] = rf.feature_importances_

    mse = mean_squared_error(predictions.loc['values'],Y.iloc[w:])
    mape = mean_absolute_percentage_error(predictions.loc['values'],Y.iloc[w:])

    fin = time.time() - start

    return {'predictions': predictions, 'mse': mse, 'mape': mape, 'runtime': fin, 'feature importance': feature_importance}



def GetSignal(model,market):
    if model >= market:
        return(1)
    else:
        return(-1)
    


def GetRatios(sc,cc,pc,avg_IV):
    from scipy.optimize import minimize

    # Objective function
    def objective(x):
        return 0  # Dummy function

    # Constraints
    def constraint1(x):
        return x[0]*cc + x[1]*pc - 100  # x + y = 1

    def constraint2(x):
        return (ceil(sc) + x[0]*cc) - (floor(sc) - x[1]*pc) - (2*sc*avg_IV)/sqrt(365)  # x - y = constant

    # Initial guess
    x0 = [20, 20]

    cons = [{'type': 'eq', 'fun': constraint1},
            {'type': 'eq', 'fun': constraint2}]
    bounds = [(0, None), (0, None)]  # x >= 0, y >= 0

    solution = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons)

    return solution.x
    


def OptionStrategy(model_estimate,date,trading_days):
    result = IV_grab(date,trading_days)

    cc = result['Call close']
    pc = result['Put close']
    sc = result['Stock close']
    sc_next = result['Stock close next']
    market_estimate = result['Avg IV']

    signal = GetSignal(model_estimate,market_estimate)

    x,y = GetRatios(sc,cc,pc,market_estimate)

    #profit = +- (payoff from x calls + payoff from y puts - initial cost)
    payoff_calls = x*max(sc_next - ceil(sc),0)
    payoff_puts = y*max(floor(sc)-sc_next,0)
    initial_cost = (x*cc + y*pc)*np.exp(-.05375/365)

    profit = signal * (payoff_calls + payoff_puts - initial_cost)

    return{'Profit': profit, 'Investment': initial_cost}