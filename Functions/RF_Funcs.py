import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
import time
import warnings
warnings.filterwarnings('ignore')
from math import ceil, floor
from Functions.polygon_scrape import *


def RollingWindowRF(X,Y,dates,w=300):
    '''
    For timeseries data: fit the previous w days using a random forest model to predict the next day. Repeat for range(w+1,len(X)).
    X: Predictor dataframe.
    Y: Target variable dataframe.
    dates: Series containing dates corresponding to X and Y data.
    '''
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
    '''
    Determine whether to buy or sell the straddle strategy. If the model prediction is higher than the market's, the strategy
    is bought and vice versa.
    '''
    if model >= market:
        return(1)
    else:
        return(-1)
    


def GetRatios(sc,cc,pc,avg_IV):
    '''
    Determine the number of puts and calls to buy such that the option strategy breaks even with the market's estimate for 
    next day's vol.
    sc: Today's SPY closing price.
    cc: Today's closing price for the closest above call strike expiring the next day.
    pc: Today's closing price for the closest below put strike expiring the next day.
    avg_IG: The market's estimate for next day vol based on the average IV of the closest above and below call and put strikes respectively (see IV_grab in polygonscrape.py).
    Output: [#calls, #puts]
    '''
    from scipy.optimize import minimize
    #Dummy function
    def objective(x):
        return 0 

    #calls_cost + puts_cost = 100
    def constraint1(x):
        return x[0]*cc + x[1]*pc - 182*1.036  

    #Distance between intercepts on profit diagram = market volatility estimate
    def constraint2(x):
        return (ceil(sc) + x[0]*cc) - (floor(sc) - x[1]*pc) - (2*sc*avg_IV)/sqrt(365)  

    cons = [{'type': 'eq', 'fun': constraint1},
            {'type': 'eq', 'fun': constraint2}]
    
    # Initial guess
    x0 = [20, 20]

    # x >= 0, y >= 0
    bounds = [(0, None), (0, None)]

    solution = minimize(objective, x0, method='trust-constr', bounds=bounds, constraints=cons)

    return solution.x
    


def OptionStrategy(model_estimate,date,trading_days):
    '''
    Execute (buy or sell) a straddle using the closest call strike above and closest put strike below to SPY's close. The decision
    to buy or sell the strategy is determined by comparing the model estimate to the markets. If the model estimate is higher, the
    straddle is bought and vice versa.
    model_estimate: An estimate for volatility that is purportedly able to "beat the market."
    date: The date for which the strategy would be bought.
    trading_days: A list of relevant trading days.
    '''
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