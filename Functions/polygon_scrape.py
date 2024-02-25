import pandas as pd
import polygon as pg
import matplotlib.pyplot as plt
import datetime as dt
from secret import polygon_key
from Functions.BS_functions import *
import math
from typing import Union
import numpy as np

def calculate_IV(date, 
                 trading_days, 
                 r, 
                 num_strikes=1, 
                 average=True,
                 strike_delta=None
                 ):
    """
    Retreive IV data. If average is True, gets an average implied volatility for "date" based on the IV 
    of the 'num_strikes' closest call strikes above the closing price and closest put strikes below with next-day expiration.
    For ex., if num_srikes is 3, takes the averages of three closest call strikes IV and three closest put strikes below.

    Parameters:
    date: The day for which prices are aquired.
    trading_days: A list of valid trading days (see pandas_market_calendars).
    r: Data for the risk free rate of return, often the EFFR for 0DTE options.
    num_strikes: Number of strikes to combine for 'avg_IV'.
    average: Whether to average the IV values, False if desired to retain call/put results separately.
    strike_delta: When average is False, how far to go from ATM strike.

    Returns:
    dict: A dictionary with keys 'stock_close', 'call_close', 'put_close', and 'avg_IV'. If average is False, 
    'avg_IV' key is replaced with 'call_IV' and 'put_IV'.

    Raises:
    ValueError: Date is not in datetime or a suitable string format.
    """
    # Data-type exception handling
    date = to_datetime(date)

    # Ensure date is a valid trading day
    if date not in trading_days:
        raise ValueError(f"Not a valid trading day: {date}")

    # Determine the next trading day
    next_expiry_date = find_next_expiry_date(date,trading_days)

    # Get underlying, call, and put closing prices
    res = pull_prices(date,next_expiry_date,num_strikes,strike_delta=strike_delta)
    
    sc = res['stock_close']
    call_close = res['call_close']
    put_close = res['put_close']
    call_vols = res['call_vols']
    put_vols = res['put_vols']

    # Find IV for both derivatives
    call_IV = [GetIV(target_value=cc,S=sc,K=math.ceil(sc),r=r.loc[f"{date.month}/{date.day}/{date.year % 100}"]['r']/100,T=1/365,flag='c')['IV'] for cc in call_close]
    put_IV = [GetIV(target_value=pc,S=sc,K=math.floor(sc),r=r.loc[f"{date.month}/{date.day}/{date.year % 100}"]['r']/100,T=1/365,flag='p')['IV'] for pc in put_close]

    # If average is True, take volume weighted average of IV values
    if average is True:
        sum = 0
        for j in range(len(call_close)):
            sum += call_IV[j]*call_vols[j] + put_IV[j]*put_vols[j]
        
        avg_IV = sum / (np.sum(call_vols) + np.sum(put_vols))

        res['avg_IV'] = avg_IV

        return res
    
    # If average is False, return call and put IV for the target strike individually
    else:
        return {'date':date, 'strike_delta': strike_delta, 'call_IV': call_IV, 'put_IV': put_IV}

def pull_prices(date,
                next_date,
                num_strikes,
                strike_delta=None
                ):
    """
    Get the close prices of SPY, 'num_strikes' closest call strikes above, and closest puts below with next-day expiration.

    Parameters:
    date: The day for which prices are aquired.
    next_date: The next trading day after date.
    num_strikes: Number of strikes to combine for 'avg_IV'.
    strike_delta: When average is False, how far to go from ATM strike.

    Returns:
    dict: A dictionary with keys 'stock_close', 'stock_close next', 'call_close, and 'put close'.

    Raises:
    LookupError: If an error meessage is present when data is pulled.
    KeyError: If price data for SPY close or either option cannot be found.
    """
    # Create Client Object
    client = pg.RESTClient(polygon_key)

    # Reformat Input
    date_string = date.strftime('%Y-%m-%d')
    next_date_string = next_date.strftime('%Y-%m-%d')
    tick_date = next_date.strftime('%y%m%d')

    # Pull stock close
    stock_pull = client.get_daily_open_close_agg('SPY', date=date_string)
    stock_pull_next = client.get_daily_open_close_agg('SPY', date=next_date_string)

    # Ensure stock data is pulled w/o error
    if hasattr(stock_pull, 'message'):
        raise LookupError(f"{date}: {stock_pull.message}")
    elif hasattr(stock_pull_next, 'message'):
        raise LookupError(f"{date}: {stock_pull_next.message}")

    try:
        sc = stock_pull.close
    except:
        raise KeyError(f"Data not found for {date}")
    try:
        sc_next = stock_pull_next.close
    except:
        raise KeyError(f"Data not found for following day")


    # Determine closest call strike above and put strike below the SPY closing price
    c = math.ceil(sc)
    p = math.floor(sc)
    
    # If 'strike_delta' is None, pull data for 'num_strikes' calls above and puts below the SPY closing price
    if strike_delta is None:
        call_strikes = [str(c+j) for j in range(num_strikes)]
        put_strikes = [str(p-j) for j in range(num_strikes)]
    
    # If 'strike_delta' is not None, pull data for the call and put 'strike_delta' up or down from the SPY closing price.
    elif (num_strikes == 1) and (strike_delta is not None):
        call_strikes = [str(c+strike_delta)]
        put_strikes = call_strikes

    # Pull the data
    call_pull = pull_call_data(tick_date,date_string,client,call_strikes)
    put_pull = pull_put_data(tick_date,date_string,client,put_strikes)

    # Ensure that option data has been pulled
    call_strikes, call_pull, put_strikes, put_pull, adjusted = check_data(date,sc,tick_date,date_string,client,call_strikes,call_pull,put_strikes,put_pull)

    call_close = [cp.close for cp in call_pull]
    call_vols = [cp.volume for cp in call_pull]
    put_close = [pp.close for pp in put_pull]
    put_vols = [pp.volume for pp in put_pull]

    return {'date':date, 'stock_close': sc,'stock_close_next':sc_next, 'strike_delta': strike_delta, 'call_close':call_close, 'put_close': put_close, 
            'call_vols': call_vols, 'put_vols': put_vols, 'strikes_adjusted': adjusted}

def check_data(date,sc,tick_date,date_string,client,call_strikes,call_pull,put_strikes,put_pull):
    go = True
    adjusted = False
    while go is True:
        for j in range(len(call_pull)):
            if hasattr(call_pull[j], 'message'):
                if j == 0: call_strikes = [str(int(cs) + 1) for cs in call_strikes]
                elif j == 1: call_strikes[j:] = [str(int(cs) + 1)  for cs in call_strikes[j:]]
                else: call_strikes[-1] = str(int(call_strikes[-1]) + 1)
                call_pull = pull_call_data(tick_date,date_string,client,call_strikes)
                adjusted = True
            if hasattr(put_pull[j], 'message'):
                if j == 0: put_strikes = [str(int(ps) - 1) for ps in put_strikes]
                elif j == 1: put_strikes[j:] = [str(int(ps) - 1) for ps in put_strikes[j:]]
                else: put_strikes[-1] = str(int(put_strikes[-1]) - 1)
                put_pull = pull_put_data(tick_date,date_string,client,put_strikes)
                adjusted = True
            if (hasattr(call_pull[j], 'message') is False) and (hasattr(put_pull[j], 'message') is False):
                go = False

    if adjusted is True:
        print(f"########## STRIKE ADJUSTMENT ########## \n {date}: The stock closed at {sc} and the final call and put strikes were \n {call_strikes} \n {put_strikes} \n respectively \n ########################## \n")

    return (call_strikes, call_pull, put_strikes, put_pull, adjusted)

def pull_call_data(tick_date,date_string, client, call_strikes):
    try:
        res =  [client.get_daily_open_close_agg('O:SPY'+tick_date+'C00'+cs+'000', date=date_string) for cs in call_strikes]
    except:
        call_strikes =  [str(int(cs) - 1) for cs in call_strikes]
        res = pull_call_data(tick_date,date_string, client, call_strikes)
    return res


def pull_put_data(tick_date,date_string, client, put_strikes):
    try:
        res = [client.get_daily_open_close_agg('O:SPY'+tick_date+'P00'+ps+'000', date=date_string) for ps in put_strikes]
    except:
        put_strikes =  [str(int(ps) - 1) for ps in put_strikes]
        res = pull_put_data(tick_date,date_string, client, put_strikes)
    return res
        
def find_next_expiry_date(date, trading_days):
    # Find the index for the given date in the trading_days list
    date_index = trading_days.index(date)

    # Determine the next valid expiry day based on the day of the week for the given date
    weekday = date.weekday()

    if date >= dt.date(2022,11,14):
        return trading_days[date_index + 1]
    else:
        # Logic to adjust the index based on the current day of the week
        if weekday == 0:  # Monday
                steps_forward = 2  # Target Wednesday
        elif weekday == 2:  # Wednesday
            steps_forward = 2  # Target Friday
        else:  # Tuesday, Thursday, Friday
            steps_forward = 1 

    # Adjusting steps to find the next expiry within the trading days list
    next_expiry_date_index = min(date_index + steps_forward, len(trading_days) - 1)
    
    return trading_days[next_expiry_date_index]


def to_datetime(date):
    """Converts 'date' from a string to a datetime object if needed"""
    if isinstance(date, (dt.date, dt.datetime)) is False:
        try:
            return dt.datetime.strptime(date,'%Y-%m-%d').date()
        except:
            raise ValueError("Input date not in datetime or suitable string format.")
    else:
        return date
    

def calculate_ATR(date, trading_days, lag):
    date_index = trading_days.index(date)
    previous_days = trading_days[(date_index+1-lag):(date_index+1)]

    date_string = date.strftime('%Y-%m-%d')
    previous_days_string = [pd.strftime('%Y-%m-%d') for pd in previous_days]
    TRs = []
    print(date)
    for j in range(lag):
        TRs.append(calculate_ATR_values(date_string,previous_days_string[j]))

    return {'date': date, 'ATR': np.mean(TRs)}


def calculate_ATR_values(date_string, previous_date_string):
    client = pg.RESTClient(polygon_key)
    stock_pull = client.get_daily_open_close_agg('SPY', date=date_string)
    stock_pull_previous= client.get_daily_open_close_agg('SPY', date=previous_date_string)

    H_L = stock_pull.high - stock_pull.low
    H_Cp = abs(stock_pull.high - stock_pull_previous.close)
    L_Cp = abs(stock_pull.low - stock_pull_previous.close)

    return max(H_L, H_Cp, L_Cp)