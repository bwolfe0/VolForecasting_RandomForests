import pandas as pd
import polygon as pg
import matplotlib.pyplot as plt
import datetime as dt
from secret import polygon_key
from Functions.BS_functions import *
import math
from typing import Union

def IV_grab(date, trading_days):
    """
    Get an average implied volatility at close for "date" based on the IV of the closest call strike above the closing price 
    and closest put price below with next-day expiration.

    Parameters:
    date: The day for which prices are aquired.
    trading_days: A list of valid trading days (see pandas_market_calendars).

    Returns:
    dict: A dictionary with keys 'stock_close', 'call_close', 'put_close', and 'avg_IV'.

    Raises:
    ValueError: Date is not in datetime or a suitable string format.
    """
    #Data-type exception handling
    date = to_datetime(date)
        
    #Trading Days Exception Handling
    if date not in trading_days:
        raise ValueError(f"Not a valid trading day: {date}")
    
    next_date = trading_days[trading_days.index(date)+1]

    #Get underlying, call, and put closing prices
    res = prices_grab(date,next_date)
    
    sc = res['stock_close']
    cc = res['call_close']
    pc = res['put_close']

    #Find IV for both derivatives, find average
    call_IV = GetIV(target_value=cc,S=sc,K=math.ceil(sc),r=.05375,T=1/365,flag='c')['IV']
    put_IV = GetIV(target_value=pc,S=sc,K=math.floor(sc),r=.05375,T=1/365,flag='p')['IV']
    avg_IV = (call_IV + put_IV)/2

    res['avg_IV'] = avg_IV

    return res

def prices_grab(date,next_date):
    """
    Get the close prices of SPY, closest call strike above, and closest put below with next-day expiration.

    Parameters:
    date: The day for which prices are aquired.
    next_date: The next trading day after date.

    Returns:
    dict: A dictionary with keys 'stock_close', 'stock_close next', 'call_close, and 'put close'.

    Raises:
    LookupError: If an error meessage is present when data is pulled.
    KeyError: If price data for SPY close or either option cannot be found.
    """
    #Create Client Objects
    OC = pg.OptionsClient(polygon_key)
    SC = pg.StocksClient(polygon_key)

    #Reformat Input
    date_string = date.strftime('%Y-%m-%d')
    next_date_string = next_date.strftime('%Y-%m-%d')
    tick_date = next_date.strftime('%y%m%d')

    #Pull stock close
    stock_pull = SC.get_daily_open_close('SPY', date=date_string)
    stock_pull_next = SC.get_daily_open_close('SPY', date=next_date_string)

    #Ensure stock data is pulled w/o error
    if 'message' in stock_pull:
        raise LookupError(f"{stock_pull['message']}")
    elif 'message' in stock_pull_next:
        raise LookupError(f"{stock_pull_next['message']}")

    try:
        sc = stock_pull['close']
    except:
        raise KeyError(f"Data not found for 'date'")
    try:
        sc_next = stock_pull_next['close']
    except:
        raise KeyError(f"Data not found for following day")


    #Pull close of higher strike call and lower strike put relative to stock close
    call_strike = str(math.ceil(sc))
    put_strike = str(math.floor(sc))

    call_pull = OC.get_daily_open_close('O:SPY'+tick_date+'C00'+call_strike+'000', date=date_string)
    put_pull = OC.get_daily_open_close('O:SPY'+tick_date+'P00'+put_strike+'000', date=date_string)

    #Ensure that option data has been pulled
    if 'message' in call_pull:
        raise LookupError(f"{call_pull['message']}")
    if 'message' in put_pull:
        raise LookupError(f"{put_pull['message']}")

    #Ensure that 'close' index exists
    try:
        cc = call_pull['close']
        pc = put_pull['close']
    except KeyError:
        raise LookupError("Data not found.")

    return {'stock_close': sc,'stock_close_next':sc_next, 'call_close':cc, 'put_close': pc}


def to_datetime(date):
    """Converts 'date' from a string to a datetime object if needed"""
    if isinstance(date, (dt.date, dt.datetime)) is False:
        try:
            return dt.datetime.strptime(date,'%Y-%m-%d')
        except:
            raise ValueError("Input date not in datetime or suitable string format.")
    else:
        return date