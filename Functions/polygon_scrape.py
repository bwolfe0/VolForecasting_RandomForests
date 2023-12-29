import pandas as pd
import polygon as pg
import matplotlib.pyplot as plt
import datetime as dt
from secret import polygon_key
from Functions.BS_functions import *
import math

def prices_grab(date,next_date):
    '''
    Get the close prices of SPY, closest call strike above, and closest put below with next-day expiration.
    Date: The day for which prices are aquired.
    Output: {'Stock close': sc, 'Call close':cc, 'Put close': pc}
    '''
    #Create Client Objects
    OC = pg.OptionsClient(polygon_key)
    RC = pg.reference_apis.reference_api.ReferenceClient(polygon_key)
    SC = pg.StocksClient(polygon_key)

    #Reformat Input
    date_string = date.strftime('%Y-%m-%d')
    next_date_string = next_date.strftime('%Y-%m-%d')
    tick_date = next_date.strftime('%y%m%d')

    #Pull stock close
    stock_pull = SC.get_daily_open_close('SPY', date=date_string)
    stock_pull_next = SC.get_daily_open_close('SPY', date=next_date_string)

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

    #Ensure that data has been pulled
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

    return {'Stock close': sc,'Stock close next':sc_next, 'Call close':cc, 'Put close': pc}




def IV_grab(date, trading_days):
    '''
    Get an average implied volatility at close for "date" based on the IV of the closest call strike above the closing price 
    and closest put price below with next-day expiration.. 
    Date: The day for which prices are aquired.
    Output: {'Stock close': sc, 'Call close':cc, 'Put close': pc, 'Avg IV': avg_IV}
    '''
    #Data-type exception handling
    if isinstance(date, (dt.date, dt.datetime)) is False:
        try:
            date = dt.datetime.strptime(date,'%Y-%m-%d')
        except:
            raise ValueError("Input date not in datetime or suitable string format.")
        
    #Trading Days Exception Handling
    if date not in trading_days:
        raise ValueError("Not a valid trading day")
    
    next_date = trading_days[trading_days.index(date)+1]

    #Get underlying, call, and put closing prices
    res = prices_grab(date,next_date)
    
    sc = res['Stock close']
    cc = res['Call close']
    pc = res['Put close']

    #Find IV for both derivatives, find average
    call_IV = GetIV(target_value=cc,S=sc,K=math.ceil(sc),r=.05375,T=1/365,flag='c')['IV']
    put_IV = GetIV(target_value=pc,S=sc,K=math.floor(sc),r=.05375,T=1/365,flag='p')['IV']
    avg_IV = (call_IV + put_IV)/2

    res['Avg IV'] = avg_IV

    return res