import pandas as pd
import polygon as pg
import matplotlib.pyplot as plt
import datetime as dt
from secret import polygon_key
from BS_functions import *
import math

def prices_grab(today):
    '''
    Output: {'Stock close': sc, 'Call close':cc, 'Put close': pc}
    '''
    OC = pg.OptionsClient(polygon_key)
    RC = pg.reference_apis.reference_api.ReferenceClient(polygon_key)
    SC = pg.StocksClient(polygon_key)

    #today = dt.date(2023,11,27)
    today_string = today.strftime('%Y-%m-%d')
    tick_date = (today + dt.timedelta(days=1)).strftime('%y%m%d')

    sc = SC.get_daily_open_close('SPY', date=today_string)['close']

    call_strike = str(math.ceil(sc))
    put_strike = str(math.floor(sc))

    try:
        cc = OC.get_daily_open_close('O:SPY'+tick_date+'C00'+call_strike+'000', date=today_string)['close']
        pc = OC.get_daily_open_close('O:SPY'+tick_date+'P00'+put_strike+'000', date=today_string)['close']
    except KeyError:
        return("Data not found.")

    print(f"Stock close: {sc}")
    print(f"Call close: {cc}")
    print(f"Put close: {pc}")

    return {'Stock close': sc, 'Call close':cc, 'Put close': pc}

def IV_grab(today):
    '''
    Output: {'Stock close': sc, 'Call close':cc, 'Put close': pc, 'Avg IV': avg_IV}
    '''
    res = prices_grab(today)
    if res == 'Data not found.':
        return res
    
    sc = res['Stock close']
    cc = res['Call close']
    pc = res['Put close']

    call_IV = GetIV(target_value=cc,S=sc,K=math.ceil(sc),r=.05375,T=1/365,flag='c')['IV']
    put_IV = GetIV(target_value=pc,S=sc,K=math.floor(sc),r=.05375,T=1/365,flag='p')['IV']
    avg_IV = (call_IV + put_IV)/2

    return {'Stock close': sc, 'Call close':cc, 'Put close': pc, 'Avg IV': avg_IV}