import pandas as pd
import polygon as pg
import matplotlib.pyplot as plt
import datetime as dt
from secret import polygon_key
from BS_functions import *
import math

def prices_grab(today):
    OC = pg.OptionsClient(polygon_key)
    RC = pg.reference_apis.reference_api.ReferenceClient(polygon_key)
    SC = pg.StocksClient(polygon_key)

    #today = dt.date(2023,11,27)
    today_string = today.strftime('%Y-%m-%d')
    tick_date = today.strftime('%y%m%d')

    sc = SC.get_daily_open_close('SPY', date=today_string)['close']

    call_strike = str(math.floor(sc))
    put_strike = str(math.ceil(sc))

    cc = OC.get_daily_open_close('O:SPY'+tick_date+'C00'+call_strike+'000', date=today_string)['close']
    pc = OC.get_daily_open_close('O:SPY'+tick_date+'P00'+put_strike+'000', date=today_string)['close']

    print(f"Stock close: {sc}")
    print(f"Call close: {cc}")
    print(f"Put close: {pc}")

    return {'Stock close': sc, 'Call close':cc, 'Put close': pc}