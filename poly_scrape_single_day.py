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
    tick_date = (today + dt.timedelta(days=1)).strftime('%y%m%d')

    sc = SC.get_daily_open_close('SPY', date=today_string)['close']

    call_strike = str(math.ceil(sc))
    put_strike = str(math.floor(sc))

    call_pull = OC.get_daily_open_close('O:SPY'+tick_date+'C00'+call_strike+'000', date=today_string)
    put_pull = OC.get_daily_open_close('O:SPY'+tick_date+'P00'+put_strike+'000', date=today_string)


    if (call_pull['message'] == 'Data not found.') or (put_pull['message'] == 'Data not found.'):
        raise LookupError(f'Data not found: {today_string}')

    print(f"Stock close: {sc}")
    print(f"Call close: {cc}")
    print(f"Put close: {pc}")

    return {'Stock close': sc, 'Call close':cc, 'Put close': pc}