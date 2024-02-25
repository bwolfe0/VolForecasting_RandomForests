from Functions.polygon_scrape import *
from Functions.trading_strategy import *
import pandas as pd
import numpy as np
import datetime as dt
import polygon as pg
import warnings
warnings.filterwarnings('ignore')

import os
from dotenv import load_dotenv
load_dotenv()
key = os.getenv('polygon_key')

import pandas_market_calendars as mcal
nyse = mcal.get_calendar('NYSE')
start_date = '2019-04-1'
end_date = '2023-08-11'
schedule = nyse.schedule(start_date=start_date, end_date=end_date)
trading_days = schedule['market_open'].dt.date.tolist()

r = pd.read_csv('Data/EFFR.csv')
r.dropna(axis=0, inplace=True)
r.set_index('date', inplace=True)

client = pg.RESTClient(key)

dates = trading_days.copy()[:1099]

# Drop 12/23/19, 12/24/19, 12/29/19, and 12/30/19 (missing data)
dates.pop(185)
dates.pop(185)
dates.pop(187)
dates.pop(187)

IV_data = pd.DataFrame(index=dates)

def process_date(date):
    print(date)
    result = {}
    pull = calculate_IV(date, trading_days, r, num_strikes=1)
    result['avg_ATM_IV'] = pull['avg_IV']
    for sd in [-5, 5]:
        pull = calculate_IV(date, trading_days, r, strike_delta=sd, average=False)
        result[f'{sd}c'] = pull['call_IV'][0]
        result[f'{sd}p'] = pull['put_IV'][0]
    return date, result

import concurrent.futures
with concurrent.futures.ThreadPoolExecutor() as executor:
    futures = {executor.submit(process_date, date): date for date in dates}
    for future in concurrent.futures.as_completed(futures):
        date, result = future.result()
        for key, value in result.items():
            IV_data.at[date, key] = value

# Drop more missing values
IV_data.drop(labels=[dt.date(2021,2,5),dt.date(2021,2,17), dt.date(2022,1,7), dt.date(2022,4,12)],axis=0, inplace=True)
IV_data.reset_index(inplace=True)
IV_data.at[124,'5c'] = np.mean(IV_data['5c'])

IV_data.to_csv('Data/IV_Data.csv')