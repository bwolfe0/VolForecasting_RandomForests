from Functions.polygon_scrape import *
from Functions.trading_strategy import *
import pandas as pd
import numpy as np
from secret import polygon_key
import datetime as dt
import polygon as pg
import warnings
warnings.filterwarnings('ignore')

from dotenv import load_dotenv
import os
load_dotenv()
key = os.getenv('polygon_key')

import pandas_market_calendars as mcal
nyse = mcal.get_calendar('NYSE')
start_date = '2019-03-10'
end_date = '2023-08-10'
schedule = nyse.schedule(start_date=start_date, end_date=end_date)
trading_days = schedule['market_open'].dt.date.tolist()

r = pd.read_csv('Data/EFFR.csv')
r.dropna(axis=0, inplace=True)
r.set_index('date',inplace=True)


# See Functions.polygon_scrape.calculate_ATR

ATR_data = []

dates = trading_days.copy()[15:]

for date in dates:
    ATR_data.append(
        calculate_ATR(date,trading_days,14)
    )

pd.DataFrame(ATR_data).to_csv('Data/ATR_data.csv')