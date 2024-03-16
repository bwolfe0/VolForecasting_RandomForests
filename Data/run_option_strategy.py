from Functions.polygon_scrape import *
from Functions.trading_strategy import *
import pandas as pd
import numpy as np
from secret import polygon_key
from datetime import datetime, date
import polygon as pg
import warnings
warnings.filterwarnings('ignore')
import time

import pandas_market_calendars as mcal
nyse = mcal.get_calendar('NYSE')
start_date = '2019-01-01'
end_date = '2023-12-31'
schedule = nyse.schedule(start_date=start_date, end_date=end_date)
trading_days = schedule['market_open'].dt.date.tolist()

pred_RF = pd.read_csv('Data/RF_Best_Model_Predictions.csv', index_col=0)
pred_RF['values'] = np.sqrt(pred_RF['values'])*np.sqrt(252)

dates = pd.Series(pred_RF.index)

r = pd.read_csv('Data/EFFR.csv')
r.dropna(axis=0, inplace=True)
r.set_index('date',inplace=True)

results = RunStrategy(pred_RF,dates,trading_days,r,thresh=1.25)