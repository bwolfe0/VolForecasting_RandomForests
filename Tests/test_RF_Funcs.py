import sys
sys.path.append(sys.path[0][:-6])

import unittest
from Functions.RF_Funcs import *
import datetime as dt

import pandas_market_calendars as mcal
nyse = mcal.get_calendar('NYSE')
start_date = '2020-01-01'
end_date = '2023-12-31'
schedule = nyse.schedule(start_date=start_date, end_date=end_date)
trading_days = schedule['market_open'].dt.date.tolist()

res = OptionStrategy(0.1954844102369912,dt.date(2022,11,21),trading_days,verbose=False)


class test_RF_Funcs(unittest.TestCase):
    def test_OptionStrategy(self):
        self.assertEqual(res['Return'], '91.39%')
        #self.assertEqual(res['# Puts'], 1.009)


if __name__ == '__main__':
    unittest.main()