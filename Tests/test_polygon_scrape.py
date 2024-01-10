import sys
sys.path.append(sys.path[0][:-6])

import unittest
from Functions.polygon_scrape import *

import pandas_market_calendars as mcal
nyse = mcal.get_calendar('NYSE')
start_date = '2020-01-01'
end_date = '2023-12-31'
schedule = nyse.schedule(start_date=start_date, end_date=end_date)
trading_days = schedule['market_open'].dt.date.tolist()


class test_poly_scrape(unittest.TestCase):

    def test_prices_grab(self):
        res = prices_grab(dt.date(2023,11,27),dt.date(2023,11,28))
        self.assertEqual(454.48,res['Stock close']) #Test known values
        self.assertEqual(.69,res['Call close'])
        self.assertEqual(.64,res['Put close'])

    def test_IV_grab(self):
        res = IV_grab(dt.date(2023,11,27),trading_days)
        self.assertAlmostEqual(0.094055,res['Avg IV']) #Test known value

    def test_IV_grab_exceptions(self):
        with self.assertRaises(ValueError):
            IV_grab(dt.date(2023,12,3),trading_days)
            IV_grab(5)


if __name__ == '__main__':
    unittest.main()