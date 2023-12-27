import sys
sys.path.append('/Users/benwolfe/Desktop/OneDrive/Python/OptionsData')
import unittest
from poly_scrape_single_day import *


class test_poly_scrape(unittest.TestCase):

    def test_prices_grab(self):
        res = prices_grab(dt.date(2023,11,27))
        self.assertEqual(454.48,res['Stock close'])
        self.assertEqual(.69,res['Call close'])
        self.assertEqual(.64,res['Put close'])

    def test_IV_grab(self):
        res = IV_grab(dt.date(2023,11,27))
        self.assertAlmostEqual(0.094055,res['Avg IV'])

    def test_IV_grab_exceptions(self):
        with self.assertRaises(ValueError):
            IV_grab(dt.date(2023,12,3))
            IV_grab(5)


if __name__ == '__main__':
    unittest.main()