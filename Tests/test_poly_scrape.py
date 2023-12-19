import sys
sys.path.append('/Users/benwolfe/Desktop/OneDrive/Python/OptionsData')
import unittest
from poly_scrape_single_day import *


class test_poly_scrape(unittest.TestCase):

    def test_prices_grab(self):
        res = prices_grab(dt.date(2023,11,27))
        self.assertEqual(454.48,res['Stock close'])
        self.assertEqual(.54,res['Call close'])
        self.assertEqual(.46,res['Put close'])


if __name__ == '__main__':
    unittest.main()