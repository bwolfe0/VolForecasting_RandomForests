import unittest
from Functions.polygon_scrape import *


class test_poly_scrape(unittest.TestCase):

    def test_prices_grab(self):
        res = prices_grab(dt.date(2023,11,27))
        self.assertEqual(454.48,res['Stock close']) #Test known values
        self.assertEqual(.69,res['Call close'])
        self.assertEqual(.64,res['Put close'])

    def test_IV_grab(self):
        res = IV_grab(dt.date(2023,11,27))
        self.assertAlmostEqual(0.094055,res['Avg IV']) #Test known value

    def test_IV_grab_exceptions(self):
        with self.assertRaises(ValueError):
            IV_grab(dt.date(2023,12,3))
            IV_grab(5)


if __name__ == '__main__':
    unittest.main()