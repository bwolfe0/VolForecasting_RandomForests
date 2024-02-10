import pytest
from Functions.polygon_scrape import *

import pandas_market_calendars as mcal
nyse = mcal.get_calendar('NYSE')
start_date = '2020-01-01'
end_date = '2024-2-10'
schedule = nyse.schedule(start_date=start_date, end_date=end_date)
trading_days = schedule['market_open'].dt.date.tolist()


def test_prices_grab_1():
    res = prices_grab(dt.date(2023,11,27),dt.date(2023,11,28))
    assert 454.48 == res['stock_close'] #Test known values
    assert .69 == res['call_close'][0]
    assert .64 == res['put_close'][0]

def test_prices_grab_2():
    res = prices_grab(dt.date(2024,2,5), dt.date(2024,2,6))
    assert 492.55 == res['stock_close']
    assert [0.87, 0.52, 0.29] == res['call_close']
    assert [0.89, 0.54, 0.32] == res['put_close']
    assert [134444, 150264, 140148] == res['call_vols']
    assert [101673, 81431, 86374] == res['put_vols']

def test_IV_grab():
    res = IV_grab(dt.date(2024,2,5),trading_days)
    assert pytest.approx(0.076218, rel=1e-4) == res['avg_IV'] #Test known value

def test_IV_grab_exceptions():
    with pytest.raises(ValueError):
        IV_grab(dt.date(2023,12,3),trading_days)
        IV_grab(5)


# if __name__ == '__main__':
#     unittest.main()