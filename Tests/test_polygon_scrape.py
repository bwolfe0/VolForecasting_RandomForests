import pytest
from Functions.polygon_scrape import *

import pandas_market_calendars as mcal
nyse = mcal.get_calendar('NYSE')
start_date = '2020-01-01'
end_date = '2023-12-31'
schedule = nyse.schedule(start_date=start_date, end_date=end_date)
trading_days = schedule['market_open'].dt.date.tolist()


def test_prices_grab():
    res = prices_grab(dt.date(2023,11,27),dt.date(2023,11,28))
    assert 454.48 == res['stock_close'] #Test known values
    assert .69 == res['call_close']
    assert .64 == res['put_close']

def test_IV_grab():
    res = IV_grab(dt.date(2023,11,27),trading_days)
    assert pytest.approx(0.094055, rel=1e-4) == res['avg_IV'] #Test known value

def test_IV_grab_exceptions():
    with pytest.raises(ValueError):
        IV_grab(dt.date(2023,12,3),trading_days)
        IV_grab(5)


# if __name__ == '__main__':
#     unittest.main()