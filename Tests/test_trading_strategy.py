import pytest
from Functions.trading_strategy import *
import datetime as dt

import pandas_market_calendars as mcal
nyse = mcal.get_calendar('NYSE')
start_date = '2019-01-01'
end_date = '2023-12-31'
schedule = nyse.schedule(start_date=start_date, end_date=end_date)
trading_days = schedule['market_open'].dt.date.tolist()

r = pd.read_csv('Data/EFFR.csv')
r.dropna(axis=0, inplace=True)
r.set_index('date',inplace=True)

data = pd.read_csv('Data/More_RV_Features.csv')

returns_data = pd.read_csv('Testing_Files/Archived_Data/return_data_new.csv').set_index('date')

pred_RF = pd.read_csv('Data/RF_ST_IV_predictions_2_18_24.csv').set_index('date')

dates = data['date'].iloc[-180:].reset_index(drop=True)

class TestOptionStrategy():
    def test_OptionStrategy_Output1(self):
        res = OptionStrategy(0.1954844102369912,dt.date(2022,11,21),trading_days,r,num_strikes=1,verbose=False)
        assert res['Signal'] == 1
        assert res['Return'] == 91.58
        assert pytest.approx(res['num_puts'],rel=1e-4) == 1.0071

    def test_OptionStrategy_Output2(self):
        res = OptionStrategy(0.1954844102369912,dt.date(2022,11,21),trading_days,r,num_strikes=3,verbose=False)
        assert res['Signal'] == 1
        assert res['Return'] == 86.45
        assert pytest.approx(res['num_puts'],rel=1e-4) == 1.0695 

    def test_OptionStrategy_Input_Exceptions(self):
        with pytest.raises(ValueError):
            OptionStrategy('5',dt.date(2022,11,21),trading_days,r,verbose=False)
        with pytest.raises(ValueError):
            OptionStrategy(0.1954844102369912,'11-21-23',trading_days,r,verbose=False)


# class TestRunStrategy():
#     def test_RunStrategy_Output_HAR(self):
#         assert int(RunStrategy(pred_HAR, dates, trading_days,results_data=returns_data,analysis=True)[1][-1]) == 2098
#         assert int(RunStrategy(pred_HAR, dates, trading_days,results_data=returns_data,comparison='median',analysis=True)[1][-1]) == 1557

    def test_RunStrategy_Output_RF(self):
        assert int(RunStrategy(pred_RF, dates, trading_days,results_data=returns_data,analysis=True)[1][-1]) == 1325
        assert int(RunStrategy(pred_RF, dates, trading_days,results_data=returns_data,comparison='median',analysis=True)[1][-1]) == 1071

    # def test_RunStrategy_Input(self):
    #     with self.assertRaises(ValueError):
    #         rais
