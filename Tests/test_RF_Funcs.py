import pytest
from Functions.RF_Funcs import *
import datetime as dt

import pandas_market_calendars as mcal
nyse = mcal.get_calendar('NYSE')
start_date = '2020-01-01'
end_date = '2023-12-31'
schedule = nyse.schedule(start_date=start_date, end_date=end_date)
trading_days = schedule['market_open'].dt.date.tolist()

data = pd.read_csv('Data/Treasury.csv')
returns_data = pd.read_csv('Data/Return_Data.csv').set_index('date')

pred_RF = pd.read_csv('Data/predictions_RF.csv').set_index('date')
pred_RF_MAE = pd.read_csv('Data/predictions_RF_MAE.csv').set_index('date')
pred_HAR = pd.read_csv('Data/predictions_HAR.csv').set_index('date')

dates = data['date'].iloc[-180:].reset_index(drop=True)

class TestOptionStrategy():
    def test_OptionStrategy_Output(self):
        res = OptionStrategy(0.1954844102369912,dt.date(2022,11,21),trading_days,verbose=False)
        assert res['Return'] == 91.39
        assert res['Signal'] == 1

    def test_OptionStrategy_Input_Exceptions(self):
        with pytest.raises(ValueError):
            OptionStrategy('5',dt.date(2022,11,21),trading_days,verbose=False)
        with pytest.raises(ValueError):
            OptionStrategy(0.1954844102369912,'11-21-23',trading_days,verbose=False)


class TestRunStrategy():
    def test_RunStrategy_Output_HAR(self):
        assert int(RunStrategy(pred_HAR, dates, trading_days,results_data=returns_data,analysis=True)[1][-1]) == 2098
        assert int(RunStrategy(pred_HAR, dates, trading_days,results_data=returns_data,comparison='median',analysis=True)[1][-1]) == 1557

    def test_RunStrategy_Output_RF(self):
        assert int(RunStrategy(pred_RF, dates, trading_days,results_data=returns_data,analysis=True)[1][-1]) == 1325
        assert int(RunStrategy(pred_RF, dates, trading_days,results_data=returns_data,comparison='median',analysis=True)[1][-1]) == 1071

    # def test_RunStrategy_Input(self):
    #     with self.assertRaises(ValueError):
    #         rais
