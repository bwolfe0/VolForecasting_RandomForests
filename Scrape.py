import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import datetime as dt
import boto3
from io import StringIO
from secret import aws_key, aws_secret

def Scrape(symbol,n_options):
    ticker = yf.Ticker(symbol)
    dates = ticker.options[:n_options]

    df = pd.DataFrame(columns=ticker.option_chain(date=dates[0]).calls.columns)

    #calls
    for d in dates:
        new = ticker.option_chain(date=d).calls
        new['Expiration Date'] = d
        df = pd.concat([df,new])

    df['OptionType'] = 'Call'

    #puts
    for d in dates:
        new = ticker.option_chain(date=d).puts
        new['Expiration Date'] = d
        new['OptionType'] = 'Put'
        df = pd.concat([df,new])

    cols = ['contractSymbol','Expiration Date','OptionType','lastTradeDate','strike','lastPrice','bid','ask','change','percentChange','volume','openInterest',
    'impliedVolatility','inTheMoney','contractSize','currency']
    df = df[cols]

    return(df)


def AWS_Push(df,date=dt.date.today()):
    session = boto3.Session(
        aws_access_key_id=aws_key,
        aws_secret_access_key=aws_secret
    )

    s3 = session.resource('s3')

    bucket_name = 'spyoptions'

    csv_buffer = StringIO()

    df.to_csv(csv_buffer, index=False)

    s3.Object(bucket_name, f'SPY_Scrape_{date.month}_{date.day}_{date.year}.csv').put(Body=csv_buffer.getvalue())