import numpy as np
import pandas as pd
import datetime as dt

def moving_average(df, days):
    for ma in days:
        new_col = "%s MA" %(str(ma))
        df.loc[:, new_col] = pd.DataFrame.rolling(df['Close'], ma).mean()

def calc_max(df, days):
    for ma in days:
        new_col = "Max over %s days:" %(str(ma))
        df.loc[:,new_col] = pd.DataFrame.rolling(df['Close'], ma).max()

def calc_min(df, days):
    for ma in days:
        new_col = "Min over %s days:" %(str(ma))
        df.loc[:,new_col] = pd.DataFrame.rolling(df['Close'], ma).min()

# import data sets and align all columns
tsla_df = pd.read_csv("./TSLA.csv")
aapl_df = pd.read_csv("./AAPL.csv")
amzn_df = pd.read_csv("./AMZN.csv")
msft_df = pd.read_csv("./MSFT.csv")
fb_df = pd.read_csv("./FB.csv")

# make all column names the same
aapl_df = aapl_df.rename({'Adjusted Close': 'Adj Close'}, axis=1)
amzn_df = amzn_df.rename({'Adjusted Close': 'Adj Close'}, axis=1)
msft_df = msft_df.rename({'Adjusted Close': 'Adj Close'}, axis=1)
fb_df = fb_df.rename({'Adjusted Close': 'Adj Close'}, axis=1)

# align all columns in the data sets
tsla_df = tsla_df[["Date", "Low", "Open", "Volume", "High", "Close", "Adj Close"]]
aapl_df = aapl_df[["Date", "Low", "Open", "Volume", "High", "Close", "Adj Close"]]
amzn_df = amzn_df[["Date", "Low", "Open", "Volume", "High", "Close", "Adj Close"]]
msft_df = msft_df[["Date", "Low", "Open", "Volume", "High", "Close", "Adj Close"]]
fb_df = fb_df[["Date", "Low", "Open", "Volume", "High", "Close", "Adj Close"]]

# convert the date column to datetime
tsla_df['Date'] = pd.to_datetime(tsla_df['Date'])
aapl_df['Date'] = pd.to_datetime(aapl_df['Date'])
amzn_df['Date'] = pd.to_datetime(amzn_df['Date'])
msft_df['Date'] = pd.to_datetime(msft_df['Date'])
fb_df['Date'] = pd.to_datetime(fb_df['Date'])

# delete all data prior to 1-1-2016
tsla_df = tsla_df[~(tsla_df['Date'] <= '2016-1-1')]
aapl_df = aapl_df[~(aapl_df['Date'] <= '2016-1-1')]
amzn_df = amzn_df[~(amzn_df['Date'] <= '2016-1-1')]
msft_df = msft_df[~(msft_df['Date'] <= '2016-1-1')]
fb_df = fb_df[~(fb_df['Date'] <= '2016-1-1')]

# make list of stock dataframes
stocks = [aapl_df, amzn_df, fb_df, msft_df, tsla_df]

# check to see if data contains any null values
for i,stock in enumerate(stocks):
    print(stock.info())
    print(stock.isnull().sum())
    print(stock.describe())

# establish 
movingAverage = [7, 10, 14, 21, 50, 100]
min_max = [7, 30, 365, 730]

# add moving average features to datasets
moving_average(tsla_df, movingAverage)
moving_average(aapl_df, movingAverage)
moving_average(amzn_df, movingAverage)
moving_average(msft_df, movingAverage)
moving_average(fb_df, movingAverage)

# add max features to datasets
calc_max(tsla_df, min_max)
calc_max(aapl_df, min_max)
calc_max(amzn_df, min_max)
calc_max(msft_df, min_max)
calc_max(fb_df, min_max)

# add min features to datasets
calc_min(tsla_df, min_max)
calc_min(aapl_df, min_max)
calc_min(amzn_df, min_max)
calc_min(msft_df, min_max)
calc_min(fb_df, min_max)

