import numpy as np
import pandas as pd
import datetime as dt
import sklearn
import math
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, mutual_info_regression
import matplotlib.pyplot as plt
import seaborn as sns
import time
from datetime import datetime
import plotly_express as px
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# obtain the moving averages for the passed number of days
def moving_average(df, days):
    for ma in days:
        new_col = "%s MA" %(str(ma))
        df.loc[:, new_col] = pd.DataFrame.rolling(df['Close'], ma).mean()
        # replace resulting nan values with average
        mean = df[new_col].mean()
        df[new_col].fillna(value = mean, inplace = True)

# obtain the max over the past number of passed days
def calc_max(df, days):
    for ma in days:
        new_col = "Max over %s days" %(str(ma))
        df.loc[:,new_col] = pd.DataFrame.rolling(df['Close'], ma).max()
        # replace resulting nan values with average
        mean = df[new_col].mean()
        df[new_col].fillna(value = mean, inplace = True)

# obtain the min over the past number of passed days
def calc_min(df, days):
    for ma in days:
        new_col = "Min over %s days" %(str(ma))
        df.loc[:,new_col] = pd.DataFrame.rolling(df['Close'], ma).min()
        # replace resultin nan values with average
        mean = df[new_col].mean()
        df[new_col].fillna(value = mean, inplace = True)

# normalization
def minMaxNorm(df):
    scaler = MinMaxScaler(feature_range = (0,1))
    df = scaler.fit_transform(df)
    df = pd.DataFrame(df)
    return df

# remove the null values and replace with mean
def remove_nan(df):
    mean = df['Close'].mean()

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
tsla_df['Date'] = pd.to_datetime(tsla_df['Date'],dayfirst=True)
aapl_df['Date'] = pd.to_datetime(aapl_df['Date'],dayfirst=True)
amzn_df['Date'] = pd.to_datetime(amzn_df['Date'],dayfirst=True)
msft_df['Date'] = pd.to_datetime(msft_df['Date'],dayfirst=True)
fb_df['Date'] = pd.to_datetime(fb_df['Date'],dayfirst=True)

# delete all data prior to 1-1-2016
tsla_df = tsla_df[~(tsla_df['Date'] <= '2016-1-1')]
aapl_df = aapl_df[~(aapl_df['Date'] <= '2016-1-1')]
amzn_df = amzn_df[~(amzn_df['Date'] <= '2016-1-1')]
msft_df = msft_df[~(msft_df['Date'] <= '2016-1-1')]
fb_df = fb_df[~(fb_df['Date'] <= '2016-1-1')]

# sort all data sets
tsla_df.sort_values(by=['Date'])
aapl_df.sort_values(by=['Date'])
amzn_df.sort_values(by=['Date'])
msft_df.sort_values(by=['Date'])
fb_df.sort_values(by=['Date'])

tsla_df.set_index('Date', inplace=True)
aapl_df.set_index('Date', inplace=True)
amzn_df.set_index('Date', inplace=True)
msft_df.set_index('Date', inplace=True)
fb_df.set_index('Date', inplace=True)

# make list of stock dataframes
stocks = [aapl_df, amzn_df, fb_df, msft_df, tsla_df]

# check to see if data contains any null values
for i,stock in enumerate(stocks):
    print(stock.info())
    print(stock.isnull().sum())
    print(stock.describe())

# lists for calculating moving average and min/max
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

# add 7 day standard deviation
tsla_df.loc[:,'7 day std'] = pd.DataFrame.rolling(tsla_df['Close'], 7).std()
aapl_df.loc[:,'7 day std'] = pd.DataFrame.rolling(aapl_df['Close'], 7).std()
amzn_df.loc[:,'7 day std'] = pd.DataFrame.rolling(amzn_df['Close'], 7).std()
msft_df.loc[:,'7 day std'] = pd.DataFrame.rolling(msft_df['Close'], 7).std()
fb_df.loc[:,'7 day std'] = pd.DataFrame.rolling(fb_df['Close'], 7).std()

# add high-low diff feature
tsla_df.loc[:, 'High-Low Diff'] = tsla_df['High'] - tsla_df['Low']
aapl_df.loc[:, 'High-Low Diff'] = aapl_df['High'] - aapl_df['Low']
amzn_df.loc[:, 'High-Low Diff'] = amzn_df['High'] - amzn_df['Low']
msft_df.loc[:, 'High-Low Diff'] = msft_df['High'] - msft_df['Low']
fb_df.loc[:, 'High-Low Diff'] = fb_df['High'] - fb_df['Low']

# add open-close diff feature
tsla_df.loc[:, 'Open-Close Diff'] = tsla_df['Open'] - tsla_df['Close']
aapl_df.loc[:, 'Open-Close Diff'] = aapl_df['Open'] - aapl_df['Close']
amzn_df.loc[:, 'Open-Close Diff'] = amzn_df['Open'] - amzn_df['Close']
msft_df.loc[:, 'Open-Close Diff'] = msft_df['Open'] - msft_df['Close']
fb_df.loc[:, 'Open-Close Diff'] = fb_df['Open'] - fb_df['Close']

# add daily return feature
tsla_df.loc[:, 'Daily Return'] = tsla_df['Close'].pct_change()*100
aapl_df.loc[:, 'Daily Return'] = aapl_df['Close'].pct_change()*100
amzn_df.loc[:, 'Daily Return'] = amzn_df['Close'].pct_change()*100
msft_df.loc[:, 'Daily Return'] = msft_df['Close'].pct_change()*100
fb_df.loc[:, 'Daily Return'] = fb_df['Close'].pct_change()*100

# add previous close feature
tsla_df.loc[:, 'Close Previous'] = tsla_df['Close'].shift(+1)
aapl_df.loc[:, 'Close Previous'] = aapl_df['Close'].shift(+1)
amzn_df.loc[:, 'Close Previous'] = amzn_df['Close'].shift(+1)
msft_df.loc[:, 'Close Previous'] = msft_df['Close'].shift(+1)
fb_df.loc[:, 'Close Previous'] = fb_df['Close'].shift(+1)

# replace remaining nan value with column averages
mean = tsla_df['7 day std'].mean()
tsla_df['7 day std'].fillna(value = mean, inplace = True)
mean = tsla_df['Daily Return'].mean()
tsla_df['Daily Return'].fillna(value = mean, inplace = True)
mean = tsla_df['Close Previous'].mean()
tsla_df['Close Previous'].fillna(value = mean, inplace = True)

# replace remaining nan value with column averages
mean = aapl_df['7 day std'].mean()
aapl_df['7 day std'].fillna(value = mean, inplace = True)
mean = aapl_df['Daily Return'].mean()
aapl_df['Daily Return'].fillna(value = mean, inplace = True)
mean = aapl_df['Close Previous'].mean()
aapl_df['Close Previous'].fillna(value = mean, inplace = True)

# replace remaining nan value with column averages
mean = amzn_df['7 day std'].mean()
amzn_df['7 day std'].fillna(value = mean, inplace = True)
mean = amzn_df['Daily Return'].mean()
amzn_df['Daily Return'].fillna(value = mean, inplace = True)
mean = amzn_df['Close Previous'].mean()
amzn_df['Close Previous'].fillna(value = mean, inplace = True)

# replace remaining nan value with column averages
mean = msft_df['7 day std'].mean()
msft_df['7 day std'].fillna(value = mean, inplace = True)
mean = msft_df['Daily Return'].mean()
msft_df['Daily Return'].fillna(value = mean, inplace = True)
mean = msft_df['Close Previous'].mean()
msft_df['Close Previous'].fillna(value = mean, inplace = True)

# replace remaining nan value with column averages
mean = fb_df['7 day std'].mean()
fb_df['7 day std'].fillna(value = mean, inplace = True)
mean = fb_df['Daily Return'].mean()
fb_df['Daily Return'].fillna(value = mean, inplace = True)
mean = fb_df['Close Previous'].mean()
fb_df['Close Previous'].fillna(value = mean, inplace = True)

# confirming no more null values present in data set
print(tsla_df.isnull().sum())
print(aapl_df.isnull().sum())
print(amzn_df.isnull().sum())
print(msft_df.isnull().sum())
print(fb_df.isnull().sum())

# choose stock to test/train from
y = aapl_df['Close'][1:]
#y = tsla_df['Close'][1:]
#y = amzn_df['Close'][1:]
#y = msft_df['Close'][1:]
#y = fb_df['Close'][1:]

# rid dataset of the close data for training purposes
aapl_df.loc[:,'Close Previous'] = aapl_df['Close']
aapl_df = aapl_df.drop(['Close'], axis=1)
aapl_df = aapl_df.drop(['Adj Close'], axis=1)
#aapl_df = aapl_df.shift(periods=1)[1:]

amzn_df.loc[:,'Close Previous'] = amzn_df['Close']
amzn_df = amzn_df.drop(['Close'], axis=1)
amzn_df = amzn_df.drop(['Adj Close'], axis=1)
amzn_df = amzn_df.shift(periods=1)[1:]

msft_df.loc[:,'Close Previous'] = msft_df['Close']
msft_df = msft_df.drop(['Close'], axis=1)
msft_df = msft_df.drop(['Adj Close'], axis=1)
msft_df = msft_df.shift(periods=1)[1:]

tsla_df.loc[:,'Close Previous'] = tsla_df['Close']
tsla_df = tsla_df.drop(['Close'], axis=1)
tsla_df = tsla_df.drop(['Adj Close'], axis=1)
tsla_df = tsla_df.shift(periods=1)[1:]

fb_df.loc[:,'Close Previous'] = fb_df['Close']
fb_df = fb_df.drop(['Close'], axis=1)
fb_df = fb_df.drop(['Adj Close'], axis=1)
fb_df = fb_df.shift(periods=1)[1:]

# split test/train
x_train, x_test, y_train, y_test = train_test_split(aapl_df[1:], y, test_size=0.2, shuffle=False)

def selectFeatures(func, k=10,x_train=x_train, y_train=y_train):
    #apply SelectBest class to extract top 10 best features
    bestFeat = SelectKBest(score_func=func, k=5)
    fit = bestFeat.fit(x_train, y_train)
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(x_train.columns)
    # concat two dataframes for better visualization
    scores = pd.concat([dfcolumns, dfscores], axis=1)
    scores.columns = ['Feats','Score']
    scores.nlargest(5,'Score').plot.bar(x="Feats",y="Score",figsize=(6,5), color="Orange")
    plt.title("Regression")
    plt.show()
    topTen = list(scores.nlargest(5,'Score')['Feats'])
    for col in x_train:
        if col not in topTen:
            x_train.drop([col], axis=1, inplace=True)
    return x_train

x_train = selectFeatures(func=f_regression)
x_test = x_test[x_train.columns]

plt.figure(figsize=(16,8))
sns.heatmap(x_train.corr(), cmap=plt.cm.Blues,annot=True)
plt.title('Heatmap displaying the relationship between the features of the data', fontsize=12)
plt.show()

# mean absolute percentage error calculation
def mape(actual, prediction):
    actual, prediction = np.array(actual), np.array(prediction)
    return np.mean(np.abs((actual - prediction) / actual)) * 100

#regression_model = LinearRegression().fit(x_train, y_train)
#prediction = regression_model.predict(x_test)
#plot_df_train = pd.DataFrame(y_train)
#plot_df = pd.DataFrame(y_test)
#plot_df["Predictions"] = prediction
#print(plot_df[-5:])
#plt.figure(figsize = (18,5))
#plt.plot(plot_df_train[-360:])
#plt.plot(plot_df)
#plt.show()
#plot_df = pd.DataFrame(y_test)
#plot_df["predictions"] = prediction
#plot_df[-360:].plot(legend=True, figsize = (15,8))
#plt.show()

# place above into a function

def train_n_test(modelName,model,x_train=x_train,x_test=x_test,y_train=y_train,y_test=y_test):
    model=model.fit(x_train,y_train)
    prediction = model.predict(x_test)
    plot_df_train=pd.DataFrame(y_train)
    plot_df=pd.DataFrame(y_test)
    plot_df["predictions"]=prediction
    print(plot_df[-5:])
    plt.figure(figsize=(15,8))
    plt.plot(plot_df_train[-360:])
    plt.plot(plot_df)
    plt.show()
    plot_df=pd.DataFrame(y_test)
    plot_df["predictions"]=prediction
    plot_df[-360:].plot(legend=True, figsize = (15,8))
    plt.show()
    meanSquared = sklearn.metrics.mean_squared_error(y_test, prediction)
    rootMeanSquared = math.sqrt(meanSquared)
    print("Test set MAPE: {} ".format(modelName), mape(y_test, prediction))
    print("Test set RMSE: {} ".format(modelName), rootMeanSquared)
    return prediction

regression_model = LinearRegression()
randomForestRegressor_model = RandomForestRegressor(n_estimators=10, random_state=0)
svr = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.1))

# select model to use
regression_pred = train_n_test("Linear Regression Model", regression_model)
#randomForest_pred = train_test_and_measure("Random Forest Regression Model", randomForestRegressor_model)
#svr_pred = train_test_and_measure("Support Vector Regression Model", svr)
