#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 12:19:32 2019

@author: mahya
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
import statsmodels.tsa.api as smt
import statsmodels.api as sm
from scipy.optimize import minimize  
from itertools import product                   
from tqdm import tqdm_notebook
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
#from sklearn.preprocessing import StandardScaler
#from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from keras.layers import Dropout
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import Imputer


# =============================================================================
# Classes
# =============================================================================

class HoltWinters:
        
    def __init__(self, series, slen, alpha, beta, gamma, n_preds, scaling_factor=1.96):
        self.series = series
        self.slen = slen
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.n_preds = n_preds
        self.scaling_factor = scaling_factor
        
        
    def initial_trend(self):
        sum = 0.0
        for i in range(self.slen):
            sum += float(self.series[i+self.slen] - self.series[i]) / self.slen
        return sum / self.slen  

    def initial_seasonal_components(self):
        seasonals = {}
        season_averages = []
        n_seasons = int(len(self.series)/self.slen)
        
        # let's calculate season averages
        for j in range(n_seasons):
            season_averages.append(sum(self.series[self.slen*j:self.slen*j+self.slen])/float(self.slen))
        
        # let's calculate initial values
        for i in range(self.slen):
            sum_of_vals_over_avg = 0.0
            for j in range(n_seasons):
                sum_of_vals_over_avg += self.series[self.slen*j+i]-season_averages[j]
            seasonals[i] = sum_of_vals_over_avg/n_seasons
        return seasonals   

          
    def triple_exponential_smoothing(self):
        self.result = []
        self.Smooth = []
        self.Season = []
        self.Trend = []
        self.PredictedDeviation = []
        self.UpperBond = []
        self.LowerBond = []
        
        seasonals = self.initial_seasonal_components()
        
        for i in range(len(self.series)+self.n_preds):
            if i == 0:
                smooth = self.series[0]
                trend = self.initial_trend()
                self.result.append(self.series[0])
                self.Smooth.append(smooth)
                self.Trend.append(trend)
                self.Season.append(seasonals[i%self.slen])
                self.PredictedDeviation.append(0)
                
                self.UpperBond.append(self.result[0] + 
                                      self.scaling_factor * 
                                      self.PredictedDeviation[0])
                
                self.LowerBond.append(self.result[0] - 
                                      self.scaling_factor * 
                                      self.PredictedDeviation[0])
                continue
                
            if i >= len(self.series):
                m = i - len(self.series) + 1
                self.result.append((smooth + m*trend) + seasonals[i%self.slen])
            
                self.PredictedDeviation.append(self.PredictedDeviation[-1]*1.01) 
                
            else:
                val = self.series[i]
                last_smooth, smooth = smooth, self.alpha*(val-seasonals[i%self.slen]) + (1-self.alpha)*(smooth+trend)
                trend = self.beta * (smooth-last_smooth) + (1-self.beta)*trend
                seasonals[i%self.slen] = self.gamma*(val-smooth) + (1-self.gamma)*seasonals[i%self.slen]
                self.result.append(smooth+trend+seasonals[i%self.slen])
                
                self.PredictedDeviation.append(self.gamma * np.abs(self.series[i] - self.result[i]) + (1-self.gamma)*self.PredictedDeviation[-1])
                     
            self.UpperBond.append(self.result[-1] + 
                                  self.scaling_factor * 
                                  self.PredictedDeviation[-1])

            self.LowerBond.append(self.result[-1] - 
                                  self.scaling_factor * 
                                  self.PredictedDeviation[-1])

            self.Smooth.append(smooth)
            self.Trend.append(trend)
            self.Season.append(seasonals[i%self.slen])

            

# =============================================================================
# Functions
# =============================================================================
def mean_absolute_percentage_error(y_true, y_pred):
    
    return np.mean(np.abs((y_true - y_pred)/ y_true))*100


def plotMovingAverage(series, window, plot_intervals = False, scale = 1.96, plot_anomalies = False):
    
    rolling_mean = series.rolling(window = window).mean()
    
    plt.figure(figsize = (25, 7))
    plt.title("Moving Average/n window size: {}".format(window))
    plt.plot(rolling_mean, 'g', label = 'rolling mean trend')
    
    if plot_intervals:
        mae = mean_absolute_percentage_error(series[window:], rolling_mean[window:])
        deviation = np.std(series[window:] - rolling_mean[window:])
        lower_bond = rolling_mean - (mae + scale * deviation)
        upper_bond = rolling_mean + (mae + scale * deviation)
        plt.plot(upper_bond, "r--", label="Upper Bond / Lower Bond")
        plt.plot(lower_bond, "r--")
       
    if plot_anomalies:
        anomalies = pd.DataFrame(index=series.index, columns=series.columns)
        anomalies[series<lower_bond] = series[series<lower_bond]
        anomalies[series>upper_bond] = series[series>upper_bond]
        plt.plot(anomalies, "ro", markersize=10)
        
    plt.plot(series[window:], label="Actual values")
    plt.legend(loc="upper left")
    plt.grid(True)
    
    return rolling_mean
  
def exponential_smoothing(series, alpha):

    result = [series.values[0]]
    
    for n in range(1, len(series)):
        result.append(alpha * series.values[n] + (1 - alpha) * result[n-1])
        
    return result

def plotExponentialSmoothing(series, alphas):
 
    with plt.style.context('seaborn-white'):    
        plt.figure(figsize=(25, 7))
        
        for alpha in alphas:
            plt.plot(exponential_smoothing(series, alpha), label="Alpha {}".format(alpha))
       
        plt.plot(series.values, "c", label = "Actual")
        plt.legend(loc="best")
        plt.axis('tight')
        plt.title("Exponential Smoothing")
        plt.grid(True);
    
def double_exponential_smoothing(series, alpha, beta):
    result = [series.values[0]]
    for n in range(1, len(series)+1):
        if n == 1:
            level, trend = series.values[0], series.values[1] - series.values[0]
        if n >= len(series): 
            value = result[-1]
        else:
            value = series.values[n]
        last_level, level = level, alpha*value + (1-alpha)*(level+trend)
        trend = beta*(level-last_level) + (1-beta)*trend
        result.append(level+trend)
    return result

def plotDoubleExponentialSmoothing(series, alphas, betas):
    with plt.style.context('seaborn-white'):    
        plt.figure(figsize=(25, 7))
        for alpha in alphas:
            for beta in betas:
                plt.plot(double_exponential_smoothing(series, alpha, beta), label="Alpha {}, beta {}".format(alpha, beta))
        plt.plot(series.values, label = "Actual")
        plt.legend(loc="best")
        plt.axis('tight')
        plt.title("Double Exponential Smoothing")
        plt.grid(True)

def timeseriesCVscore(params, series, loss_function = mean_absolute_percentage_error, slen = 24):
    errors = []
    values = series.values
    alpha, beta, gamma = params
    tscv = TimeSeriesSplit(n_splits = 5)
    
    for train, test in tscv.split(values):
        model = HoltWinters(series=values[train], slen=slen, alpha=alpha, beta=beta, gamma=gamma, n_preds=len(test))
        model.triple_exponential_smoothing()
        
        actual = values[test]
        predictions = model.result[-len(test):]
        
        error = loss_function(predictions, actual)
        errors.append(error)
    
    return np.mean(np.array(errors))
    

def plotHoltWinters(series, plot_intervals=False, plot_anomalies=False):
    
    plt.figure(figsize=(25, 10))
    plt.plot(model.result, label = "Model")
    plt.plot(series.values, label = "Actual")
    error = mean_absolute_percentage_error(series.values, model.result[:len(series)])
    plt.title("Mean Absolute Percentage Error: {0:.2f}%".format(error))
    
    if plot_anomalies:
        anomalies = np.array([np.NaN]*len(series))
        anomalies[series.values<model.LowerBond[:len(series)]] = \
            series.values[series.values<model.LowerBond[:len(series)]]
        anomalies[series.values>model.UpperBond[:len(series)]] = \
            series.values[series.values>model.UpperBond[:len(series)]]
        plt.plot(anomalies, "o", markersize=10, label = "Anomalies")
    
    if plot_intervals:
        plt.plot(model.UpperBond, "r--", alpha=0.5, label = "Up/Low confidence")
        plt.plot(model.LowerBond, "r--", alpha=0.5)
        plt.fill_between(x=range(0,len(model.result)), y1=model.UpperBond, 
                         y2=model.LowerBond, alpha=0.2, color = "grey")    
        
    plt.vlines(len(series), ymin=min(model.LowerBond), ymax=max(model.UpperBond), linestyles='dashed')
    plt.axvspan(len(series)-20, len(model.result), alpha=0.3, color='lightgrey')
    plt.grid(True)
    plt.axis('tight')
    plt.legend(loc="best", fontsize=13); 
 
def tsplot(y, lags=None, figsize=(25, 7), style='bmh'):
    if not isinstance(y, pd.Series):
        y = pd.Series.values(y)
        
    with plt.style.context(style):    
        plt.figure(figsize=figsize)
        layout = (2, 2)
        ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
        acf_ax = plt.subplot2grid(layout, (1, 0))
        pacf_ax = plt.subplot2grid(layout, (1, 1))
        
        y.plot(ax=ts_ax)
        p_value = sm.tsa.stattools.adfuller(y)[1]
        ts_ax.set_title('Time Series Analysis Plots\n Dickey-Fuller: p={0:.5f}'.format(p_value))
        smt.graphics.plot_acf(y, lags=lags, ax=acf_ax)
        smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax)
        plt.tight_layout()
        
def optimizeSARIMA(parameters_list, d, D, s):
    results = []
    best_aic = float('inf')
    
    for param in tqdm_notebook(parameters_list):
        try:
            model = sm.tsa.statespace.SARIMAX(airPollution.total_psi.values, order=(param[0], d, param[1]), 
                                            seasonal_order=(param[2], D, param[3], s)).fit(disp=-1)
        except:
            continue
        
        aic = model.aic
        
        if aic < best_aic:
            best_aic = aic
        
        results.append([param, model.aic])
        
    result_table = pd.DataFrame(results)
    result_table.columns = (['Params', 'aic'])
    result_table = result_table.sort_values(by='aic', ascending=True).reset_index(drop=True)
    
    return result_table

def plotSARIMA(series, model, n_steps):

    data = series.copy()
    data.columns = ['actual']
    data['arima_model'] = model.fittedvalues
    data['arima_model'][:s+d] = np.NaN
    
    forecast = model.predict(start = data.shape[0], end = data.shape[0]+n_steps)
    forecast = data.arima_model.append(forecast)
  
    error = mean_absolute_percentage_error(data['actual'][s+d:], data['arima_model'][s+d:])

    plt.figure(figsize=(25, 7))
    plt.title("Mean Absolute Percentage Error: {0:.2f}%".format(error))
    plt.plot(forecast, color='r', label="model")
    plt.axvspan(data.index[-1], forecast.index[-1], alpha=0.5, color='lightgrey')
    plt.plot(data.actual, label="actual")
    plt.legend()
    plt.grid(True);
   
def timeseries_train_test_split(X, y, test_size):
    
    test_index = int(len(X) * (1-test_size))
    
    X_train = X.iloc[:test_index]
    y_train = y.iloc[:test_index]
    X_test = X.iloc[test_index:]
    y_test = y.iloc[test_index:]
    
    return X_train, X_test, y_train, y_test

def plotModelResults(model, X_train, X_test, plotName, plot_intervals=False, plot_anomalies=False):
    
    prediction = model.predict(X_test)
    
    plt.figure(figsize=(25, 7))
    plt.plot(prediction, "g", label="prediction", linewidth=2.0)
    plt.plot(y_test.values, label="actual", linewidth=2.0)
    
    if plot_intervals:
        cv = cross_val_score(model, X_train, y_train, cv=tscv, scoring="neg_mean_absolute_error")
        mae = cv.mean() * (-1)
        deviation = cv.std()
        
        scale = 1.96
        lower = prediction - (mae + scale * deviation)
        upper = prediction + (mae + scale * deviation)
        
        plt.plot(lower, "r--", label="upper bond / lower bond", alpha=0.5)
        plt.plot(upper, "r--", alpha=0.5)
        
        if plot_anomalies:
            anomalies = np.array([np.NaN]*len(y_test))
            anomalies[y_test<lower] = y_test[y_test<lower]
            anomalies[y_test>upper] = y_test[y_test>upper]
            plt.plot(anomalies, "o", markersize=10, label = "Anomalies")
    
    error = mean_absolute_percentage_error(prediction, y_test)
    plt.title("Mean absolute percentage error {0:.2f}%/Plot name: {1}".format(error, plotName))
    plt.legend(loc="best")
    plt.tight_layout()
    plt.grid(True);
    
def plotCoefficients(model):
    
    coefs = pd.DataFrame(model.coef_, X_train.columns)
    coefs.columns = ["coef"]
    coefs["abs"] = coefs.coef.apply(np.abs)
    coefs = coefs.sort_values(by="abs", ascending=False).drop(["abs"], axis=1)
    
    plt.figure(figsize=(25, 7))
    coefs.coef.plot(kind='bar')
    plt.grid(True, axis='y')
    plt.hlines(y=0, xmin=0, xmax=len(coefs), linestyles='dashed');
    
    return coefs.coef 

def prepareData(series, lags, test_size):

    data = pd.DataFrame(series.copy())
    data.columns = ["y"]
 
    for i in lags:
        data["f{}".format(i)] = data.y.shift(i)
    
    y = data.dropna().y
    X = data.dropna().drop(['y'], axis=1)
    X_train, X_test, y_train, y_test = timeseries_train_test_split(X, y, test_size=test_size)

    return X_train, X_test, y_train, y_test

    
 
def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)       
# =============================================================================
# Data Preparation
# =============================================================================

#df = pd.read_excel(r'pollutionDataset.xlsx')
##print(df.info())
#df = df.drop(["?????_?????", "?????_??????"], axis=1)
#df.columns = ['feature1', 'date', 'psi_co', 'psi_o3', 'psi_no2', 'psi_so2', 'psi_pm10', 'total_psi', 'year_month','year','month', 'feature2']
##print(df.head())
#
#df['date'] = df['date'].str.replace('-', '/')
#df['date'] = df['date'].str.split('/')
#df['year'] = (df['date'].apply(lambda x: x[0])).astype(int)
#df['month'] = (df['date'].apply(lambda x: x[1]))
#df['day'] = (df['date'].apply(lambda x: x[-1]))
#
#indexNames = df[(df['year'] == 13898) | (df['year'] == 139608) | (df['year'] == 1386)].index
#df.drop(indexNames , inplace=True)
#indexNames = df[(df['month'] == '1125') | (df['month'] == '1105') | (df['month'] == '' )].index
#df.drop(indexNames , inplace=True)
#
#df['month'] = ["%02d" % x for x in df['month'].astype(int)]
#indexNames = df[(df['day'] == '.0')].index
#
#df.drop(indexNames , inplace=True)
#df['day'] = ["%02d" % x for x in df['day'].astype(int)]
#
#df['date'] = df['year'].map(str) + '-' + df['month'].map(str) + '-' + df['day'].map(str)
#df['year_month'] = df['year'].map(str) + '-' + df['month'].map(str)
#df = df.sort_values('date')
#df['date'] = pd.to_datetime(df.date, format='%Y/%m/%d', errors = 'ignore')
#df['year_month'] = pd.to_datetime(df.year_month, format='%Y-%m', errors = 'ignore')
#df.set_index(['date'])
#
##print(df['year'].value_counts())
#
#df.to_csv('airPullotion.csv')

airPollution = pd.read_csv(r'airPullotion.csv', index_col=['date'], parse_dates=['date'])
#airPollution = pd.read_csv(r'airPullotion.csv', header=0, index_col=0)
airPollution = airPollution.fillna(method='ffill')
airPollution = airPollution.sort_values('date')

# =============================================================================
# Air pollution plots for total psi
# =============================================================================
#airPollution.groupby('year').plot(x='year_month', y='total_psi')

airPollution['total_psi'] = airPollution.total_psi.fillna(airPollution.total_psi - airPollution.total_psi.mean())
indexNames = airPollution[airPollution['total_psi'] > 1000.0 ].index
airPollution.drop(indexNames , inplace=True)
airPollution = airPollution[['total_psi']]
print(airPollution)
# # =============================================================================
# # Identify trend/ Level/ Sesonal 
# # =============================================================================
# ######################## Moving Avearge ################################
airPollution_movingAverage = plotMovingAverage(airPollution, 7, plot_intervals=False, plot_anomalies=False)
print(airPollution_movingAverage)
plt.plot(airPollution_movingAverage)
plt.show()
# ####################### Exponential data ###############################
#plotExponentialSmoothing(airPollution_total.total_psi, [0.7, 0.05])
#airPollution_ExpSmoothing = exponential_smoothing(airPollution_total.total_psi, 0.7)

# ################### Double exponential data #############################
#plotDoubleExponentialSmoothing(airPollution_total.total_psi, alphas=[0.9, 0.02], betas=[0.9, 0.02])
#airPollution_DoubleExpSmoothing = double_exponential_smoothing(airPollution_total.total_psi, 0.9, 0.9)

# # =============================================================================
# # Corss Validation / Stationarity
# # =============================================================================
# ########################## Triple exponetioal data smoothing ###########
#data = airPollution.total_psi[:-30]
#x=[0, 0, 0]
#slen = 30
# 
#opt = minimize(timeseriesCVscore, x0=x, 
#               args=(data, mean_absolute_percentage_error, slen), 
#               method="TNC", bounds = ((0, 1), (0, 1), (0, 1))
#              )
# 
#alpha_final, beta_final, gamma_final = opt.x
# 
#model = HoltWinters(data.values, slen = 30, 
#                    alpha = alpha_final,
#                    beta = beta_final,
#                    gamma = gamma_final,
#                    n_preds = 100, scaling_factor = 3)
# 
#model.triple_exponential_smoothing()
# 
#plotHoltWinters(airPollution_total.total_psi, plot_intervals=True, plot_anomalies=True)
#
#tsplot(airPollution_total.total_psi, lags=60)
# 
#airPollution_diff = airPollution_total.total_psi - airPollution_total.total_psi.shift(30)
#airPollution_diff = airPollution_diff - airPollution_diff.shift(1)
#tsplot(airPollution_diff[40+1:], lags=90)

# ########################## SARIMA #########################
#ps = range(2, 5)
#d=1 
#qs = range(2, 5)
#Ps = range(0, 2)
#D=1 
#Qs = range(0, 2)
#s = 30
#
#parameters = product(ps, qs, Ps, Qs)
#parameters_list = list(parameters)
#len(parameters_list)
#result_table = optimizeSARIMA(parameters_list, d, D, s)
#
#p, q, P, Q = result_table.parameters[0]
#
#best_model = sm.tsa.statespace.SARIMAX(airPollution.total_psi, order=(p, d, q), seasonal_order=(P, D, Q, s)).fit(disp=-1)
#best_model.summary()
# 
#tsplot(best_model.resid[29+1:], lags=60)
#plotSARIMA(airPollution_total, best_model, 50)

# # =============================================================================
# # Model
# # =============================================================================
# ########################## Linear Regression ####################
data = pd.DataFrame(airPollution_movingAverage.copy())
data.columns = ['y']

for i in range(3, 60):
   data['lags: {}'.format(i)] = data.y.shift(i)
 
tscv = TimeSeriesSplit(n_splits=3)
 
y = data.dropna().y
X = data.dropna().drop(['y'], axis = 1)
X_train, X_test, y_train, y_test = timeseries_train_test_split(X, y, test_size = 0.3)

lr = LinearRegression()
lr.fit(X_train, y_train)
 
plotModelResults(lr,X_train, X_test, 'Linear Regression', plot_intervals=True)
linear_lags = plotCoefficients(lr)

# ## =============================================================================
# ## Scale
# ## =============================================================================
# ################### Linear Reg. with scaling #######
scaler = preprocessing.StandardScaler()
#y = data.dropna().y
#X = data.dropna().drop(['y'], axis=1)
# 
#X_train, X_test, y_train, y_test = timeseries_train_test_split(X, y, test_size=0.3)
# 
#X_train_scaled = scaler.fit_transform(X_train)
#X_test_scaled = scaler.transform(X_test)
# 
#lr = LinearRegression()
#lr.fit(X_train_scaled, y_train)
# 
#plotModelResults(lr, X_train_scaled, X_test_scaled, 'Linear Reg. with scaling', plot_intervals=True, plot_anomalies=True)
#plotCoefficients(lr)
# 
# ######################### Linear Reg. with acaling and prepared data #########
 
#X_train, X_test, y_train, y_test = prepareData(airPollution_movingAverage.total_psi.values, lags=[3,21], test_size=0.3)
#X_train_scaled = scaler.fit_transform(X_train)
#X_test_scaled = scaler.transform(X_test)
# 
#lr = LinearRegression()
#lr.fit(X_train_scaled, y_train)
# 
#plotModelResults(lr, X_train_scaled, X_test_scaled, 'Linear Reg. with scaling and prepared data', plot_intervals=True, plot_anomalies=True)
#plotCoefficients(lr)

############################## Ridge ###########################
#from sklearn.linear_model import LassoCV, RidgeCV
#
#ridge = RidgeCV(cv=tscv)
#ridge.fit(X_train_scaled, y_train)
#
#plotModelResults(ridge, X_train_scaled, X_test_scaled, 'Ridge',plot_intervals=True, plot_anomalies=True)
#plotCoefficients(ridge)

############################## LASSO ###########################
#lasso = LassoCV(cv=tscv)
#lasso.fit(X_train_scaled, y_train)
#
#plotModelResults(lasso, X_train_scaled, X_test_scaled, 'Lasso', plot_intervals=True, plot_anomalies=True)
#plotCoefficients(lasso)

############################## Xgboost ###########################
#X_train, X_test, y_train, y_test = prepareData(airPollution_movingAverage.total_psi.values, 
#                                               lags=[2, 7, 14, 31, 21, 28, 38, 4,35], test_size=0.3)
X_train, X_test, y_train, y_test = timeseries_train_test_split(X, y, test_size=0.3)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
from xgboost import XGBRegressor 

my_imputer = Imputer()
train_X = my_imputer.fit_transform(X_train_scaled)
test_X = my_imputer.transform(X_test_scaled)

#xgb = XGBRegressor(n_estimators=1000, learning_rate=0.05)
#xgb.fit(train_X, y_train, early_stopping_rounds=5, 
#             eval_set=[(X_test, y_test)], verbose=False)

xgb = XGBRegressor()
xgb.fit(train_X, y_train)

plotModelResults(xgb, X_train_scaled, X_test_scaled, 'Xgboost', plot_intervals=True, plot_anomalies=True)

################################# LSTM #############################
np.random.seed(7)
df = airPollution_movingAverage
df['total_psi'] = pd.to_numeric(df['total_psi'], errors='coerce')
df = df.dropna(subset=['total_psi'])
dataset = df.total_psi.values #numpy.ndarray
dataset = dataset.astype('float32')

dataset = np.reshape(dataset, (-1, 1))
scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)
train_size = int(len(dataset) * 0.80)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
    
look_back = 30
X_train, y_train = create_dataset(train, look_back)
X_test, y_test = create_dataset(test, look_back)

X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

model = Sequential()

model.add(LSTM(100, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

history = model.fit(X_train, y_train, epochs=20, batch_size=70, validation_data=(X_test, y_test), 
                    callbacks=[EarlyStopping(monitor='val_loss', patience=10)], verbose=1, shuffle=False)

model.summary()
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

train_predict = scaler.inverse_transform(train_predict)
y_train = scaler.inverse_transform([y_train])
test_predict = scaler.inverse_transform(test_predict)
y_test = scaler.inverse_transform([y_test])

plt.legend(loc="best")
plt.tight_layout()
plt.grid(True);

error = mean_absolute_percentage_error(y_test[0], test_predict[:,0])

aa=[x for x in range(200)]
plt.figure(figsize=(15,7))
plt.plot(aa, y_test[0][:200], marker='.', label="actual")
plt.plot(aa, test_predict[:,0][:200], 'r', label="prediction")
plt.title("Mean absolute percentage error {0:.2f}%/Plot name: {1}".format(error, 'LSTM'))
plt.tight_layout()
sns.despine(top=True)
plt.subplots_adjust(left=0.07)
plt.ylabel('Total psi', size=15)
plt.xlabel('Time step', size=15)
plt.legend(fontsize=15)
plt.grid(True)
plt.show();

