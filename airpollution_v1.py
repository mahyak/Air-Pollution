# =============================================================================
# # =============================================================================
# # #!/usr/bin/env python3
# # =============================================================================
# # -*- coding: utf-8 -*-
# """
# Created on Wed Nov 20 23:44:14 2019
# 
# @author: mahya
# """
# 
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Created on Sat Oct 26 12:19:32 2019
# 
# @author: mahya
# """
# 
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import warnings
# warnings.filterwarnings('ignore')
# import seaborn as sns
# import statsmodels.tsa.api as smt
# import statsmodels.api as sm
# from scipy.optimize import minimize  
# from itertools import product                   
# from tqdm import tqdm_notebook
# from sklearn.model_selection import TimeSeriesSplit
# from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import cross_val_score
# #from sklearn.preprocessing import StandardScaler
# #from sklearn.preprocessing import MinMaxScaler
# from sklearn import preprocessing
# from keras.layers import Dropout
# from keras.callbacks import EarlyStopping
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.layers import LSTM
# from sklearn.preprocessing import Imputer
# import datetime
# import statsmodels.api as sm
# from statsmodels.tsa.stattools import adfuller, acf, pacf,arma_order_select_ic
# from pandas import Series as Series
# from sklearn.preprocessing import MinMaxScaler
# 
# 
# # =============================================================================
# # Classes
# # =============================================================================
# 
# class HoltWinters:
#         
#     def __init__(self, series, slen, alpha, beta, gamma, n_preds, scaling_factor=1.96):
#         self.series = series
#         self.slen = slen
#         self.alpha = alpha
#         self.beta = beta
#         self.gamma = gamma
#         self.n_preds = n_preds
#         self.scaling_factor = scaling_factor
#         
#         
#     def initial_trend(self):
#         sum = 0.0
#         for i in range(self.slen):
#             sum += float(self.series[i+self.slen] - self.series[i]) / self.slen
#         return sum / self.slen  
# 
#     def initial_seasonal_components(self):
#         seasonals = {}
#         season_averages = []
#         n_seasons = int(len(self.series)/self.slen)
#         
#         # let's calculate season averages
#         for j in range(n_seasons):
#             season_averages.append(sum(self.series[self.slen*j:self.slen*j+self.slen])/float(self.slen))
#         
#         # let's calculate initial values
#         for i in range(self.slen):
#             sum_of_vals_over_avg = 0.0
#             for j in range(n_seasons):
#                 sum_of_vals_over_avg += self.series[self.slen*j+i]-season_averages[j]
#             seasonals[i] = sum_of_vals_over_avg/n_seasons
#         return seasonals   
# 
#           
#     def triple_exponential_smoothing(self):
#         self.result = []
#         self.Smooth = []
#         self.Season = []
#         self.Trend = []
#         self.PredictedDeviation = []
#         self.UpperBond = []
#         self.LowerBond = []
#         
#         seasonals = self.initial_seasonal_components()
#         
#         for i in range(len(self.series)+self.n_preds):
#             if i == 0:
#                 smooth = self.series[0]
#                 trend = self.initial_trend()
#                 self.result.append(self.series[0])
#                 self.Smooth.append(smooth)
#                 self.Trend.append(trend)
#                 self.Season.append(seasonals[i%self.slen])
#                 self.PredictedDeviation.append(0)
#                 
#                 self.UpperBond.append(self.result[0] + 
#                                       self.scaling_factor * 
#                                       self.PredictedDeviation[0])
#                 
#                 self.LowerBond.append(self.result[0] - 
#                                       self.scaling_factor * 
#                                       self.PredictedDeviation[0])
#                 continue
#                 
#             if i >= len(self.series):
#                 m = i - len(self.series) + 1
#                 self.result.append((smooth + m*trend) + seasonals[i%self.slen])
#             
#                 self.PredictedDeviation.append(self.PredictedDeviation[-1]*1.01) 
#                 
#             else:
#                 val = self.series[i]
#                 last_smooth, smooth = smooth, self.alpha*(val-seasonals[i%self.slen]) + (1-self.alpha)*(smooth+trend)
#                 trend = self.beta * (smooth-last_smooth) + (1-self.beta)*trend
#                 seasonals[i%self.slen] = self.gamma*(val-smooth) + (1-self.gamma)*seasonals[i%self.slen]
#                 self.result.append(smooth+trend+seasonals[i%self.slen])
#                 
#                 self.PredictedDeviation.append(self.gamma * np.abs(self.series[i] - self.result[i]) + (1-self.gamma)*self.PredictedDeviation[-1])
#                      
#             self.UpperBond.append(self.result[-1] + 
#                                   self.scaling_factor * 
#                                   self.PredictedDeviation[-1])
# 
#             self.LowerBond.append(self.result[-1] - 
#                                   self.scaling_factor * 
#                                   self.PredictedDeviation[-1])
# 
#             self.Smooth.append(smooth)
#             self.Trend.append(trend)
#             self.Season.append(seasonals[i%self.slen])
# 
#             
# 
# # =============================================================================
# # Functions
# # =============================================================================
# def mean_absolute_percentage_error(y_true, y_pred):
#     
#     return np.mean(np.abs((y_true - y_pred)/ y_true))*100
# 
# 
# def plotMovingAverage(series, window, types,plot_intervals = False, scale = 1.96, plot_anomalies = False):
#     
#     if types == 'std':
#         rolling_series = series.rolling(window = window).std()
#         plt.figure(figsize = (16, 8))
#         plt.title("Moving Average/n window size: {}".format(window))
#         plt.plot(rolling_series, 'g', label = 'rolling std trend')
#     
#     elif types == 'mean':
#         rolling_series = series.rolling(window = window).mean()
#         plt.figure(figsize = (16, 8))
#         plt.title("Moving Average/n window size: {}".format(window))
#         plt.plot(rolling_series, 'g', label = 'rolling mean trend')
#     
#     if plot_intervals:
#         mae = mean_absolute_percentage_error(series[window:], rolling_series[window:])
#         deviation = np.std(series[window:] - rolling_series[window:])
#         lower_bond = rolling_series - (mae + scale * deviation)
#         upper_bond = rolling_series + (mae + scale * deviation)
#         plt.plot(upper_bond, "r--", label="Upper Bond / Lower Bond")
#         plt.plot(lower_bond, "r--")
#        
#     if plot_anomalies:
#         anomalies = pd.DataFrame(index=series.index, columns=series.columns)
#         anomalies[series<lower_bond] = series[series<lower_bond]
#         anomalies[series>upper_bond] = series[series>upper_bond]
#         plt.plot(anomalies, "ro", markersize=10)
#         
#     plt.plot(series[window:], label="Actual values")
#     plt.legend(loc="upper left")
#     plt.grid(True)
#     
#     return rolling_series
#   
# def exponential_smoothing(series, alpha):
# 
#     result = [series.values[0]]
#     
#     for n in range(1, len(series)):
#         result.append(alpha * series.values[n] + (1 - alpha) * result[n-1])
#         
#     return result
# 
# def plotExponentialSmoothing(series, alphas):
#  
#     with plt.style.context('seaborn-white'):    
#         plt.figure(figsize=(25, 7))
#         
#         for alpha in alphas:
#             plt.plot(exponential_smoothing(series, alpha), label="Alpha {}".format(alpha))
#        
#         plt.plot(series.values, "c", label = "Actual")
#         plt.legend(loc="best")
#         plt.axis('tight')
#         plt.title("Exponential Smoothing")
#         plt.grid(True);
#     
# def double_exponential_smoothing(series, alpha, beta):
#     result = [series.values[0]]
#     for n in range(1, len(series)+1):
#         if n == 1:
#             level, trend = series.values[0], series.values[1] - series.values[0]
#         if n >= len(series): 
#             value = result[-1]
#         else:
#             value = series.values[n]
#         last_level, level = level, alpha*value + (1-alpha)*(level+trend)
#         trend = beta*(level-last_level) + (1-beta)*trend
#         result.append(level+trend)
#     return result
# 
# def plotDoubleExponentialSmoothing(series, alphas, betas):
#     with plt.style.context('seaborn-white'):    
#         plt.figure(figsize=(25, 7))
#         for alpha in alphas:
#             for beta in betas:
#                 plt.plot(double_exponential_smoothing(series, alpha, beta), label="Alpha {}, beta {}".format(alpha, beta))
#         plt.plot(series.values, label = "Actual")
#         plt.legend(loc="best")
#         plt.axis('tight')
#         plt.title("Double Exponential Smoothing")
#         plt.grid(True)
# 
# def timeseriesCVscore(params, series, loss_function = mean_absolute_percentage_error, slen = 24):
#     errors = []
#     values = series.values
#     alpha, beta, gamma = params
#     tscv = TimeSeriesSplit(n_splits = 5)
#     
#     for train, test in tscv.split(values):
#         model = HoltWinters(series=values[train], slen=slen, alpha=alpha, beta=beta, gamma=gamma, n_preds=len(test))
#         model.triple_exponential_smoothing()
#         
#         actual = values[test]
#         predictions = model.result[-len(test):]
#         
#         error = loss_function(predictions, actual)
#         errors.append(error)
#     
#     return np.mean(np.array(errors))
#     
# 
# def plotHoltWinters(series, plot_intervals=False, plot_anomalies=False):
#     
#     plt.figure(figsize=(25, 10))
#     plt.plot(model.result, label = "Model")
#     plt.plot(series.values, label = "Actual")
#     error = mean_absolute_percentage_error(series.values, model.result[:len(series)])
#     plt.title("Mean Absolute Percentage Error: {0:.2f}%".format(error))
#     
#     if plot_anomalies:
#         anomalies = np.array([np.NaN]*len(series))
#         anomalies[series.values<model.LowerBond[:len(series)]] = \
#             series.values[series.values<model.LowerBond[:len(series)]]
#         anomalies[series.values>model.UpperBond[:len(series)]] = \
#             series.values[series.values>model.UpperBond[:len(series)]]
#         plt.plot(anomalies, "o", markersize=10, label = "Anomalies")
#     
#     if plot_intervals:
#         plt.plot(model.UpperBond, "r--", alpha=0.5, label = "Up/Low confidence")
#         plt.plot(model.LowerBond, "r--", alpha=0.5)
#         plt.fill_between(x=range(0,len(model.result)), y1=model.UpperBond, 
#                          y2=model.LowerBond, alpha=0.2, color = "grey")    
#         
#     plt.vlines(len(series), ymin=min(model.LowerBond), ymax=max(model.UpperBond), linestyles='dashed')
#     plt.axvspan(len(series)-20, len(model.result), alpha=0.3, color='lightgrey')
#     plt.grid(True)
#     plt.axis('tight')
#     plt.legend(loc="best", fontsize=13); 
#  
# def tsplot(y, lags=None, figsize=(25, 7), style='bmh'):
#     if not isinstance(y, pd.Series):
#         y = pd.Series.values(y)
#         
#     with plt.style.context(style):    
#         plt.figure(figsize=figsize)
#         layout = (2, 2)
#         ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
#         acf_ax = plt.subplot2grid(layout, (1, 0))
#         pacf_ax = plt.subplot2grid(layout, (1, 1))
#         
#         y.plot(ax=ts_ax)
#         p_value = sm.tsa.stattools.adfuller(y)[1]
#         ts_ax.set_title('Time Series Analysis Plots\n Dickey-Fuller: p={0:.5f}'.format(p_value))
#         smt.graphics.plot_acf(y, lags=lags, ax=acf_ax)
#         smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax)
#         plt.tight_layout()
#         
# def optimizeSARIMA(parameters_list, d, D, s):
#     results = []
#     best_aic = float('inf')
#     
#     for param in tqdm_notebook(parameters_list):
#         try:
#             model = sm.tsa.statespace.SARIMAX(airPollution.total_psi.values, order=(param[0], d, param[1]), 
#                                             seasonal_order=(param[2], D, param[3], s)).fit(disp=-1)
#         except:
#             continue
#         
#         aic = model.aic
#         
#         if aic < best_aic:
#             best_aic = aic
#         
#         results.append([param, model.aic])
#         
#     result_table = pd.DataFrame(results)
#     result_table.columns = (['Params', 'aic'])
#     result_table = result_table.sort_values(by='aic', ascending=True).reset_index(drop=True)
#     
#     return result_table
# 
# def plotSARIMA(series, model, n_steps):
# 
#     data = series.copy()
#     data.columns = ['actual']
#     data['arima_model'] = model.fittedvalues
#     data['arima_model'][:s+d] = np.NaN
#     
#     forecast = model.predict(start = data.shape[0], end = data.shape[0]+n_steps)
#     forecast = data.arima_model.append(forecast)
#   
#     error = mean_absolute_percentage_error(data['actual'][s+d:], data['arima_model'][s+d:])
# 
#     plt.figure(figsize=(16, 8))
#     plt.title("Mean Absolute Percentage Error: {0:.2f}%".format(error))
#     plt.plot(forecast, color='r', label="model")
#     plt.axvspan(data.index[-1], forecast.index[-1], alpha=0.5, color='lightgrey')
#     plt.plot(data.actual, label="actual")
#     plt.legend()
#     plt.grid(True);
#    
# def timeseries_train_test_split(X, y, test_size):
#     
#     test_index = int(len(X) * (1-test_size))
#     
#     X_train = X.iloc[:test_index]
#     y_train = y.iloc[:test_index]
#     X_test = X.iloc[test_index:]
#     y_test = y.iloc[test_index:]
#     
#     return X_train, X_test, y_train, y_test
# 
# def plotModelResults(model, X_train, X_test, plotName, plot_intervals=False, plot_anomalies=False):
#     
#     prediction = model.predict(X_test)
#     
#     plt.figure(figsize=(25, 7))
#     plt.plot(prediction, "g", label="prediction", linewidth=2.0)
#     plt.plot(y_test.values, label="actual", linewidth=2.0)
#     
#     if plot_intervals:
#         cv = cross_val_score(model, X_train, y_train, cv=tscv, scoring="neg_mean_absolute_error")
#         mae = cv.mean() * (-1)
#         deviation = cv.std()
#         
#         scale = 1.96
#         lower = prediction - (mae + scale * deviation)
#         upper = prediction + (mae + scale * deviation)
#         
#         plt.plot(lower, "r--", label="upper bond / lower bond", alpha=0.5)
#         plt.plot(upper, "r--", alpha=0.5)
#         
#         if plot_anomalies:
#             anomalies = np.array([np.NaN]*len(y_test))
#             anomalies[y_test<lower] = y_test[y_test<lower]
#             anomalies[y_test>upper] = y_test[y_test>upper]
#             plt.plot(anomalies, "o", markersize=10, label = "Anomalies")
#     
#     error = mean_absolute_percentage_error(prediction, y_test)
#     plt.title("Mean absolute percentage error {0:.2f}%/Plot name: {1}".format(error, plotName))
#     plt.legend(loc="best")
#     plt.tight_layout()
#     plt.grid(True);
#     
# def plotCoefficients(model):
#     
#     coefs = pd.DataFrame(model.coef_, X_train.columns)
#     coefs.columns = ["coef"]
#     coefs["abs"] = coefs.coef.apply(np.abs)
#     coefs = coefs.sort_values(by="abs", ascending=False).drop(["abs"], axis=1)
#     
#     plt.figure(figsize=(25, 7))
#     coefs.coef.plot(kind='bar')
#     plt.grid(True, axis='y')
#     plt.hlines(y=0, xmin=0, xmax=len(coefs), linestyles='dashed');
#     
#     return coefs.coef 
# 
# def prepareData(series, lags, test_size):
# 
#     data = pd.DataFrame(series.copy())
#     data.columns = ["y"]
#  
#     for i in lags:
#         data["f{}".format(i)] = data.y.shift(i)
#     
#     y = data.dropna().y
#     X = data.dropna().drop(['y'], axis=1)
#     X_train, X_test, y_train, y_test = timeseries_train_test_split(X, y, test_size=test_size)
# 
#     return X_train, X_test, y_train, y_test
# 
#     
#  
# def create_dataset(dataset, look_back=1):
#     X, Y = [], []
#     for i in range(len(dataset)-look_back-1):
#         a = dataset[i:(i+look_back), 0]
#         X.append(a)
#         Y.append(dataset[i + look_back, 0])
#     return np.array(X), np.array(Y)    
# 
# def test_stationarity(timeseries):
#     print('Result of Dicky_Fuller test:')
#     dftest = adfuller(timeseries, autolag='AIC')
#     dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
#     for key, value in dftest[4].items():
#         dfoutput['Critical value (%s)' %key] = value
#     print(dfoutput)
# 
# 
# def difference(dataset, interval=1):
#     diff = list()
#     for i in range(interval, len(dataset)):
#         value = dataset[i] - dataset[i - interval]
#         diff.append(value)
#     return Series(diff)
# 
#     
#         
# # =============================================================================
# # Data Preparation
# # =============================================================================
# #df = pd.read_excel(r'pollutionDataset.xlsx')
# ##print(df.info())
# #df = df.drop(["?????_?????", "?????_??????"], axis=1)
# #df.columns = ['feature1', 'date', 'psi_co', 'psi_o3', 'psi_no2', 'psi_so2', 'psi_pm10', 'total_psi', 'year_month','year','month', 'feature2']
# ##print(df.head())
# #
# #df['date'] = df['date'].str.replace('-', '/')
# #df['date'] = df['date'].str.split('/')
# #df['year'] = (df['date'].apply(lambda x: x[0])).astype(int)
# #df['month'] = (df['date'].apply(lambda x: x[1]))
# #df['day'] = (df['date'].apply(lambda x: x[-1]))
# #
# #indexNames = df[(df['year'] == 13898) | (df['year'] == 139608) | (df['year'] == 1386)].index
# #df.drop(indexNames , inplace=True)
# #indexNames = df[(df['month'] == '1125') | (df['month'] == '1105') | (df['month'] == '' )].index
# #df.drop(indexNames , inplace=True)
# #
# #df['month'] = ["%02d" % x for x in df['month'].astype(int)]
# #indexNames = df[(df['day'] == '.0')].index
# #
# #df.drop(indexNames , inplace=True)
# #df['day'] = ["%02d" % x for x in df['day'].astype(int)]
# #
# #df['date'] = df['year'].map(str) + '-' + df['month'].map(str) + '-' + df['day'].map(str)
# #df['year_month'] = df['year'].map(str) + '-' + df['month'].map(str)
# #df = df.sort_values('date')
# #df['date'] = pd.to_datetime(df.date, format='%Y/%m/%d', errors = 'ignore')
# #df['year_month'] = pd.to_datetime(df.year_month, format='%Y-%m', errors = 'ignore')
# #df.set_index(['date'])
# #
# ##print(df['year'].value_counts())
# #
# #df.to_csv('airPullotion.csv')
# 
# airPollution = pd.read_csv(r'airPullotion.csv', index_col=['date'], parse_dates=['date'])
# #print(airpollution.info())
# airPollution = airPollution.fillna(method='ffill')
# airPollution = airPollution.sort_values('date')
# 
# monthly_airpollution = airPollution[['year_month', 'total_psi']].groupby(['year_month'], as_index=False).sum()
# 
# #monthly_airpollution.total_psi.astype(float)
# #plt.figure(figsize=(16,8))
# #plt.title('Total psi per month')
# #plt.xlabel('time')
# #plt.ylabel('Total psi')
# #plt.plot(monthly_airpollution.total_psi)
# 
# 
# #res = sm.tsa.seasonal_decompose(monthly_airpollution.total_psi.values, freq=12, model='multiplicative')
# #fig=res.plot()
# #
# #res= sm.tsa.seasonal_decompose(monthly_airpollution.total_psi.values, freq=12, model='additive')
# #fir=res.plot()
# #
# #test_stationarity(monthly_airpollution.total_psi)
# 
# #plt.figure(figsize=(16,16))
# #plt.subplot(311)
# #plt.title('Original')
# #plt.xlabel('Time')
# #plt.ylabel('Total psi')
# #plt.plot(airPollution.total_psi)
# #plt.subplot(312)
# #plt.title('After De-trend')
# #plt.xlabel('Time')
# #plt.ylabel('Total psi')
# new_tps = difference(airPollution.total_psi.values)
# #plt.plot(new_tps)
# #plt.plot()
# 
# #plt.subplot(313)
# #plt.title('After De-seasonalization')
# #plt.xlabel('Time')
# #plt.ylabel('Total psi')
# #new_tps=difference(airPollution.total_psi.values,12)
# #plt.plot(new_tps)
# #plt.plot()
# #
# #test_stationarity(new_tps)
# print(new_tps)
# 
# for i in range(len(new_tps)):
#     airPollution.total_psi.values[i] = new_tps[i]
#     
# #airPollution['total_psi'] = new_tps
# 
# 
# # =============================================================================
# # Air pollution plots for total psi
# # =============================================================================
# #airPollution.groupby('year').plot(x='year_month', y='total_psi')
# 
# #airPollution['total_psi'] = airPollution.total_psi.fillna(airPollution.total_psi - airPollution.total_psi.mean())
# #indexNames = airPollution[airPollution['total_psi'] > 1000.0 ].index
# #airPollution.drop(indexNames , inplace=True)
# 
# airPollution = airPollution[['total_psi']]
# print(airPollution)
# # # =============================================================================
# # # Identify trend/ Level/ Sesonal 
# # # =============================================================================
# # ######################## Moving Avearge ################################
# #airPollution_mean = plotMovingAverage(monthly_airpollution.total_psi, 7, 'mean',plot_intervals=False, plot_anomalies=False)
# #airPollution_std = plotMovingAverage(monthly_airpollution.total_psi, 7,  'std',plot_intervals=False, plot_anomalies=False)
# #print(airPollution_movingAverage)
# #plt.plot(airPollution_mean)
# #plt.show()
# #
# #plt.plot(airpollustion_std)
# #plt.show()
# # ####################### Exponential data ###############################
# #plotExponentialSmoothing(airPollution_total.total_psi, [0.7, 0.05])
# #airPollution_ExpSmoothing = exponential_smoothing(airPollution_total.total_psi, 0.7)
# 
# # ################### Double exponential data #############################
# #plotDoubleExponentialSmoothing(airPollution_total.total_psi, alphas=[0.9, 0.02], betas=[0.9, 0.02])
# #airPollution_DoubleExpSmoothing = double_exponential_smoothing(airPollution_total.total_psi, 0.9, 0.9)
# 
# # # =============================================================================
# # # Corss Validation / Stationarity
# # # =============================================================================
# # ########################## Triple exponetioal data smoothing ###########
# #data = airPollution.total_psi[:-30]
# #x=[0, 0, 0]
# #slen = 30
# # 
# #opt = minimize(timeseriesCVscore, x0=x, 
# #               args=(data, mean_absolute_percentage_error, slen), 
# #               method="TNC", bounds = ((0, 1), (0, 1), (0, 1))
# #              )
# # 
# #alpha_final, beta_final, gamma_final = opt.x
# # 
# #model = HoltWinters(data.values, slen = 30, 
# #                    alpha = alpha_final,
# #                    beta = beta_final,
# #                    gamma = gamma_final,
# #                    n_preds = 100, scaling_factor = 3)
# # 
# #model.triple_exponential_smoothing()
# # 
# #plotHoltWinters(airPollution_total.total_psi, plot_intervals=True, plot_anomalies=True)
# #
# #tsplot(airPollution_total.total_psi, lags=60)
# # 
# #airPollution_diff = airPollution_total.total_psi - airPollution_total.total_psi.shift(30)
# #airPollution_diff = airPollution_diff - airPollution_diff.shift(1)
# #tsplot(airPollution_diff[40+1:], lags=90)
# 
# # ########################## SARIMA #########################
# #ps = range(2, 5)
# #d=1 
# #qs = range(2, 5)
# #Ps = range(0, 2)
# #D=1 
# #Qs = range(0, 2)
# #s = 30
# #
# #parameters = product(ps, qs, Ps, Qs)
# #parameters_list = list(parameters)
# #len(parameters_list)
# #result_table = optimizeSARIMA(parameters_list, d, D, s)
# #
# #p, q, P, Q = result_table.parameters[0]
# #
# #best_model = sm.tsa.statespace.SARIMAX(airPollution.total_psi, order=(p, d, q), seasonal_order=(P, D, Q, s)).fit(disp=-1)
# #best_model.summary()
# # 
# #tsplot(best_model.resid[29+1:], lags=60)
# #plotSARIMA(airPollution_total, best_model, 50)
# 
# # # =============================================================================
# # # Model
# # # =============================================================================
# # ########################## Linear Regression ####################
# data = pd.DataFrame(airPollution.total_psi.copy())
# data.columns = ['y']
# 
# for i in range(3, 60):
#    data['lags: {}'.format(i)] = data.y.shift(i)
#  
# tscv = TimeSeriesSplit(n_splits=3)
#  
# y = data.dropna().y
# X = data.dropna().drop(['y'], axis = 1)
# X_train, X_test, y_train, y_test = timeseries_train_test_split(X, y, test_size = 0.3)
# 
# lr = LinearRegression()
# lr.fit(X_train, y_train)
#  
# plotModelResults(lr,X_train, X_test, 'Linear Regression', plot_intervals=True)
# linear_lags = plotCoefficients(lr)
# 
# # ## =============================================================================
# # ## Scale
# # ## =============================================================================
# # ################### Linear Reg. with scaling #######
# #scaler = preprocessing.StandardScaler()
# #y = data.dropna().y
# #X = data.dropna().drop(['y'], axis=1)
# # 
# #X_train, X_test, y_train, y_test = timeseries_train_test_split(X, y, test_size=0.3)
# # 
# #X_train_scaled = scaler.fit_transform(X_train)
# #X_test_scaled = scaler.transform(X_test)
# # 
# #lr = LinearRegression()
# #lr.fit(X_train_scaled, y_train)
# # 
# #plotModelResults(lr, X_train_scaled, X_test_scaled, 'Linear Reg. with scaling', plot_intervals=True, plot_anomalies=True)
# #plotCoefficients(lr)
# # 
# # ######################### Linear Reg. with acaling and prepared data #########
#  
# #X_train, X_test, y_train, y_test = prepareData(airPollution_movingAverage.total_psi.values, lags=[3,21], test_size=0.3)
# #X_train_scaled = scaler.fit_transform(X_train)
# #X_test_scaled = scaler.transform(X_test)
# # 
# #lr = LinearRegression()
# #lr.fit(X_train_scaled, y_train)
# # 
# #plotModelResults(lr, X_train_scaled, X_test_scaled, 'Linear Reg. with scaling and prepared data', plot_intervals=True, plot_anomalies=True)
# #plotCoefficients(lr)
# 
# ############################## Ridge ###########################
# #from sklearn.linear_model import LassoCV, RidgeCV
# #
# #ridge = RidgeCV(cv=tscv)
# #ridge.fit(X_train_scaled, y_train)
# #
# #plotModelResults(ridge, X_train_scaled, X_test_scaled, 'Ridge',plot_intervals=True, plot_anomalies=True)
# #plotCoefficients(ridge)
# 
# ############################## LASSO ###########################
# #lasso = LassoCV(cv=tscv)
# #lasso.fit(X_train_scaled, y_train)
# #
# #plotModelResults(lasso, X_train_scaled, X_test_scaled, 'Lasso', plot_intervals=True, plot_anomalies=True)
# #plotCoefficients(lasso)
# 
# ############################## Xgboost ###########################
# #X_train, X_test, y_train, y_test = prepareData(airPollution_movingAverage.total_psi.values, 
# #                                               lags=[2, 7, 14, 31, 21, 28, 38, 4,35], test_size=0.3)
# #X_train_scaled = scaler.fit_transform(X_train)
# #X_test_scaled = scaler.transform(X_test)
# #from xgboost import XGBRegressor 
# #
# #my_imputer = Imputer()
# #train_X = my_imputer.fit_transform(X_train_scaled)
# #test_X = my_imputer.transform(X_test_scaled)
# 
# #xgb = XGBRegressor(n_estimators=1000, learning_rate=0.05)
# #xgb.fit(train_X, y_train, early_stopping_rounds=5, 
# #             eval_set=[(X_test, y_test)], verbose=False)
# 
# #xgb = XGBRegressor()
# #xgb.fit(train_X, y_train)
# #
# #plotModelResults(xgb, X_train_scaled, X_test_scaled, 'Xgboost', plot_intervals=True, plot_anomalies=True)
# 
# ################################# LSTM #############################
# #np.random.seed(7)
# #df = airPollution_movingAverage
# ##df['total_psi'] = pd.to_numeric(df['total_psi'], errors='coerce')
# #df = df.dropna(subset=['total_psi'])
# #dataset = df.total_psi.values #numpy.ndarray
# #dataset = dataset.astype('float32')
# #
# #dataset = np.reshape(dataset, (-1, 1))
# #scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
# #dataset = scaler.fit_transform(dataset)
# #train_size = int(len(dataset) * 0.80)
# #test_size = len(dataset) - train_size
# #train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
# #    
# #look_back = 30
# #X_train, y_train = create_dataset(train, look_back)
# #X_test, y_test = create_dataset(test, look_back)
# #
# #X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
# #X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
# #
# #model = Sequential()
# #
# #model.add(LSTM(100, input_shape=(X_train.shape[1], X_train.shape[2])))
# #model.add(Dropout(0.2))
# #model.add(Dense(1))
# #model.compile(loss='mean_squared_error', optimizer='adam')
# #
# #history = model.fit(X_train, y_train, epochs=20, batch_size=70, validation_data=(X_test, y_test), 
# #                    callbacks=[EarlyStopping(monitor='val_loss', patience=10)], verbose=1, shuffle=False)
# #
# #model.summary()
# #train_predict = model.predict(X_train)
# #test_predict = model.predict(X_test)
# #
# #train_predict = scaler.inverse_transform(train_predict)
# #y_train = scaler.inverse_transform([y_train])
# #test_predict = scaler.inverse_transform(test_predict)
# #y_test = scaler.inverse_transform([y_test])
# #
# #plt.legend(loc="best")
# #plt.tight_layout()
# #plt.grid(True);
# #
# #error = mean_absolute_percentage_error(y_test[0], test_predict[:,0])
# #
# #aa=[x for x in range(200)]
# #plt.figure(figsize=(15,7))
# #plt.plot(aa, y_test[0][:200], marker='.', label="actual")
# #plt.plot(aa, test_predict[:,0][:200], 'r', label="prediction")
# #plt.title("Mean absolute percentage error {0:.2f}%/Plot name: {1}".format(error, 'LSTM'))
# #plt.tight_layout()
# #sns.despine(top=True)
# #plt.subplots_adjust(left=0.07)
# #plt.ylabel('Total psi', size=15)
# #plt.xlabel('Time step', size=15)
# #plt.legend(fontsize=15)
# #plt.grid(True)
# #plt.show();
# 
# 
# 
# =============================================================================

# import pandas as pd
# import matplotlib.pyplot as plt
# import jdatetime
# from datetime import datetime

# train = pd.read_csv('airPullotion-Train.csv', index_col=['date'], parse_dates=['date'])

# train = train.fillna(method='ffill')
# airPollution = train.sort_values('date')
# airPollution['total_psi'] = airPollution.total_psi.fillna(airPollution.total_psi - airPollution.total_psi.mean())
# indexNames = airPollution[airPollution['total_psi'] > 1000.0 ].index
# airPollution.drop(indexNames , inplace=True)
# airPollution = airPollution[['total_psi']]

# ts = airPollution['total_psi'] 
# plt.figure(figsize=(50,6)) 
# plt.plot(ts, label='air pollution') 
# plt.title('Time Series') 
# plt.xlabel("Time(date)") 
# plt.ylabel("Air pollution") 
# plt.legend(loc='best')



# from pandas.plotting import scatter_matrix

# attributes = ['psi_co',	'psi_o3', 'psi_no2', 'psi_so2', 'psi_pm10',	'total_psi']
# scatter_matrix(train[attributes], figsize=(12, 8))


# corr_matrix = train.corr()
# print(corr_matrix["psi_o3"].sort_values(ascending=False))

# train  = pd.read_csv('airPollution-Train.csv')
# test = pd.read_csv('airPollution-Test.csv')



# print("Summary Base on All Observation")
# print(train.psi_o3.describe())


# =============================================================================
# 2010->jan->(Max(week(1)), Max(week(2)), Max(week(3)), Max(week(4)))/4
# =============================================================================


# train['Date'] = [jdatetime.date(row.year, row.month, row.day).togregorian() for 
#                   index, row in train.iterrows()]
# test['Date'] = [jdatetime.date(row.year, row.month, row.day).togregorian() for 
#                   index, row in test.iterrows()]

# train['Date'] = pd.to_datetime(train.Date,format='%Y-%m-%d')
# test['Date'] = pd.to_datetime(test.Date,format='%Y-%m-%d')

# for i in (train, test):
#     i['year']=i.Date.dt.year
#     i['month']=i.Date.dt.month 
#     i['day']=i.Date.dt.day
    
# train.index = train['Date']
# ts = train['psi_o3'] 
# plt.figure(figsize=(16,8)) 
# plt.plot(ts, label='O3') 
# plt.title('Time Series') 
# plt.xlabel("Time(year-month)") 
# plt.ylabel("Ground-level Ozone") 
# plt.legend(loc='best')




# train = train[['psi_o3', 'psi_pm10', 'psi_so2', 'psi_no2']]
# df = pd.DataFrame(train,
#                   columns=['psi_o3', 'psi_pm10', 'psi_so2', 'psi_no2'])
# boxplot = df.boxplot(column=['psi_o3', 'psi_pm10', 'psi_so2', 'psi_no2'])

# import seaborn as sns

# sns.distplot(train.psi_o3)
# sns.distplot(train.psi_pm10)
# sns.distplot(train.psi_so2)
# sns.distplot(train.psi_no2)


import jdatetime
import numpy as np
import pandas as pd                   # For mathematical calculations 
import matplotlib.pyplot as plt  # For plotting graphs 
from datetime import date    # To access datetime 
from pandas import Series        # To work on series 
import warnings                   # To ignore the warnings warnings.filterwarnings("ignore")

airPollution_train  = pd.read_csv('airPollution_ahvaz.csv')

airPollution_train['date'] = [jdatetime.date(row.year, row.month, row.day).togregorian() for 
                  index, row in airPollution_train.iterrows()]
train = airPollution_train[airPollution_train['date'] < date(2017,1,1)]
valid = airPollution_train[(date(2015,12,31) < airPollution_train['date']) & 
                            (airPollution_train['date']< date(2017,1,1))]

################### Naive ###############
dd = np.asarray(train.psi_so2) 
y_hat = valid.copy() 
y_hat['naive'] = dd[len(dd)-1] 
plt.figure(figsize=(12,8)) 
plt.plot(train.index, train['psi_so2'], label='Train') 
plt.plot(valid.index,valid['psi_so2'], label='Valid') 
plt.plot(y_hat.index,y_hat['naive'], label='Naive Forecast') 
plt.legend(loc='best') 
plt.title("Naive Forecast") 
plt.show()


from sklearn.metrics import mean_squared_error 
from math import sqrt 
rms = sqrt(mean_squared_error(valid.psi_so2, y_hat.naive)) 
print(rms)


################# Moving average ############

y_hat_avg = valid.copy() 
y_hat_avg['moving_avg_forecast'] = train['psi_pm10'].rolling(10).mean().iloc[-1] # average of last 10 observations. 
plt.figure(figsize=(15,5)) 
plt.plot(train['psi_pm10'], label='Train') 
plt.plot(valid['psi_pm10'], label='Valid') 
plt.plot(y_hat_avg['moving_avg_forecast'], label='Moving Average Forecast using 10 observations') 
plt.legend(loc='best') 
plt.show() 
y_hat_avg = valid.copy() 
y_hat_avg['moving_avg_forecast'] = train['psi_pm10'].rolling(20).mean().iloc[-1] # average of last 20 observations. 
plt.figure(figsize=(15,5)) 
plt.plot(train['psi_pm10'], label='Train') 
plt.plot(valid['psi_pm10'], label='Valid') 
plt.plot(y_hat_avg['moving_avg_forecast'], label='Moving Average Forecast using 20 observations') 
plt.legend(loc='best') 
plt.show() 
y_hat_avg = valid.copy() 
y_hat_avg['moving_avg_forecast'] = train['psi_pm10'].rolling(50).mean().iloc[-1] # average of last 50 observations. 
plt.figure(figsize=(15,5)) 
plt.plot(train['psi_pm10'], label='Train') 
plt.plot(valid['psi_pm10'], label='Valid') 
plt.plot(y_hat_avg['moving_avg_forecast'], label='Moving Average Forecast using 50 observations') 
plt.legend(loc='best') 
plt.show()

rms = sqrt(mean_squared_error(valid.psi_pm10, y_hat_avg.moving_avg_forecast)) 
print(rms)

############## Simple Expo #############

from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt 

y_hat_avg = valid.copy() 
fit2 = SimpleExpSmoothing(np.asarray(train['psi_pm10'])).fit(smoothing_level=0.6,optimized=False) 
y_hat_avg['SES'] = fit2.forecast(len(valid)) 
plt.figure(figsize=(16,8)) 
plt.plot(train['psi_pm10'], label='Train') 
plt.plot(valid['psi_pm10'], label='Valid') 
plt.plot(y_hat_avg['SES'], label='SES') 
plt.legend(loc='best') 
plt.show()

rms = sqrt(mean_squared_error(valid.psi_pm10, y_hat_avg.SES)) 
print(rms)

######## Holt Linear ###########

import statsmodels.api as sm 
sm.tsa.seasonal_decompose(train.psi_pm10, freq=12).plot() 
result = sm.tsa.stattools.adfuller(train.psi_pm10) 
plt.show()



y_hat_avg = valid.copy() 
fit1 = ExponentialSmoothing(np.asarray(train['psi_no2']) ,seasonal_periods=7 ,trend='add', seasonal='add').fit() 
y_hat_avg['Holt_Winter'] = fit1.forecast(len(valid)) 
plt.figure(figsize=(16,8)) 
plt.plot( train['psi_no2'], label='Train') 
plt.plot(valid['psi_no2'], label='Valid') 
plt.plot(y_hat_avg['Holt_Winter'], label='Holt_Winter') 
plt.legend(loc='best') 
plt.show()

rms = sqrt(mean_squared_error(valid.psi_no2, y_hat_avg.Holt_Winter)) 
print(rms)

################### ARIMA #############
from statsmodels.tsa.stattools import adfuller 
def test_stationarity(timeseries):
        #Determing rolling statistics
    rolmean = timeseries.rolling(24).mean() # 24 hours on each day
    rolstd = timeseries.rolling(24).std()
        #Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
        #Perform Dickey-Fuller test:
    print ('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])

    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)

from matplotlib.pylab import rcParams 

rcParams['figure.figsize'] = 20,10
test_stationarity(train['psi_pm10'])

######## removing trend
Train_log = np.log(train['psi_o3']) 
valid_log = np.log(valid['psi_pm10'])
moving_avg = Train_log.rolling(24).mean()
plt.plot(Train_log) 
plt.plot(moving_avg, color = 'red') 
plt.show()

####### make our series staitionary
train_log_moving_avg_diff = Train_log - moving_avg
train_log_moving_avg_diff.dropna(inplace = True) 
test_stationarity(train_log_moving_avg_diff)

train_log_moving_avg_diff.dropna(inplace = True) 
test_stationarity(train_log_moving_avg_diff)

#####Differencing can help to make the series stable and eliminate the trend.
train_log_diff = Train_log - Train_log.shift(1) 
test_stationarity(train_log_diff.dropna())

#### remove seasonality
from statsmodels.tsa.seasonal import seasonal_decompose 
decomposition = seasonal_decompose(pd.DataFrame(Train_log).psi_pm10.values, freq = 24) 

trend = decomposition.trend 
seasonal = decomposition.seasonal 
residual = decomposition.resid 

plt.subplot(411) 
plt.plot(Train_log, label='Original') 
plt.legend(loc='best') 
plt.subplot(412) 
plt.plot(trend, label='Trend') 
plt.legend(loc='best') 
plt.subplot(413) 
plt.plot(seasonal,label='Seasonality') 
plt.legend(loc='best') 
plt.subplot(414) 
plt.plot(residual, label='Residuals') 
plt.legend(loc='best') 
plt.tight_layout() 
plt.show()


# #######  check stationarity of residuals
train_log_decompose = pd.DataFrame(residual) 
train_log_decompose['date'] = Train_log.index 
train_log_decompose.set_index('date', inplace = True) 
train_log_decompose.dropna(inplace=True) 
test_stationarity(train_log_decompose[0])


from statsmodels.tsa.stattools import acf, pacf 
lag_acf = acf(train_log_diff.dropna(), nlags=25) 
lag_pacf = pacf(train_log_diff.dropna(), nlags=25, method='ols')

plt.plot(lag_acf) 
plt.axhline(y=0,linestyle='--',color='gray') 
plt.axhline(y=-1.96/np.sqrt(len(train_log_diff.dropna())),linestyle='--',color='gray') 
plt.axhline(y=1.96/np.sqrt(len(train_log_diff.dropna())),linestyle='--',color='gray') 
plt.title('Autocorrelation Function') 
plt.show() 
plt.plot(lag_pacf) 
plt.axhline(y=0,linestyle='--',color='gray') 
plt.axhline(y=-1.96/np.sqrt(len(train_log_diff.dropna())),linestyle='--',color='gray') 
plt.axhline(y=1.96/np.sqrt(len(train_log_diff.dropna())),linestyle='--',color='gray') 
plt.title('Partial Autocorrelation Function') 
plt.show()


from statsmodels.tsa.arima_model import ARIMA
model = ARIMA(Train_log, order=(2, 1, 0))  # here the q value is zero since it is just the AR model 
results_AR = model.fit(disp=-1)  
plt.plot(train_log_diff.dropna(), label='original') 
plt.plot(results_AR.fittedvalues, color='red', label='predictions') 
plt.legend(loc='best') 
plt.show()

AR_predict=results_AR.predict() 
AR_predict=AR_predict.cumsum().shift().fillna(0) 
AR_predict1=pd.Series(np.ones(valid.shape[0]) * np.log(valid['psi_pm10']), index = valid.index) 
AR_predict1=AR_predict1.add(AR_predict,fill_value=0) 
AR_predict = np.exp(AR_predict1)
AR_predict = AR_predict[AR_predict.index>2183]
plt.plot(valid['psi_pm10'], label = "Valid") 
plt.plot(AR_predict, color = 'red', label = "Predict") 
plt.legend(loc= 'best') 
plt.title('RMSE: %.4f'% (np.sqrt(np.dot(AR_predict, valid['psi_pm10']))/valid.shape[0])) 
plt.show()

model = ARIMA(Train_log, order=(2, 1, 2))  
results_ARIMA = model.fit(disp=-1)  
# plt.plot(train_log_diff.dropna(),  label='original') 
# plt.plot(results_ARIMA.fittedvalues, color='red', label='predicted') 
# plt.legend(loc='best') 
# plt.show()

def check_prediction_diff(predict_diff, given_set):
    predict_diff= predict_diff.cumsum().shift().fillna(0)
    predict_base = pd.Series(np.ones(given_set.shape[0]) * np.log(given_set), index = given_set.index)
    predict_log = predict_base.add(predict_diff,fill_value=0)
    predict = np.exp(predict_log)
    predict = predict[predict.index>2183]
    plt.plot(given_set, label = "Given set")
    # plt.plot(predict, color = 'red', label = "Predict")
    # plt.legend(loc= 'best')
    plt.title('RMSE: %.4f'% (np.sqrt(np.dot(predict, given_set))/given_set.shape[0]))
    plt.show()

def check_prediction_log(predict_log, given_set):
    predict = np.exp(predict_log)
    predict = predict[predict.index>2183]
    plt.plot(given_set, label = "Given set")
    # plt.plot(predict, color = 'red', label = "Predict")
    # plt.legend(loc= 'best')
    plt.title('RMSE: %.4f'% (np.sqrt(np.dot(predict, given_set))/given_set.shape[0]))
    plt.show()


ARIMA_predict_diff=results_ARIMA.predict()
check_prediction_diff(ARIMA_predict_diff, valid.psi_o3)



################# SAIMAX ###################
import statsmodels.api as sm
y_hat_avg = valid.copy() 
fit1 = sm.tsa.statespace.SARIMAX(train.psi_o3, order=(2, 1, 4),seasonal_order=(0,1,1,7)).fit() 
predict = fit1.predict() 
temp = predict[predict.index>2183]
y_hat_avg['SARIMA'] = temp


plt.figure(figsize=(16,8)) 
plt.plot(train['psi_o3'], label='Train') 
plt.plot(valid['psi_o3'], label='Valid') 
plt.plot(y_hat_avg['SARIMA'], label='SARIMA') 
plt.legend(loc='best') 
plt.show()


rms = sqrt(mean_squared_error(valid.psi_o3, y_hat_avg.SARIMA)) 
print(rms)




from xgboost import XGBRegressor 


X_train = train.index
y_train = train.psi_o3
X_test = valid.index
y_test = valid.psi_o3

reg = XGBRegressor(n_estimators=1000)
reg.fit(X_train, y_train, eval_set=[(X_train, y_train), 
                                    (X_test, y_test)], 
        early_stopping_rounds=50,verbose=False)




predict = reg.predict(X_test)
rms = sqrt(mean_squared_error(y_test, predict)) 
print(rms)



