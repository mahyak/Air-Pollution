# Air-Pollution

Air pollution forecasting
It is undeniable that Air pollution has negative effects on health, which science says it because of air pollution in big cities health issues and traffic restrictions are contiguously increasing. In this project I worked on prediction the air pollution in a metropolitan area. There are four major pollutants that cause threat to human health: O3, PM10, SO2 and NO2. These pollutants have been considered as a parameter to predict the air pollution. SARIMA and Xgboost has been used in order to predict air pollution for the last month, and also for evaluating the model, Mean Absolute Percentage Error (MAPE) has been chosen which is an interpretable metric (it has the same unit of measurement as the initial series to measure the quality of our predictions, [0, +∞). Hence, the results show that XGBoost can predicts our data more accurate. There is a saying based on forecasting: “You ‘ll never get it right but you can always get it less wrong”.

Discover and visualize the data to gain insights 
In this work I focused on 4 major pollutants: O3 (Ground-level Ozone, PM10 (Particulate Matter (soot and dust)), SO2 (Sulphur Dioxide), NO2 (Nitrogen Dioxide)

Time series on our data set
I started with monthly air pollution. The time series data is taken from average of every month (31 days) from daily maximum value of pollutant (Max(all observation in a day)).

Train set: The range starts from 2010-01-01 till 2015-12-31.
Valid set: The range starts from 2016-01-01 till 2016-12-31.
The forecast will be for 2017-01-01 till 2017-12-31.

The result of prediction for each algorithm evaluate by MAPE measure are as bellow:

<img src='images/MAPE measure.png'>

<a href='http://mahyak.ca' target='_blanck'>Read More ...</a>
