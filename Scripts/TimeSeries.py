import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

#Generate sample time series data
np.random.seed(0)
dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
values = np.cumsum(np.random.randn(100))
df = pd.DataFrame({'Date': dates, 'Value': values})
df.set_index('Date', inplace=True)

#Plot the time series data
plt.figure(figsize=(10, 4))
plt.plot(df.index, df['Value'])
plt.title('Sample Time Series Data')
plt.xlabel('Date')
plt.ylabel('Value')
plt.show()

#Decompose the time series into trend, seasonality, and residual components
decomposition = sm.tsa.seasonal_decompose(df['Value'], model='additive')
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid


#Plot the decomposed components
plt.figure(figsize=(10, 8))
plt.subplot(411)
plt.plot(df['Value'], label='Original')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend, label='Trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal, label='Seasonality')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual, label='Residuals')
plt.legend(loc='best')
plt.tight_layout()
plt.show()

#Perform autocorrelation analysis
sm.graphics.tsa.plot_acf(df['Value'].values, lags=30)
plt.title('Autocorrelation')
plt.xlabel('Lag')
plt.ylabel('Autocorrelation')
plt.show()

#Perform partial autocorrelation analysis
sm.graphics.tsa.plot_pacf(df['Value'].values, lags=30)
plt.title('Partial Autocorrelation')
plt.xlabel('Lag')
plt.ylabel('Partial Autocorrelation')
plt.show()

#Perform time series forecasting using ARIMA model
model = sm.tsa.ARIMA(df['Value'], order=(1, 1, 1))
results = model.fit()
forecast = results.predict(start=100, end=120)

#Plot the original data and the forecasted values
plt.figure(figsize=(10, 4))
plt.plot(df['Value'], label='Original')
plt.plot(forecast, label='ARIMA Forecast')
plt.title('Time Series Forecasting (ARIMA)')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend(loc='best')
plt.show()

#Perform time series forecasting using SARIMA model
model_sarima = sm.tsa.SARIMAX(df['Value'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
results_sarima = model_sarima.fit()
forecast_sarima = results_sarima.predict(start=100, end=120)

#Plot the original data and the SARIMA forecasted values
plt.figure(figsize=(10, 4))
plt.plot(df['Value'], label='Original')
plt.plot(forecast_sarima, label='SARIMA Forecast')
plt.title('Time Series Forecasting (SARIMA)')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend(loc='best')
plt.show()

#Perform exponential smoothing
model_ets = sm.tsa.ExponentialSmoothing(df['Value'], trend='add', seasonal='add', seasonal_periods=7)
results_ets = model_ets.fit()
forecast_ets = results_ets.predict(start=100, end=120)

#Plot the original data and the exponential smoothing forecasted values
plt.figure(figsize=(10, 4))
plt.plot(df['Value'], label='Original')
plt.plot(forecast_ets, label='ETS Forecast')
plt.title('Time Series Forecasting (Exponential Smoothing)')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend(loc='best')
plt.show()

#Perform vector autoregression (VAR)
data = pd.DataFrame({'Value': df['Value'], 'Value Lag': df['Value'].shift(1)})
data.dropna(inplace=True)
model_var = sm.tsa.VAR(data)
results_var = model_var.fit()
forecast_var = results_var.forecast(data.values[-1:], steps=20)
print(forecast_var)

#Extract forecasted values for 'Value' variable
forecast_var_value = forecast_var[:, 0]
#In this line, we extract the forecasted values for the 'Value' variable from the forecast_var array. Since forecast_var contains the forecasted values for multiple variables in the VAR model, we use slicing with [:, 0] to extract the values only for the first variable (which is 'Value' in this case).

#Create a date range with the same length as forecasted values
date_range = pd.date_range(start=df.index[-1] + pd.DateOffset(days=1), periods=20, freq='D')

#Plot the original data and the VAR forecasted values
plt.figure(figsize=(10, 4))
plt.plot(df['Value'], label='Original')
plt.plot(date_range, forecast_var_value, label='VAR Forecast')
plt.title('Time Series Forecasting (VAR)')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend(loc='best')
plt.show()