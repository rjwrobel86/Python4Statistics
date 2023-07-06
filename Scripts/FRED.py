#FRED 
import pandas as pd
import matplotlib.pyplot as plt
from fredapi import Fred

fred = Fred(api_key='4df2eca0f8e50e893ab9de7a4c006757')

series_set = ['GDP','UNRATE','CPIAUCSL','FEDFUNDS']

df = pd.DataFrame()

for i in series_set:
    data = fred.get_series(series_id=i, observation_start='2000-1-1',observation_end='2022-12-31', frequency="a")
    df[i] = data
    
df.head()

plt.plot(df['GDP'])

#Function to get series 
#fred.get_series(PARAMETERS-GO-HERE)

#Some Paramerters

#series_id="GDP" 
#observation_start="1990-01-01" 
#observation_end="2012-01-01" 
#frequency="Annual" 
#units="Billions of Chained 2009 Dollars" 
#seasonal_adjustment="Not Seasonally Adjusted" 

fig,axs = plt.subplots(2, 2, figsize=(10,5))
fig.suptitle('Select Indicators of the US Economy 2000-2022')
plt.subplots_adjust(hspace=0.5)

axs[0][0].plot(df['GDP'])
axs[0][0].set_title("Gross Domestic Product")
axs[0][0].set_ylabel("Billions of $")
axs[0][0].set_xlabel("Year")

axs[0][1].plot(df['UNRATE'])
axs[0][1].set_title("Unemployment Rate")
axs[0][1].set_ylabel("% Unemployed")
axs[0][1].set_xlabel("Year")

axs[1][0].plot(df['CPIAUCSL'])
axs[1][0].set_title("Consumer Price Index")
axs[1][0].set_ylabel("CPI Index - Base Year 1984")
axs[1][0].set_xlabel("Year")

axs[1][1].plot(df['FEDFUNDS'])
axs[1][1].set_title("Federal Funds Rate")
axs[1][1].set_ylabel("Interest Rate as %")
axs[1][1].set_xlabel("Year")