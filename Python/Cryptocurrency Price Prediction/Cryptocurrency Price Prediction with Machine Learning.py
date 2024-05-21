#!/usr/bin/env python
# coding: utf-8

# <h1 style="text-align:center;">Cryptocurrency Price Prediction </h1>
# 

# * <u>**Author**</u> **:**[Younes Dahami](https://www.linkedin.com/in/dahami/)

# ![alt](crypto_prediction.png)

# 
# 
# Cryptocurrency is a digital medium of exchange that is encrypted and decentralized. Many people use cryptocurrencies as a form of investing because it gives great returns even in a short period. Bitcoin, Ethereum, and Binance Coin are among the popular cryptocurrencies today. If you want to know how to predict the future prices of any cryptocurrency with machine learning, this notebook is for you. In this notebook, I will walk through the task of cryptocurrency price prediction with machine learning using Python.

# # Cryptocurrency Price Prediction with Machine Learning
# 
# Predicting the price of cryptocurrencies is a popular case study in the data science community. The prices of stocks and cryptocurrencies are influenced by more than just the number of buyers and sellers. Nowadays, changes in financial policies by governments regarding cryptocurrencies also impact their prices. Additionally, public sentiment toward a specific cryptocurrency or a personality endorsing it (such as Elon Musk with Dogecoin) can lead to significant buying and selling activity, resulting in price fluctuations.
# 
# In short, while buying and selling directly affect cryptocurrency prices, these trends are driven by various factors. Machine learning can be effective for predicting cryptocurrency prices in situations where price changes are influenced by historical price trends that people consider before trading. In the following section, we will explore how to predict bitcoin prices (one of the most popular cryptocurrencies) for the next 30 days.

# # Cryptocurrency Price Prediction using Python
# 
# We’ll begin the task of cryptocurrency price prediction by importing the necessary Python libraries and the required dataset. For this task, we will collect the latest Bitcoin price data from [Yahoo Finance](https://finance.yahoo.com/?guccounter=1&guce_referrer=aHR0cHM6Ly90aGVjbGV2ZXJwcm9ncmFtbWVyLmNvbS8&guce_referrer_sig=AQAAAIZlwaqDOhpVmIqlI99UBCwsUuEbv-htbyH8HfuvPGOOCIJ0nL4iqrm5sXBA6ZgU76jlB1EFAPEfLfEsYj68acb81n5dkqrSkQIVN9w3U7eGu_kIxFJJPpp-S7sGTTsk_KtWrWK7LLA_qVPbO1Ebb3PJ053HpT0eJp_Rirp7uZtj) using the [yfinance API](https://thecleverprogrammer.com/2021/12/21/get-stock-price-data-using-python/). This will allow us to collect the latest data each time we run the code.

# In[1]:


#!pip install yfinance
#!pip install forex_python
#!pip install autots


# In[2]:


import pandas as pd
import yfinance as yf
import datetime
from datetime import date, timedelta

# To convert currencies (i.e. USD to MAD)
from forex_python.converter import CurrencyRates

# to visualize
import plotly.graph_objects as go

# The AutoTS library for time series analysis
from autots import AutoTS

import warnings
warnings.filterwarnings('ignore')


# In[3]:


# Today's date
today = date.today()
print(today)


# In[4]:


type(today)


# In[5]:


# Convert today to a string 
d1 = today.strftime("%Y-%m-%d")
end_date = d1


# Calculating past date from 2 years ago (365*2=730)
d2 = date.today() - timedelta(days = 730)
d2 = d2.strftime("%Y-%m-%d")
start_date = d2

print(f"Start date is :{start_date}")
print(f"End date is : {end_date}")


# # 1) Uploading the data
# 

# In[6]:


data = yf.download("BTC-USD",
                  start = start_date,
                  end = end_date,
                  progress = False)

data.head()


# In[7]:


#Let's change the currency from USD to MAD.

# Get exchange rate data for USD to MAD
#c = CurrencyRates()
#exchange_rates = c.get_rates('USD')

# Define the conversion function
#def convert_to_mad(usd_price, exchange_rate):
#    return usd_price * exchange_rate

# Convert Bitcoin prices from USD to MAD
#data['BTC-MAD'] = data['Close'].apply(lambda x: convert_to_mad(x, exchange_rates['MAD']))

# Print the first few rows of the data
#print(data.head())


# In[8]:


data["Date"] = data.index
data = data[["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]]
data.reset_index(drop = True, inplace = True)
data.head()


# In the above code, I have collected the latest Bitcoin price data for the past 730 days (last 2 years) and prepared it for any data science task. Now, let’s examine the shape of this dataset to confirm that we are working with 730 rows :

# In[9]:


data.shape


# In[10]:


data.columns


# In[11]:


data.info()


# In[12]:


data.isnull().sum()


# So the dataset contains 731 rows. Now let’s visualize the change in bitcoin prices till today by using a candlestick chart:

# In[13]:


figure = go.Figure(data=[go.Candlestick(x = data["Date"],
                                        open= data["Open"],
                                       high = data["High"],
                                       low = data["Low"],
                                       close = data["Close"])])

figure.update_layout(title = "Bitcoin Price Analysis",
                     xaxis_rangeslider_visible = True) 

figure.show()


# The `Close` column in the dataset contains the values we need to predict.

# In[14]:


data["Close"].describe()


# We can see that during the last 2 years Bitcoin achieved a max value of 73,083  USD, and a minimum value of 15,787  USD. Let’s have a look at the correlation of all the columns in the data concerning the Close column :

# In[15]:


correlation = data.corr()
print(correlation["Close"].sort_values(ascending=False))


# # 2) Cryptocurrency Price Prediction Model
# 
# Predicting the future prices of cryptocurrency is based on the problem of Time series analysis. The [AutoTS](https://thecleverprogrammer.com/2021/04/19/autots-in-python-tutorial/) library in Python is one of the best libraries for time series analysis. So here I will be using the `AutoTS` library to predict the bitcoin prices for the next 30 days :

# In[16]:


# Building the model
model = AutoTS(forecast_length = 30,
              frequency = "infer",
              ensemble = "simple")

# Fitting the model
model = model.fit(data,
                  date_col= "Date",
                 value_col = "Close",
                 id_col = None)

# Making predictions
predictions = model.predict()
predictions


# In[17]:


forecast = predictions.forecast
print(forecast)


# So this is how we can use machine learning to predict the price of any cryptocurrency by using the Python programming language.

# # Conclusion
# 
# Buying and selling influence the price of any cryptocurrency, but these trends depend on various factors. Using machine learning for cryptocurrency price prediction is effective only when price changes are driven by historical prices that people consider before buying and selling their cryptocurrency.

# # Change Log
# 
# | Date (YYYY-MM-DD) | Version | Changed By      | Change Description      |
# | ----------------- | ------- | -------------   | ----------------------- |
# | 2023-05-16       | 1.0     | Younes Dahami   |  initial version |
# | 2024-05-21       | 1.0     | Younes Dahami   |  initial version |

# In[ ]:




