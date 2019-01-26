import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import pyplot
#import datetime
import warnings
#from datetime import datetime
#import numpy as np
#from pandas import series
from pandas.tools.plotting import autocorrelation_plot
warnings.filterwarnings("ignore")
#%matplotlib inline 
Superstore_df=pd.read_csv('Superstore.csv')
retail=Superstore_df.copy()
print(Superstore_df.columns)
print(Superstore_df.dtypes)
print(Superstore_df.shape)
Superstore_df.rename(columns = {'Order Date':'Order_Date'}, inplace=True)
retail.rename(columns = {'Order Date':'Order_Date'}, inplace=True)
print(Superstore_df.columns)
Superstore_df['Order_Date'] = pd.to_datetime(Superstore_df.Order_Date,format= '%m/%d/%Y')
retail['Order_Date']= pd.to_datetime(Superstore_df.Order_Date,format= '%m/%d/%Y')
for i in (Superstore_df, retail):
    i['year']=i.Order_Date.dt.year
    i['month']=i.Order_Date.dt.month
    i['day']=i.Order_Date.dt.day
Superstore_df['day of week']=Superstore_df['Order_Date'].dt.dayofweek
temp = Superstore_df['Order_Date']
def applyer(row):
    if row.dayofweek == 5 or row.dayofweek == 6:
        return 1
    else:
        return 0
temp2 = Superstore_df['Order_Date'].apply(applyer)
Superstore_df['weekend']=temp2
Superstore_df.index = Superstore_df['Order_Date'] # indexing the Datetime to get the time period on the x-axis.
df=Superstore_df.drop(['Row ID','Order ID','Ship Date','Ship Mode', 'Customer ID',  'Customer Name', 'Segment', 'Country', 'City', 'State', 'Postal Code', 'Region', 'Product ID', 'Category', 'Sub-Category', 'Product Name', 'Sales', 'Quantity', 'Discount'], axis=1, inplace=True)
           # drop ID variable to get only the Datetime on x-axis.
ts = Superstore_df['Profit']
plt.figure(figsize=(16,8))
plt.plot(ts, label='Store Profit')
plt.title('Time Series')
plt.xlabel("Time(year-month)")
plt.ylabel("Profit")
plt.legend(loc='best')
#(Date= Superstore_df['Order_Date']
#Date.head()
#Date=pd.to_datetime(Superstore_df.Order_Date,format= '%m/%d/%Y')
#Superstore_df.shape
##from datetime import datetime
#ordered_data = sorted(Date.items(), reverse=False)
#
#print(ordered_data)
##Superstore_df.rename(columns = {'Order ID':'Order_ID'}, inplace=True)
##Superstore_df.Order_ID.duplicated()(To delete those items which were returned but data seems inconsitent)
##Superstore_df.drop(Superstore_df.loc[9996:10297, :], axis=0,inplace=True)
#y=Superstore_df['Profit']
##from sklearn.model_selection import train_test_split
##x_train,x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=10)
##print(x_train.shape, y_train.shape)
##len(x_train)
##len(x_test)  
##from sklearn.linear_model import LinearRegression
##clf = LinearRegression()#object of above class
##clf.fit(x_train,y_train)
##clf.predict(x_test)
##y_test
##clf.score[x_test, y_test]
#Train=Date.ix['2014-01-03':'2017-09-18']
#Valid=Date.ix['2017-09-19':'2017-12-30']
#plt.plot(Train.index, Train['Count'], figsize=(15,8), title= 'Daily Profit', fontsize=14, label='training')
#plt.plot(Valid.index, Valid['Count'], figsize=(15,8), title= 'Daily Profit', fontsize=14, label='testing')
#plt.xlabel("Order_Date")
#plt.ylabel("Profit")
#plt.legend(loc='best')
#plt.show())
#variation=pd.pivot_table(Superstore_df, index=Superstore_df.index.month, columns=Superstore_df.index.year, values='Profit', aggfunc='sum')
#variation.plot.bar()
#from statsmodels.tsa.stattools import adfuller
#def test_stationarity(timeseries):
#  rolmean = pd.rolling_mean(timeseries, window=24) # 24 hours on each day
#  rolstd = pd.rolling_std(timeseries, window=24)
#  orig = plt.plot(timeseries, color='blue',label='Original')
#  mean = plt.plot(rolmean, color='red', label='Rolling Mean')
#  std = plt.plot(rolstd, color='black', label = 'Rolling Std')
#  plt.legend(loc='best')
#  plt.title('Rolling Mean & Standard Deviation')
#  plt.show(block=False)
autocorrelation_plot(Superstore_df)

pyplot.show()