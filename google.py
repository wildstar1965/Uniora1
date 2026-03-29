#!/usr/bin/env python
# coding: utf-8

# In[68]:


import numpy as np
import pandas as pd


# In[69]:


data = pd.read_csv('/content/AirQualityUCI.csv', sep = ';', decimal = ',')
data.head(10)


# In[70]:


data = data.iloc[:, :-2]
data.info()


# In[71]:


data = data.head(9357)
data.tail()


# In[72]:


data.isin([-200]).sum(axis = 0)


# In[73]:


data = data.replace(to_replace = -200, value = np.nan)
data.isnull().sum()


# In[74]:


data.head()


# In[75]:


mean = data.mean(numeric_only=True)
print(mean)


# In[76]:


data = data.fillna(mean)


# In[77]:


data.tail()


# In[78]:


data.isnull().sum()


# In[79]:


import numpy as np
import pandas as pd

# CO sub-index
def aqi_co(co):
    if pd.isna(co):
        return np.nan
    if co <= 4.4:
        return (50/4.4) * co
    elif co <= 9.4:
        return 50 + ((100-50)/(9.4-4.4)) * (co-4.4)
    elif co <= 12.4:
        return 100 + ((150-100)/(12.4-9.4)) * (co-9.4)
    else:
        return 200


# NO2 sub-index
def aqi_no2(no2):
    if pd.isna(no2):
        return np.nan
    if no2 <= 53:
        return (50/53) * no2
    elif no2 <= 100:
        return 50 + ((100-50)/(100-53)) * (no2-53)
    elif no2 <= 360:
        return 100 + ((150-100)/(360-100)) * (no2-100)
    else:
        return 200


# Benzene (C6H6) proxy AQI (approximation)
def aqi_benzene(c6h6):
    if pd.isna(c6h6):
        return np.nan
    if c6h6 <= 5:
        return (50/5) * c6h6
    elif c6h6 <= 10:
        return 50 + ((100-50)/(10-5)) * (c6h6-5)
    elif c6h6 <= 20:
        return 100 + ((150-100)/(20-10)) * (c6h6-10)
    else:
        return 200


# In[80]:


def aqi_nox(nox):
    if pd.isna(nox):
        return np.nan
    if nox <= 50:
        return (50/50) * nox
    elif nox <= 100:
        return 50 + ((100-50)/(100-50)) * (nox-50)
    else:
        return 200


def aqi_nmhc(nmhc):
    if pd.isna(nmhc):
        return np.nan
    if nmhc <= 200:
        return (50/200) * nmhc
    elif nmhc <= 400:
        return 50 + ((100-50)/(400-200)) * (nmhc-200)
    else:
        return 200


# In[81]:


def calculate_aqi(row):
    indices = []

    indices.append(aqi_co(row['CO(GT)']))
    indices.append(aqi_no2(row['NO2(GT)']))
    indices.append(aqi_nox(row['NOx(GT)']))
    indices.append(aqi_nmhc(row['NMHC(GT)']))
    indices.append(aqi_benzene(row['C6H6(GT)']))

    return np.nanmax(indices)


# In[88]:


data['AQI'] = data.apply(calculate_aqi, axis=1)
data = data.dropna(subset=['AQI'])
data['AQI'] = data['AQI'].clip(0, 500)
data.head()


# In[89]:


data['AQI'].value_counts()


# In[90]:


date_info = pd.to_datetime(data['Date'], dayfirst=True)
time_info = pd.to_datetime(data['Time'], format='%H.%M.%S')
time_info = time_info.dt.strftime('%H:%M:%S')
print(time_info)


# In[91]:


date_time = pd.concat([date_info, time_info], axis = 1)
date_time.head()


# In[92]:


date_time['DT'] = date_time['Date'].astype(str) + ' ' + date_time['Time'].astype(str)
date_time.head()


# In[103]:


df = pd.DataFrame()
df['ds'] = pd.to_datetime(date_time['DT'])
df['y'] = data['AQI']
df.head()


# In[94]:


get_ipython().system('pip install prophet')


# In[99]:


from prophet import Prophet
model = Prophet()
model.fit(df)


# In[106]:


future = model.make_future_dataframe(periods=365)
future.tail()


# In[102]:


forecast = model.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()


# In[108]:


fig1 = model.plot(forecast)


# In[110]:


fig2 = model.plot_components(forecast)


# In[ ]:




