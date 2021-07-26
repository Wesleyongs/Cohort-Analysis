import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import glob
import os


df_shop = pd.read_csv('Shopify orders online and pos 2020-July 2021.csv',
                   parse_dates=['month'])
df_shop_transformed = df_shop[['month','order_id','customer_email']]
df_shop_transformed.columns = ['month','order_id','customer_id']
df_shop_transformed=df_shop_transformed[df_shop_transformed['month']>='2020-01-01']

df_sf = pd.read_csv('SF Orders from 2019-2020.csv', encoding='latin-1', parse_dates=['Transaction Date'])
df_sf_transformed = df_sf[['Transaction Date','Transaction No','Person Account: Email']]
df_sf_transformed.columns = ['month','order_id','customer_id']
df_sf_transformed=df_sf_transformed[df_sf_transformed['month']>='2020-01-01']

df=pd.concat([df_shop_transformed,df_sf_transformed])


# In[3]:


#create order period column Y-m
df['OrderPeriod'] = df.month.apply(lambda x: x.strftime('%Y-%m'))
df.head()


# In[4]:


#create corhort group, Y-m of first purchase
df.set_index('customer_id', inplace=True)

df['CohortGroup'] = df.groupby(level=0)['month'].min().apply(lambda x: x.strftime('%Y-%m'))
df.reset_index(inplace=True)
df.head()


# In[5]:


#group them by cohortgroup and orderperiod
grouped = df.groupby(['CohortGroup', 'OrderPeriod'])

# count the unique users, orders, and total revenue per Group + Period
cohorts = grouped.agg({'customer_id': pd.Series.nunique,
                       'order_id': pd.Series.nunique})

# make the column names more meaningful
cohorts.rename(columns={'customer_id': 'TotalUsers',
                        'order_id': 'TotalOrders'}, inplace=True)
cohorts


# In[6]:


#Creates a cohortperiod column, which is the Nth period based on the user's first purchase.

def cohort_period(df):
    df['CohortPeriod'] = np.arange(len(df)) + 1
    return df

cohorts = cohorts.groupby(level=0).apply(cohort_period)
cohorts.head()


# # User retention by cohort group

# In[7]:


# reindex the DataFrame
cohorts.reset_index(inplace=True)
cohorts.set_index(['CohortGroup', 'CohortPeriod'], inplace=True)

# create a Series holding the total size of each CohortGroup. This is to be used in later calculations
cohort_group_size = cohorts['TotalUsers'].groupby(level=0).first()
cohort_group_size.head()


# In[8]:


cohorts['TotalUsers'].head()


# In[9]:


cohorts['TotalUsers'].unstack(0).head()


# In[10]:


user_retention = cohorts['TotalUsers'].unstack(0).divide(cohort_group_size, axis=1)
user_retention.head(10)


# In[11]:


user_retention.plot(figsize=(10,5))
plt.title('Cohorts: User Retention')
plt.xticks(np.arange(1, 12.1, 1))
plt.xlim(1, 12)
plt.ylabel('% of Cohort Purchasing');


# In[12]:


import seaborn as sns
sns.set(style='white')

plt.figure(figsize=(24, 16))
plt.title('Cohorts: User Retention')
sns.heatmap(user_retention.T, mask=user_retention.T.isnull(), annot=True, fmt='.0%');


# # Retention curve 

# In[13]:


# Unstack the TotalUsers
unstacked = cohorts['TotalUsers'].unstack(0)
unstacked.reset_index()

# Create a weighted data frame and reset the index
weighted = unstacked.reset_index()
# Add a Total Subs column which sums up all of the subscribers within each Cohort Period.
weighted['Total_Subs'] = weighted.drop('CohortPeriod', axis=1).sum(axis=1)
# Count non-NaN values in the row, call n
# Add up first n values of the first row, n_sum
# Divide the value in the total subs column of that row by n_sum
weighted['num_months'] = weighted['CohortPeriod'].count() - weighted.isnull().sum(axis=1)
def calc_sum(col_end):
    ans = 0 
    for i in range(1,int(col_end)):
        ans = ans + weighted.iloc[0, i]
        
    return ans
def calc_ret_pct(total_subs, num_months):
    sum_initial = calc_sum(1 + num_months)
    
    return total_subs / sum_initial
# Create a retention percentage column with use of a lambda function to apply calc ret pct for each row
weighted['Ret_Pct'] = weighted.apply(lambda row: calc_ret_pct(row['Total_Subs'], row['num_months']), axis=1)
# weighted
# Grab only the Cohort Period and Ret Pct columns
weighted_avg = weighted.filter(items=['CohortPeriod', 'Ret_Pct'])
weighted_avg['Ret_Pct'] = pd.Series(["{0:.2f}%".format(val * 100) for val in weighted_avg['Ret_Pct']], index = weighted_avg.index)
weighted_avg['CohortPeriod'] = weighted_avg['CohortPeriod'].astype(int)
weighted['Retention Percentage'] = (100 * weighted['Ret_Pct'].round(3))


weighted_avg[['CohortPeriod','Ret_Pct']]


# In[14]:


#Plotly
import sys
import plotly as py
from plotly.offline import download_plotlyjs, init_notebook_mode, iplot
from plotly.graph_objs import *
import plotly.graph_objs as go
import plotly
import plotly.express as px

layout = go.Layout(
    title = 'Monthly Retention Curve',
    yaxis=dict(
        tickformat= '%',
        categoryorder="category ascending",
        ticksuffix='%'
    )
)
fig = px.line(weighted, x="CohortPeriod", y="Retention Percentage", title='Cohorts: User Retention')
fig.show(layout = layout)


# # Cohort table and Retention Curve

# In[15]:


plt.figure(figsize=(12, 8))
plt.title('Cohorts: User Retention')
sns.heatmap(user_retention.T, mask=user_retention.T.isnull(), annot=True, fmt='.0%');


# In this version of the retention curve, I removed cohortperiod1(100%) as its an outlier and does not provide much data here.

# In[16]:


import plotly.graph_objects as go

fig = go.Figure()

fig.add_trace(go.Scatter(
    x = weighted_avg.iloc[1:]['CohortPeriod'],
    y = weighted_avg.iloc[1:]['Ret_Pct'],
))

fig.update_layout(
    yaxis=dict(
        tickformat="%",
        categoryorder="category ascending",
    ),
    title="Retention Curve (excluding cohort 1)",
    xaxis_title="Cohort Period",
    yaxis_title="Retention Percentage",
    font=dict(
        family="Courier New, monospace",
        size=18,
        color="RebeccaPurple"
    )
)

fig.show()


# # Reflections and suggestions

# Looking at the Cohort table, with the exception of period 2 of the 2018-04 and period 4 of 2018-08, the overall retention is very low among users.
# 
# Similiarly on the retention curve, there isn't a clear trend after that sharp drop in rentention rate from cohort period 1-2
# 
# We do have enough data to properly conclude the root cause for the low retention rate. One broad suggestion I would recommend is to have users sign up for a news-letter. Sign-ups can be encouraged via a website pop-up that offers users a one time promo code in exchange for signing up with the news-letter. The process should be as easy as entering their email and hitting send. We do not want to deter customers with long and tedious steps.
# 
# With the news-letter service, exisiting customers will be notified pediodically of special sales and events. This ensures a higher customer recovery rate as well as the sustainbility of Skin brand awareness With this in mind, customers are more likely to make new purchases

# In[17]:


weighted_avg

import plotly.express as px

fig = px.line(weighted_avg, x='CohortPeriod',y='Ret_Pct')
fig.update_layout(
    yaxis=dict(
        tickformat="%",
        categoryorder="category ascending",
    ),
    title="Retention Curve (excluding cohort 1)",
    xaxis_title="Cohort Period",
    yaxis_title="Retention Percentage",
    font=dict(
        family="Courier New, monospace",
        size=18,
        color="RebeccaPurple"
    )
)
fig.show()


# In[18]:


weighted_avg


# In[ ]:




