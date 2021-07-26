import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import glob
import os

st.title("Skin Inc Corhort Analysis — Wesley Ong")


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


#create order period column Y-m
df['OrderPeriod'] = df.month.apply(lambda x: x.strftime('%Y-%m'))

#create corhort group, Y-m of first purchase
df.set_index('customer_id', inplace=True)
df['CohortGroup'] = df.groupby(level=0)['month'].min().apply(lambda x: x.strftime('%Y-%m'))
df.reset_index(inplace=True)

#group them by cohortgroup and orderperiod
grouped = df.groupby(['CohortGroup', 'OrderPeriod'])

# count the unique users, orders, and total revenue per Group + Period
cohorts = grouped.agg({'customer_id': pd.Series.nunique,
                       'order_id': pd.Series.nunique})

# make the column names more meaningful
cohorts.rename(columns={'customer_id': 'TotalUsers',
                        'order_id': 'TotalOrders'}, inplace=True)

#Creates a cohortperiod column, which is the Nth period based on the user's first purchase.
def cohort_period(df):
    df['CohortPeriod'] = np.arange(len(df)) + 1
    return df

cohorts = cohorts.groupby(level=0).apply(cohort_period)

# reindex the DataFrame
cohorts.reset_index(inplace=True)
cohorts.set_index(['CohortGroup', 'CohortPeriod'], inplace=True)

# create a Series holding the total size of each CohortGroup. This is to be used in later calculations
cohort_group_size = cohorts['TotalUsers'].groupby(level=0).first()
user_retention = cohorts['TotalUsers'].unstack(0).divide(cohort_group_size, axis=1)


########################
## Corhort Line Graph ##
########################
selection = st.selectbox('Choose Corhorts',user_retention.columns)
st.line_chart(user_retention[selection])


sns.set(style='white')
fig3 = plt.figure(figsize=(24, 16))
plt.title('Cohorts: User Retention')
sns.heatmap(user_retention.T, mask=user_retention.T.isnull(), annot=True, fmt='.0%')
st.pyplot(fig3)

# # Retention curve 
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
# fig.show(layout = layout)


# # Cohort table and Retention Curve

# In[15]:


plt.figure(figsize=(12, 8))
plt.title('Cohorts: User Retention')
sns.heatmap(user_retention.T, mask=user_retention.T.isnull(), annot=True, fmt='.0%');


# In this version of the retention curve, I removed cohortperiod1(100%) as its an outlier and does not provide much data here.

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

# fig.show()
st.plotly_chart(fig)

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
# fig.show()


