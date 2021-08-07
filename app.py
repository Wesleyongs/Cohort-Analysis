import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import glob
import os
import sys
import plotly as py
from plotly.offline import download_plotlyjs, init_notebook_mode, iplot
from plotly.graph_objs import *
import plotly.graph_objs as go
import plotly
import plotly.express as px
import xlsxwriter
import base64
from io import BytesIO
import os

###########
# heading #
###########
st.set_page_config(layout="wide")
st.write("""
# Skin Inc Corhort Analysis
This app tracks **cohort analysis** from 2020 - Jul 2021!
Created by [Wesley Ong](https://Wesleyongs).
""")

################
# Upload Files #
################

# SG
uploaded_file = st.file_uploader('Upload SG file', type="csv")
if uploaded_file is not None:
    df_shop = pd.read_csv(uploaded_file,
                   parse_dates=['month'])
else:    
    df_shop = pd.read_csv('Shopify orders online and pos 2020-July 2021.csv',
                    parse_dates=['month'])
df_shop_transformed = df_shop[['month','order_id','customer_email']]
df_shop_transformed.columns = ['month','order_id','customer_id']
df_sg_transformed = df_shop_transformed[df_shop_transformed['month']>='2020-01-01']

# ROW
uploaded_file = st.file_uploader('Upload ROW file', type="csv")
if uploaded_file is not None:
    df_shop = pd.read_csv(uploaded_file,
                   parse_dates=['month'])
else:
    df_shop = pd.read_csv('Shopify orders row 2020-July 2021.csv',
                   parse_dates=['month'])
df_shop_transformed = df_shop[['month','order_id','customer_email']]
df_shop_transformed.columns = ['month','order_id','customer_id']
df_row_transformed=df_shop_transformed[df_shop_transformed['month']>='2020-01-01']

# US
uploaded_file = st.file_uploader('Upload US file', type="csv")
if uploaded_file is not None:
    df_shop = pd.read_csv(uploaded_file,
                   parse_dates=['month'])
else:
    df_shop = pd.read_csv('Shopify orders online and pos 2020-July 2021.csv',
                   parse_dates=['month'])
df_shop_transformed = df_shop[['month','order_id','customer_email']]
df_shop_transformed.columns = ['month','order_id','customer_id']
df_us_transformed=df_shop_transformed[df_shop_transformed['month']>='2020-01-01']

# SF
df_sf = pd.read_csv('SF Orders from 2019-2020.csv', encoding='latin-1', parse_dates=['Transaction Date'])
df_sf_transformed = df_sf[['Transaction Date','Transaction No','Person Account: Email']]
df_sf_transformed.columns = ['month','order_id','customer_id']
df_sf_transformed=df_sf_transformed[df_sf_transformed['month']>='2020-01-01']

#Concat
df=pd.concat([df_sg_transformed,df_sf_transformed])

# Download table 
def get_table_download_link(df):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}">Download csv file</a>'

def show_cohort_analysis(df, region):    
    
    st.title("Data for " + region,"linkto_SG")
    col1,col2 = st.beta_columns((1,1))

    ########################
    ##   Data Wraggling   ##
    ########################

    df['OrderPeriod'] = df.month.apply(lambda x: x.strftime('%Y-%m'))
    df.set_index('customer_id', inplace=True)
    df['CohortGroup'] = df.groupby(level=0)['month'].min().apply(lambda x: x.strftime('%Y-%m'))
    df.reset_index(inplace=True)
    grouped = df.groupby(['CohortGroup', 'OrderPeriod'])
    cohorts = grouped.agg({'customer_id': pd.Series.nunique,
                        'order_id': pd.Series.nunique})
    cohorts.rename(columns={'customer_id': 'TotalUsers',
                            'order_id': 'TotalOrders'}, inplace=True)
    def cohort_period(df):
        df['CohortPeriod'] = np.arange(len(df)) + 1
        return df
    cohorts = cohorts.groupby(level=0).apply(cohort_period)
    cohorts.reset_index(inplace=True)
    cohorts.set_index(['CohortGroup', 'CohortPeriod'], inplace=True)
    cohort_group_size = cohorts['TotalUsers'].groupby(level=0).first()
    user_retention = cohorts['TotalUsers'].unstack(0).divide(cohort_group_size, axis=1)

    ########################
    ##      Heat Map      ##
    ########################

    # st.write("""
    # ## Heat Map overview
    # """)
    # st.write("""
    # ## Heat Map Cu
    # """)
    
    sns.set(style='white')
    fig3 = plt.figure(figsize=(18, 12))
    sns.heatmap(user_retention.T, mask=user_retention.T.isnull(), annot=True, fmt='.0%')
    col1.pyplot(fig3,use_column_width=True)
    
    heat_map_values = cohorts['TotalUsers'].unstack(0)
    fig4 = plt.figure(figsize=(18, 12))
    sns.heatmap(heat_map_values.T, mask=heat_map_values.T.isnull(), annot=True, fmt='.20g')
    col2.pyplot(fig4,use_column_width=True)
    
    # st.dataframe(heat_map_values.T)

    ########################
    ## Corhort Line Graph ##
    ########################

    col1.write("""
    ## View line graph for individual cohorts
    """)
    selection = col1.selectbox('Choose '+region+' Corhorts',user_retention.columns)
    col1.line_chart(user_retention[selection])

    ########################
    ##   Retention Data   ##
    ########################

    unstacked = cohorts['TotalUsers'].unstack(0)
    unstacked.reset_index()
    weighted = unstacked.reset_index()
    weighted['Total_Subs'] = weighted.drop('CohortPeriod', axis=1).sum(axis=1)
    weighted['num_months'] = weighted['CohortPeriod'].count() - weighted.isnull().sum(axis=1)
    def calc_sum(col_end):
        ans = 0 
        for i in range(1,int(col_end)):
            ans = ans + weighted.iloc[0, i]
        return ans
    def calc_ret_pct(total_subs, num_months):
        sum_initial = calc_sum(1 + num_months)
        
        return total_subs / sum_initial
    weighted['Ret_Pct'] = weighted.apply(lambda row: calc_ret_pct(row['Total_Subs'], row['num_months']), axis=1)
    weighted_avg = weighted.filter(items=['CohortPeriod', 'Ret_Pct'])
    weighted_avg['Ret_Pct'] = pd.Series(["{0:.2f}%".format(val * 100) for val in weighted_avg['Ret_Pct']], index = weighted_avg.index)
    weighted_avg['CohortPeriod'] = weighted_avg['CohortPeriod'].astype(int)
    weighted['Retention Percentage'] = (100 * weighted['Ret_Pct'].round(3))


    ########################
    ##  Cumulative Curve  ##
    ########################

    col2.write("""
    ## Cumulative Retention Curve (excluding period 1)
    """)
   
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x = weighted_avg.iloc[1:]['CohortPeriod'],
        y = weighted_avg.iloc[1:]['Ret_Pct'].str.rstrip('%').astype('float')/100,
    ))
    fig.update_layout(
        yaxis=dict(
            tickformat="%",
            categoryorder="category ascending",
        ),
        xaxis_title="Cohort Period",
        yaxis_title="Retention Percentage",
        font=dict(
            size=18,
        )
    )
    col2.plotly_chart(fig)
    
     #############
    # DL LINK ###
    #############
    
    download=st.button('Download '+region+' cumulative Excel File')
    if download:
        'Download Started! Please wait a link will appear below for your to download the file'
        csv = weighted_avg.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()  # some strings
        linko= f'<a href="data:file/csv;base64,{b64}" download="myfilename.csv">Download csv file</a>'
        st.markdown(linko, unsafe_allow_html=True)
    
    ########################
    ##  Export            ##
    ########################
    
    # Create a Pandas Excel writer using XlsxWriter as the engine.
    writer = pd.ExcelWriter(region+'2.0.xlsx', engine='xlsxwriter')

    # Write each dataframe to a different worksheet.
    weighted_avg.to_excel(writer,sheet_name='Cumulative')
    user_retention.to_excel(writer,sheet_name='Cohort')
    heat_map_values.T.to_excel(writer,sheet_name="Cohort Values")

    # Close the Pandas Excel writer and output the Excel file.
    writer.save()

show_cohort_analysis(df,'SG')
show_cohort_analysis(df_us_transformed,'US')
show_cohort_analysis(df_shop_transformed,'ROW')

# # Sidebar Column
# st.sidebar.title('Sidebar Widgets')
# #radio button 
# rating = st.sidebar.radio('Are You Happy with the Example',('Yes','No','Not Sure'))
# if rating == 'Yes':
#     st.sidebar.success('Thank You for Selecting Yes')
# elif rating =='No':
#     st.sidebar.info('Thank You for Selecting No')
# elif rating =='Not Sure':
#     st.sidebar.info('Thank You for Selecting Not sure')
# #selectbox
# rating = st.sidebar.selectbox("How much would you rate this App? ",
#                      ['5 Stars', '4 Stars', '3 Stars','2 Stars','1 Star'])
# st.sidebar.success(rating)
# st.sidebar.write('Find Square of a Number')
# #slider
# get_number = st.sidebar.slider("Select a Number", 1, 10)
# st.sidebar.write('Square of Number',get_number, 'is', get_number*get_number)