import streamlit as st
import pandas as pd
from PIL import Image
import subprocess
import os
import base64
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
pd.set_option('display.max_rows', 12)

def expand_df(df):
    data = df.copy()
    data['day'] = data.index.day
    data['month'] = data.index.month
    data['year'] = data.index.year
    data['dayofweek'] = data.index.dayofweek
    return data

# Page title
st.markdown("""
# Bioactivity Prediction App (Acetylcholinesterase)

This app allows you to predict the bioactivity towards inhibting the `Acetylcholinesterase` enzyme. `Acetylcholinesterase` is a drug target for Alzheimer's disease.

**Credits**
- App built in `Python` + `Streamlit` by [Chanin Nantasenamat](https://medium.com/@chanin.nantasenamat) (aka [Data Professor](http://youtube.com/dataprofessor))
- Descriptor calculated using [PaDEL-Descriptor](http://www.yapcwsoft.com/dd/padeldescriptor/) [[Read the Paper]](https://doi.org/10.1002/jcc.21707).
---
""")

# Sidebar
with st.sidebar.header('Upload your CSV data'):
    uploaded_file = st.sidebar.file_uploader("Upload your input file", type=['txt'])
    st.sidebar.markdown("""
[Example input file](https://raw.githubusercontent.com/dataprofessor/bioactivity-prediction-app/main/example_acetylcholinesterase.txt)
""")

if st.sidebar.button('Predict'):
    with st.spinner("Loading Data"):
        train = pd.read_csv(f"train.csv", low_memory=False,
                            parse_dates=['date'], index_col=['date'])
        test = pd.read_csv(f"test.csv", low_memory=False,
                           parse_dates=['date'], index_col=['date'])

        st.header('**Original input data**')
        data = expand_df(train)
        st.write(data)

        grand_avg = data.sales.mean()
        st.write(f"The grand average of sales in this dataset is {grand_avg:.4f}")

    with st.spinner("Loading Yearly Change Graphs:"):
        agg_year_item = pd.pivot_table(data, index='year', columns='item',
                                       values='sales', aggfunc=np.mean).values
        agg_year_store = pd.pivot_table(data, index='year', columns='store',
                                        values='sales', aggfunc=np.mean).values

        plt.figure(figsize=(12, 5))
        plt.subplot(121)
        plt.plot(agg_year_item / agg_year_item.mean(0)[np.newaxis])
        plt.title("Items")
        plt.xlabel("Year")
        plt.ylabel("Relative Sales")
        plt.subplot(122)
        plt.plot(agg_year_store / agg_year_store.mean(0)[np.newaxis])
        plt.title("Stores")
        plt.xlabel("Year")
        plt.ylabel("Relative Sales")
        st.pyplot(plt)

    with st.spinner("Loading Monthly Change Graphs:"):
        agg_month_item = pd.pivot_table(data, index='month', columns='item',
                                        values='sales', aggfunc=np.mean).values
        agg_month_store = pd.pivot_table(data, index='month', columns='store',
                                         values='sales', aggfunc=np.mean).values

        plt.figure(figsize=(12, 5))
        plt.subplot(121)
        plt.plot(agg_month_item / agg_month_item.mean(0)[np.newaxis])
        plt.title("Items")
        plt.xlabel("Month")
        plt.ylabel("Relative Sales")
        plt.subplot(122)
        plt.plot(agg_month_store / agg_month_store.mean(0)[np.newaxis])
        plt.title("Stores")
        plt.xlabel("Month")
        plt.ylabel("Relative Sales")
        st.pyplot(plt)

    with st.spinner("Loading Weekly Change (Day Basis) Graphs:"):
        agg_dow_item = pd.pivot_table(data, index='dayofweek', columns='item',
                                      values='sales', aggfunc=np.mean).values
        agg_dow_store = pd.pivot_table(data, index='dayofweek', columns='store',
                                       values='sales', aggfunc=np.mean).values

        plt.figure(figsize=(12, 5))
        plt.subplot(121)
        plt.plot(agg_dow_item / agg_dow_item.mean(0)[np.newaxis])
        plt.title("Items")
        plt.xlabel("Day of Week")
        plt.ylabel("Relative Sales")
        plt.subplot(122)
        plt.plot(agg_dow_store / agg_dow_store.mean(0)[np.newaxis])
        plt.title("Stores")
        plt.xlabel("Day of Week")
        plt.ylabel("Relative Sales")
        st.pyplot(plt)

    with st.spinner("Loading Item & Stores Relationship Graphs:"):
        agg_store_item = pd.pivot_table(data, index='store', columns='item',
                                        values='sales', aggfunc=np.mean).values

        plt.figure(figsize=(14, 5))
        plt.subplot(121)
        plt.plot(agg_store_item / agg_store_item.mean(0)[np.newaxis])
        plt.title("Items")
        plt.xlabel("Store")
        plt.ylabel("Relative Sales")
        plt.subplot(122)
        plt.plot(agg_store_item.T / agg_store_item.T.mean(0)[np.newaxis])
        plt.title("Stores")
        plt.xlabel("Item")
        plt.ylabel("Relative Sales")
        st.pyplot(plt)

    with st.spinner("Loading Item-Store Table:"):
        store_item_table = pd.pivot_table(data, index='store', columns='item',
                                          values='sales', aggfunc=np.mean)
        st.write(store_item_table)

        dow_table = pd.pivot_table(data, index='dayofweek', values='sales', aggfunc=np.mean)
        dow_table.sales /= grand_avg

        # Yearly growth pattern
        year_table = pd.pivot_table(data, index='year', values='sales', aggfunc=np.mean)
        year_table /= grand_avg

        years = np.arange(2013, 2019)
        annual_sales_avg = year_table.values.squeeze()

        p2 = np.poly1d(np.polyfit(years[:-1], annual_sales_avg, 2))

        plt.figure(figsize=(8, 6))
        plt.plot(years[:-1], annual_sales_avg, 'ko')
        plt.plot(years, p2(years), 'C1-')
        plt.xlim(2012.5, 2018.5)
        plt.title("Relative Sales by Year")
        plt.ylabel("Relative Sales")
        plt.xlabel("Year")
        st.pyplot(plt)

        st.write(f"2018 Relative Sales by Degree-2 (Quadratic) Fit = {p2(2018):.4f}")

else:
    st.info('Upload input data in the sidebar to start!')