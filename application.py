import streamlit as st
import pandas as pd
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
# Provlepsi

This app allows business to forecast of Food sales for various businesses to allow them to prepare for customer demand.

**Credits**
- App built in `Python` + `Streamlit` by Sang Hyun Chun, Vishwa Murugappan, and Dheeraj Vislawath
---
""")

with st.sidebar.header('Upload your CSV data'):
    uploaded_file = st.sidebar.file_uploader("Upload your input file", type=['txt'])

if st.sidebar.button('Predict with Selected .TXT File'):
    with st.spinner("Loading Data"):
        load_data = pd.read_table(uploaded_file, sep=' ', header=None)
        load_data.to_csv('user_train.csv', sep='\t', header=False, index=False)

        train = pd.read_csv(f"train.csv", low_memory=False,
                            parse_dates=['date'], index_col=['date'])
        test = pd.read_csv(f"user_train.csv", low_memory=False,
                           parse_dates=['date'], index_col=['date'])

        st.header('**Original Input Data**')
        data = expand_df(train)
        st.write(data)

        grand_avg = data.sales.mean()
        st.write(f"The grand average of sales in this dataset is {grand_avg:.4f}")

    with st.spinner("Loading Yearly Change Graphs:"):
        st.header('**Sales Changes by Year**')
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
        st.header('**Sales Changes by Month**')
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
        st.header('**Sales Changes by Week**')
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

    with st.spinner("Loading Item & Store Relationships:"):
        st.header('**Item & Stores Correlation Analysis**')
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

    with st.spinner("Loading Item & Store Table:"):
        st.header('**Item & Store Output Table**')
        store_item_table = pd.pivot_table(data, index='store', columns='item',
                                          values='sales', aggfunc=np.mean)
        st.write(store_item_table)

        dow_table = pd.pivot_table(data, index='dayofweek', values='sales', aggfunc=np.mean)
        dow_table.sales /= grand_avg

    with st.spinner("Loading Yearly Sales Growth Prediction:"):
        st.header('**Sales Growth Prediction by Year**')
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

elif st.sidebar.button('Predict with Example Data'):
    with st.spinner("Loading Data"):
        train = pd.read_csv(f"train.csv", low_memory=False,
                            parse_dates=['date'], index_col=['date'])
        test = pd.read_csv(f"new_test.csv", low_memory=False,
                           parse_dates=['date'], index_col=['date'])

        st.header('**Original Input Data**')
        data = expand_df(train)
        st.write(data)

        grand_avg = data.sales.mean()
        st.write(f"The grand average of sales in this dataset is {grand_avg:.4f}")

    with st.spinner("Loading Yearly Change Graphs:"):
        st.header('**Sales Changes by Year**')
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
        st.header('**Sales Changes by Month**')
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
        st.header('**Sales Changes by Week**')
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

    with st.spinner("Loading Item & Store Relationships:"):
        st.header('**Item & Stores Correlation Analysis**')
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

    with st.spinner("Loading Item & Store Table:"):
        st.header('**Item & Store Output Table**')
        store_item_table = pd.pivot_table(data, index='store', columns='item',
                                          values='sales', aggfunc=np.mean)
        st.write(store_item_table)

        dow_table = pd.pivot_table(data, index='dayofweek', values='sales', aggfunc=np.mean)
        dow_table.sales /= grand_avg

    with st.spinner("Loading Yearly Sales Growth Prediction:"):
        st.header('**Sales Growth Prediction by Year**')
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
else:
    st.info('Upload input data!')