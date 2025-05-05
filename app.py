# -*- coding: utf-8 -*-
"""
Created on Wed April  23 18:59:43 2025

@author: simmo
"""

import streamlit as st
import yfinance as yf
import pandas as df
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pmdarima import auto_arima

st.set_page_config(page_title="Stock Forecast", layout="centered")

def get_data(ticker):
    end = datetime.today()
    start = end - timedelta(days=365)
    data = yf.download(ticker, start=start, end=end)
    return data

def plot_close_prices(data, title):
    st.subheader(title)
    st.line_chart(data['Close'])

def forecast_arima(data, days=7):
    close_prices = data['Close']
    model = auto_arima(close_prices, seasonal=False, suppress_warnings=True)
    forecast = model.predict(n_periods=days)

    future_dates = [data.index[-1] + timedelta(days=i) for i in range(1, days+1)]
    forecast_df = df.DataFrame({'Forecast': forecast}, index=future_dates)

    # Plot both historical and forecast
    plt.figure(figsize=(10, 4))
    plt.plot(close_prices.index, close_prices.values, label='Historical')
    plt.plot(forecast_df.index, forecast_df['Forecast'], label='Forecast', linestyle='--')
    plt.title("Next 7-Day Forecast")
    plt.xlabel("Date")
    plt.ylabel("Price ($)")
    plt.legend()
    st.pyplot(plt)

def main():
    st.title("Stock Trend & Forecast Viewer")
    ticker = st.text_input("Enter stock ticker (e.g., AAPL, MSFT)", "AAPL").upper()

    try:
        data = get_data(ticker)
        if data.empty:
            st.error("No data found for that ticker.")
            return

        # Show charts for week/month/year
        plot_close_prices(data[-7:], "Past Week")
        plot_close_prices(data[-30:], "Past Month")
        plot_close_prices(data, "Past Year")

        st.subheader("Forecast (ARIMA)")
        forecast_arima(data)

    except Exception as e:
        st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
