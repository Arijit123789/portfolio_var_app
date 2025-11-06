import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, t, jarque_bera, ttest_1samp

st.set_page_config(page_title="Portfolio Risk & VaR Analysis", layout="wide")

st.title("📊 Portfolio Risk Assessment & 1-Day 95% VaR Calculator")

# Sidebar
st.sidebar.header("Portfolio Settings")
tickers = st.sidebar.multiselect("Select Stocks", ["AAPL", "MSFT", "GOOGL", "AMZN"], default=["AAPL", "MSFT", "GOOGL", "AMZN"])
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2024-01-01"))
weights = np.array([1/len(tickers)] * len(tickers))

if st.sidebar.button("Run Analysis"):
    data = yf.download(tickers, start=start_date, end=end_date)["Adj Close"]
    log_returns = np.log(data / data.shift(1)).dropna()
    portfolio_returns = log_returns.dot(weights)

    st.subheader("1️⃣ Basic Statistics")
    st.write(pd.DataFrame({
        "Mean": log_returns.mean(),
        "Std Dev": log_returns.std(),
        "Skewness": log_returns.skew(),
        "Kurtosis": log_returns.kurt()
    }))

    st.subheader("2️⃣ Portfolio Return Distribution")
    fig, ax = plt.subplots()
    ax.hist(portfolio_returns, bins=50, alpha=0.7)
    ax.set_title("Portfolio Daily Log Returns")
    st.pyplot(fig)

    mu, sigma = portfolio_returns.mean(), portfolio_returns.std()

    # VaR Calculations
    var_norm = norm.ppf(0.05, mu, sigma)
    df, loc, scale = t.fit(portfolio_returns)
    var_t = t.ppf(0.05, df, loc, scale)
    var_hist = np.percentile(portfolio_returns, 5)

    st.subheader("3️⃣ Value at Risk (VaR) Results (1-Day, 95%)")
    st.write(pd.DataFrame({
        "Normal VaR (%)": [-var_norm * 100],
        "Student-t VaR (%)": [-var_t * 100],
        "Historical VaR (%)": [-var_hist * 100]
    }))

    # Expected Shortfall (ES)
    es_hist = portfolio_returns[portfolio_returns <= var_hist].mean()
    es_norm = mu - sigma * norm.pdf(norm.ppf(0.05)) / 0.05

    st.subheader("4️⃣ Expected Shortfall (ES)")
    st.write(pd.DataFrame({
        "Historical ES (%)": [-es_hist * 100],
        "Normal ES (%)": [-es_norm * 100]
    }))

    # Jarque-Bera Test
    jb_stat, jb_p = jarque_bera(portfolio_returns)
    st.subheader("5️⃣ Normality Test (Jarque-Bera)")
    st.write(f"JB Statistic: {jb_stat:.4f}, p-value: {jb_p:.4f}")
    st.write("➡️ " + ("Reject Normality (Not Normally Distributed)" if jb_p < 0.05 else "Fail to Reject Normality (Approx Normal)"))

    # Backtesting
    var_exceedances = (portfolio_returns < var_norm).sum()
    st.subheader("6️⃣ VaR Backtesting")
    st.write(f"Exceptions (Days returns < VaR): {var_exceedances}")
    st.write(f"Actual Exception Rate: {var_exceedances / len(portfolio_returns) * 100:.2f}%")

    # Rolling VaR
    st.subheader("7️⃣ Rolling 60-Day Historical VaR")
    rolling_var = portfolio_returns.rolling(60).quantile(0.05)
    fig, ax = plt.subplots()
    ax.plot(rolling_var, label="Rolling VaR (95%)")
    ax.axhline(var_hist, color="r", linestyle="--", label="Static VaR")
    ax.legend()
    st.pyplot(fig)

    # Cumulative Return & Drawdown
    st.subheader("8️⃣ Portfolio Growth & Drawdown")
    cum_return = (1 + portfolio_returns).cumprod()
    rolling_max = cum_return.cummax()
    drawdown = (cum_return - rolling_max) / rolling_max

    fig, ax = plt.subplots(2, 1, figsize=(8, 6))
    ax[0].plot(cum_return, label="Cumulative Return")
    ax[0].legend()
    ax[1].plot(drawdown, color="r", label="Drawdown")
    ax[1].legend()
    st.pyplot(fig)
