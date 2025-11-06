import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, t, jarque_bera, ttest_1samp
import io

st.set_page_config(page_title="Portfolio VaR & Risk Analysis", layout="wide")

st.title("📊 Portfolio Risk Assessment & 1-Day 95% VaR Calculator")

# Sidebar configuration
st.sidebar.header("Portfolio Settings")

tickers = st.sidebar.multiselect(
    "Select Stocks", ["AAPL", "MSFT", "GOOGL", "AMZN"],
    default=["AAPL", "MSFT", "GOOGL", "AMZN"]
)
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2024-01-01"))
run_button = st.sidebar.button("Run Analysis")

# Helper function: download data safely
@st.cache_data
def get_data(tickers, start, end):
    try:
        data = yf.download(tickers, start=start, end=end, progress=False)["Adj Close"]
        return data
    except Exception as e:
        st.error(f"❌ Error downloading data: {e}")
        return pd.DataFrame()

# Main logic
if run_button:
    if not tickers:
        st.warning("⚠️ Please select at least one stock.")
        st.stop()

    data = get_data(tickers, start_date, end_date)

    if data.empty:
        st.error("❌ No data retrieved. Check internet connection, ticker symbols, or date range.")
        st.stop()

    # Calculate log returns
    log_returns = np.log(data / data.shift(1)).dropna()

    # Handle missing tickers
    available_tickers = list(log_returns.columns)
    if len(available_tickers) != len(tickers):
        missing = [t for t in tickers if t not in available_tickers]
        st.warning(f"⚠️ Missing data for: {', '.join(missing)}")

    # Equal weights based on available tickers
    weights = np.array([1 / len(available_tickers)] * len(available_tickers))

    # Portfolio log return
    try:
        portfolio_returns = log_returns.dot(weights)
    except ValueError:
        st.error("❌ Portfolio calculation failed due to shape mismatch. Try refreshing.")
        st.stop()

    # ---- Display Basic Statistics ----
    st.subheader("1️⃣ Daily Return Statistics")
    summary_stats = pd.DataFrame({
        "Mean": log_returns.mean(),
        "Std Dev": log_returns.std(),
        "Skewness": log_returns.skew(),
        "Kurtosis": log_returns.kurt()
    })
    st.dataframe(summary_stats.style.format("{:.4f}"))

    # ---- Portfolio Return Distribution ----
    st.subheader("2️⃣ Portfolio Return Distribution")
    fig, ax = plt.subplots()
    ax.hist(portfolio_returns, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax.set_title("Portfolio Daily Log Returns")
    st.pyplot(fig)

    mu, sigma = portfolio_returns.mean(), portfolio_returns.std()

    # ---- Value at Risk (VaR) ----
    var_norm = norm.ppf(0.05, mu, sigma)
    df, loc, scale = t.fit(portfolio_returns)
    var_t = t.ppf(0.05, df, loc, scale)
    var_hist = np.percentile(portfolio_returns, 5)

    st.subheader("3️⃣ Value at Risk (VaR) (1-Day, 95%)")
    st.table(pd.DataFrame({
        "Normal VaR (%)": [-var_norm * 100],
        "Student-t VaR (%)": [-var_t * 100],
        "Historical VaR (%)": [-var_hist * 100]
    }).round(4))

    # ---- Expected Shortfall (ES) ----
    es_hist = portfolio_returns[portfolio_returns <= var_hist].mean()
    es_norm = mu - sigma * norm.pdf(norm.ppf(0.05)) / 0.05

    st.subheader("4️⃣ Expected Shortfall (ES)")
    st.table(pd.DataFrame({
        "Historical ES (%)": [-es_hist * 100],
        "Normal ES (%)": [-es_norm * 100]
    }).round(4))

    # ---- Normality Test ----
    jb_stat, jb_p = jarque_bera(portfolio_returns)
    st.subheader("5️⃣ Normality Test (Jarque-Bera)")
    st.write(f"JB Statistic: **{jb_stat:.4f}**, p-value: **{jb_p:.6f}**")
    if jb_p < 0.05:
        st.warning("⚠️ Reject Normality — returns are not normally distributed (fat tails).")
    else:
        st.success("✅ Fail to reject normality — returns are approximately normal.")

    # ---- Backtesting ----
    var_exceedances = (portfolio_returns < var_norm).sum()
    st.subheader("6️⃣ VaR Backtesting")
    st.write(f"Exceptions (days where loss exceeded VaR): **{var_exceedances}**")
    st.write(f"Actual Exception Rate: **{var_exceedances / len(portfolio_returns) * 100:.2f}%**")

    # ---- Rolling 60-Day VaR ----
    st.subheader("7️⃣ Rolling 60-Day Historical VaR")
    rolling_var = portfolio_returns.rolling(60).quantile(0.05)
    fig, ax = plt.subplots()
    ax.plot(rolling_var, label="Rolling VaR (95%)")
    ax.axhline(var_hist, color="r", linestyle="--", label="Static VaR")
    ax.legend()
    st.pyplot(fig)

    # ---- Portfolio Growth & Drawdown ----
    st.subheader("8️⃣ Portfolio Growth & Drawdown")
    cum_return = (1 + portfolio_returns).cumprod()
    rolling_max = cum_return.cummax()
    drawdown = (cum_return - rolling_max) / rolling_max

    fig, ax = plt.subplots(2, 1, figsize=(8, 6))
    ax[0].plot(cum_return, label="Cumulative Return", color='green')
    ax[0].legend()
    ax[1].plot(drawdown, color='red', label="Drawdown")
    ax[1].legend()
    st.pyplot(fig)

    st.success("✅ Analysis Completed Successfully!")
    st.caption("Built with Python • Streamlit • yFinance • SciPy • NumPy • Matplotlib")
