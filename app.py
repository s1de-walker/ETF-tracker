#%%writefile app.py

import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Streamlit UI
st.title("ETF Tracker")
st.divider()
# 📃 Date Selection (Side-by-side)
st.markdown("### Select Time Period for Analysis")

col1, col2 = st.columns(2)

with col1:
    start_date = st.date_input("Start Date", datetime.today() - timedelta(days=365))

with col2:
    end_date = st.date_input("End Date", datetime.today())

# **Validation Checks**
error_flag = False  

if end_date < start_date:
    st.error("🚨 End Date cannot be earlier than Start Date. Please select a valid range.")
    error_flag = True

if start_date > datetime.today().date() or end_date > datetime.today().date():
    st.error("🚨 Dates cannot be in the future. Please select a valid range.")
    error_flag = True

# **Run only if there are no errors**
if not error_flag:
    start_date = start_date.strftime('%Y-%m-%d')
    end_date = end_date.strftime('%Y-%m-%d')

    st.write("")
    st.write("")

    # 📊 ETF Selection
    st.markdown("### Select ETFs")

    tickers = ["NIFTYBEES.NS", "NEXT50IETF.NS", "BANKBEES.NS", "MASPTOP50.NS", "MON100.NS", "MAFANG.NS", "GOLDBEES.NS", "SILVERETF.NS",
        "HDFCQUAL.NS", "LOWVOLIETF.NS", "MOMOMENTUM.NS", "HDFCGROWTH.NS", "MOVALUE.NS", "SMALLCAP.NS", "MIDCAPETF.NS", "TOP100CASE.NS", "ALPHA.NS", 
        "MNC.NS", "CONS.NS", "ITIETF.NS", "FMCGIETF.NS", "OILIETF.NS", "HEALTHIETF.NS", 
        "DIVOPPBEES.NS", "LTGILTBEES.NS", "GSEC5IETF.NS", "CPSEETF.NS", "MAKEINDIA.NS", "LIQUIDCASE.NS", "LIQUIDBEES.NS",
        "VYM", "BBUS", "IYF", "VTI", "JGRO", "MTUM", "QUAL", "JCTR", "SPY", "VOO", "USMV"
    ]

    default_tickers = ["MASPTOP50.NS", "NIFTYBEES.NS", "GOLDBEES.NS", "HDFCQUAL.NS", "LOWVOLIETF.NS", "MOMOMENTUM.NS"]
    selected_tickers = st.multiselect("Choose ETFs:", tickers, default=default_tickers)

    st.divider()

    data = yf.download(selected_tickers, start=start_date, end=end_date)['Close']

    # Calculate compounded returns
    returns = data.pct_change().add(1).cumprod() - 1

    # ✅ **Interactive Plotly Chart**
    if selected_tickers:
        st.markdown("### Cumulative Performance")
        
        fig = go.Figure()

        for ticker in selected_tickers:
            fig.add_trace(go.Scatter(
                x=returns.index, 
                y=returns[ticker], 
                mode="lines",
                name=ticker,  
                line=dict(width=2)  
            ))
        
        fig.update_layout(
            template="plotly_dark",  # ✅ Dark Mode
            hovermode="x unified",  # ✅ Shows all values when hovering
            legend=dict(title="ETFs", bgcolor="rgba(0,0,0,0.5)"),  # Semi-transparent legend
            margin=dict(l=40, r=40, t=40, b=40),
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)

    # Calculate summary statistics
    daily_returns = data.pct_change()
    cov_matrix = daily_returns.cov()

    if "NIFTYBEES.NS" in selected_tickers:
        cov_with_n50 = cov_matrix.loc[selected_tickers, "NIFTYBEES.NS"]
        n50_variance = daily_returns["NIFTYBEES.NS"].var()
        beta_vs_n50 = cov_with_n50 / n50_variance
    else:
        beta_vs_n50 = pd.Series(index=selected_tickers, dtype="float64")

    summary_stats = pd.DataFrame({
        "Total Return (%)": returns.iloc[-1] * 100,
        "Annualized Volatility (%)": daily_returns.std() * (252 ** 0.5) * 100,
        "Sharpe Ratio": (returns.iloc[-1] / (daily_returns.std() * (252 ** 0.5))).round(2),
        "VaR 95 (%)": daily_returns.quantile(0.05) * 100,
        "Beta (vs N50)": beta_vs_n50
    }).T

    filtered_summary_stats = summary_stats[selected_tickers]

    if not filtered_summary_stats.empty:
        st.subheader("ETF Summary")

        best_performer = filtered_summary_stats.loc["Total Return (%)"].idxmax()
        best_performer_value = filtered_summary_stats.loc["Total Return (%)", best_performer]

        most_volatile = filtered_summary_stats.loc["Annualized Volatility (%)"].idxmax()
        most_volatile_value = filtered_summary_stats.loc["Annualized Volatility (%)", most_volatile]

        best_sharpe = filtered_summary_stats.loc["Sharpe Ratio"].idxmax()
        best_sharpe_value = filtered_summary_stats.loc["Sharpe Ratio", best_sharpe]

        col1, col2, col3 = st.columns(3)

        col1.metric("🚀 Highest Return ETF", best_performer, f"{best_performer_value:.0f}%")
        col2.metric("⚡ Most Volatile ETF", most_volatile, f"{most_volatile_value:.0f}%")
        col3.metric("🎯 Best Sharpe Ratio", best_sharpe, f"{best_sharpe_value:.1f}")
        
    st.write("")
    st.write("")
    
    st.dataframe(filtered_summary_stats.T.style.format("{:.1f}"))

    st.write("")
    
    # Display correlation matrix
    st.subheader("ETF Correlation")

    colors = ["#1b3368", "white", "#7c2f57"]
    cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)
   
    correlation_matrix = returns[selected_tickers].corr().round(2)

    st.dataframe(correlation_matrix.style.format("{:.2f}").background_gradient(cmap=cmap, axis=None, vmin=-1, vmax=1))
