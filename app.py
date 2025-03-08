
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
st.write("")
# 🗓️ Date Selection (Side-by-side)
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

    # 📊 Factor Selection
    st.markdown("### Select ETFs")

    factors = {
        "Quality": "HDFCQUAL.NS",
        "LowVolt": "LOWVOLIETF.NS",
        "Momntm": "MOMOMENTUM.NS",
        "Size": "SMALLCAP.NS",
        "NIFTY50": "NIFTYBEES.NS",
        "Bank": "BANKBEES.NS",
        "SP50": "MASPTOP50.NS",
        "N100": "MON100.NS",
        "MAFANG": "MAFANG.NS",
        "GOLD": "GOLDBEES.NS",
        "VYM": "VYM",
        "BBUS": "BBUS",
        "IYF":"IYF",
        "VTI":"VTI",
        "JGRO":"JGRO",
        "MTUM":"MTUM",
        "QUAL":"QUAL",
        "JCTR":"JCTR",
        "SPY":"SPY",
        "VOO":"VOO",
        "USMV":"USMV"
    }

    default_factors = ["SP50", "NIFTY50", "GOLD", "Quality","LowVolt", "Momntm","VOO"]
    selected_factors = st.multiselect("Choose factors:", factors.keys(), default=default_factors)

    st.divider()

    tickers = [factors[f] for f in selected_factors]
    data = yf.download(tickers, start=start_date, end=end_date)['Close']

    # Calculate compounded returns
    returns = data.pct_change().add(1).cumprod() - 1

    # ✅ **Interactive Plotly Chart**
    if selected_factors:
        st.markdown("### Cumulative Performance")
        
        fig = go.Figure()

        for factor in selected_factors:
            ticker = factors[factor]
            fig.add_trace(go.Scatter(
                x=returns.index, 
                y=returns[ticker], 
                mode="lines",
                name=factor,  
                line=dict(width=2)  
            ))
        
        fig.update_layout(
            template="plotly_dark",  # ✅ Dark Mode
            hovermode="x unified",  # ✅ Shows all values when hovering
            legend=dict(title="Factors", bgcolor="rgba(0,0,0,0.5)"),  # Semi-transparent legend
            margin=dict(l=40, r=40, t=40, b=40),
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)

    # Calculate summary statistics
    daily_returns = data.pct_change()
    cov_matrix = daily_returns.cov()

    if "NIFTY50" in selected_factors:
        cov_with_n50 = cov_matrix.loc[tickers, "NIFTYBEES.NS"]
        n50_variance = daily_returns["NIFTYBEES.NS"].var()
        beta_vs_n50 = cov_with_n50 / n50_variance
    else:
        beta_vs_n50 = pd.Series(index=tickers, dtype="float64")

    summary_stats = pd.DataFrame({
        "Total Return (%)": returns.iloc[-1] * 100,
        "Annualized Volatility (%)": daily_returns.std() * (252 ** 0.5) * 100,
        "Sharpe Ratio": (returns.iloc[-1] / (daily_returns.std() * (252 ** 0.5))).round(2),
        "VaR 95 (%)": daily_returns.quantile(0.05) * 100,
        "Beta (vs N50)": beta_vs_n50
    }).T

    filtered_summary_stats = summary_stats[tickers]

    # Display Metrics in Columns
    reverse_factors = {v: k for k, v in factors.items()}

    if not filtered_summary_stats.empty:
        st.subheader("ETF Summary")

        best_performer = filtered_summary_stats.loc["Total Return (%)"].idxmax()
        best_performer_value = filtered_summary_stats.loc["Total Return (%)", best_performer]

        most_volatile = filtered_summary_stats.loc["Annualized Volatility (%)"].idxmax()
        most_volatile_value = filtered_summary_stats.loc["Annualized Volatility (%)", most_volatile]

        best_sharpe = filtered_summary_stats.loc["Sharpe Ratio"].idxmax()
        best_sharpe_value = filtered_summary_stats.loc["Sharpe Ratio", best_sharpe]

        col1, col2, col3 = st.columns(3)

        col1.metric("🚀 Highest Return ETF", f"{reverse_factors.get(best_performer, best_performer)}", f"{best_performer_value:.0f}%")
        col2.metric("⚡ Most Volatile ETF", f"{reverse_factors.get(most_volatile, most_volatile)}", f"{most_volatile_value:.0f}%")
        col3.metric("🎯 Best Sharpe Ratio", f"{reverse_factors.get(best_sharpe, best_sharpe)}", f"{best_sharpe_value:.1f}")

    st.write("")

    # Rename index in summary statistics from tickers to ETF names
    etf_names = {v: k for k, v in factors.items()}
    filtered_summary_stats = filtered_summary_stats.rename(columns=etf_names)

    # Display summary DataFrame
    st.dataframe(filtered_summary_stats.T.style.format("{:.1f}"))

    st.write("")
    st.write("")
    st.write("")
    
    # Display correlation matrix
    # Heatmap for Factor Correlations
    st.subheader("ETF Correlation")

    # Create a custom diverging color palette
    # Create a custom diverging color map
    colors = ["#1b3368", "white", "#7c2f57"]
    cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)
   
    correlation_matrix = returns[[factors[f] for f in selected_factors]].corr().round(2)
    correlation_matrix = correlation_matrix.rename(index=etf_names, columns=etf_names)

    st.dataframe(correlation_matrix.style.format("{:.2f}").background_gradient(cmap=cmap, axis=None, vmin=-1, vmax=1))

