import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(layout="wide")

# -----------------------------
# Helper Functions
# -----------------------------
def calculate_stats(df):
    returns = df.pct_change().dropna()
    total_return = (df.iloc[-1] / df.iloc[0]) - 1
    ann_vol = returns.std() * np.sqrt(252)
    sharpe = (returns.mean() * 252) / (returns.std() * np.sqrt(252))
    stats_df = pd.DataFrame({
        "Total Return": total_return,
        "Annualized Volatility": ann_vol,
        "Sharpe Ratio": sharpe
    })
    return stats_df


# -----------------------------------------------------------
# FIXED FACTOR ETF LIST (RIGHT-HAND SIDE DISPLAY)
# -----------------------------------------------------------
factor_etfs = {
    "Value": "MOVALUE.NS",
    "Momentum": "MOMOMENTUM.NS",
    "Quality": "HDFCQUAL.NS",
    "Growth": "HDFCGROWTH.NS",
    "Low Volatility": "LOWVOLIETF.NS"
}


# -----------------------------------------------------------
# MAIN APP LAYOUT
# -----------------------------------------------------------
left, right = st.columns([1, 1])


# -----------------------------------------------------------
# LEFT COLUMN CONTENT
# (YOUR EXISTING INPUTS + SCATTER PLOT MOVED UP)
# -----------------------------------------------------------
with left:

    st.header("Your Controls & Scatter Plot Section")

    # DATE INPUTS
    start = st.date_input("Start Date")
    end = st.date_input("End Date")

    st.write("---")

    st.subheader("Scatter Plot Inputs")
    ticker1 = st.text_input("Ticker 1")
    ticker2 = st.text_input("Ticker 2")

    if ticker1 and ticker2 and start and end:
        data = yf.download([ticker1, ticker2], start=start, end=end)["Adj Close"]
        data = data.dropna()

        if len(data.columns) == 2:
            returns = data.pct_change().dropna()

            fig_scatter = px.scatter(
                returns,
                x=returns.columns[0],
                y=returns.columns[1],
                trendline="ols",
                title="Correlation Scatter Plot"
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.warning("Could not download both tickers.")


    st.write("---")
    st.subheader("Other Inputs")
    user_number = st.number_input("Example numeric input", min_value=0, value=10)


# -----------------------------------------------------------
# RIGHT COLUMN: FACTOR ETF PERFORMANCE PANEL
# -----------------------------------------------------------
with right:

    st.header("Factor ETF Dashboard (Fixed List – No Selection)")

    # Download all factor ETF prices
    tickers = list(factor_etfs.values())
    df_factors = yf.download(tickers, period="3y")["Adj Close"].dropna()

    # Rename columns → Factor Names
    rename_map = {v: k for k, v in factor_etfs.items()}
    df_factors.rename(columns=rename_map, inplace=True)

    # Compute cumulative returns
    cumulative = (df_factors / df_factors.iloc[0]) * 100

    # ---------------------------------------------------
    # Plot cumulative performance chart of all factors
    # ---------------------------------------------------
    fig = go.Figure()
    for col in cumulative.columns:
        fig.add_trace(go.Scatter(
            x=cumulative.index,
            y=cumulative[col],
            mode="lines",
            name=col
        ))

    fig.update_layout(
        title="Cumulative Returns of Factor ETFs (Indexed to 100)",
        xaxis_title="Date",
        yaxis_title="Indexed Value (100 = Start)",
        legend_title="Factor ETFs"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.write("---")

    # ---------------------------------------------------
    # Show Stats Table
    # ---------------------------------------------------
    stats_table = calculate_stats(df_factors)
    st.subheader("Factor Performance Statistics")
    st.dataframe(stats_table.style.format("{:.2%}"), use_container_width=True)
