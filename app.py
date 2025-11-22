# app.py - Final version
# - Left column: persistent (user selects ETFs & dates)
# - Computes metrics: Annualized Return, Annualized Volatility, Rolling Sharpe (60d), Beta vs NIFTYBEES (90d)
# - Correlation heatmap of metrics with custom colors
# - Scatter plot (metrics) at bottom of left column
# - Right column: fixed factors panel (5 ETFs)

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

st.set_page_config(layout="wide", page_title="ETF Tracker - Final")

# ----------------------------
# PARAMETERS (defaults you asked for)
# ----------------------------
SHARPE_WINDOW = 60  # days (rolling)
BETA_WINDOW = 90    # days (rolling)

# Factor mapping for RHS
FACTOR_MAP = {
    "MOVALUE.NS": "Value",
    "MOMOMENTUM.NS": "Momentum",
    "HDFCQUAL.NS": "Quality",
    "HDFCGROWTH.NS": "Growth",
    "LOWVOLIETF.NS": "Low Volatility"
}
FACTOR_TICKERS = list(FACTOR_MAP.keys())

# ----------------------------
# Helpers
# ----------------------------
def safe_extract_prices(df):
    """Handle different shapes returned by yf.download and return price DataFrame with tickers as columns."""
    if df is None or df.empty:
        return pd.DataFrame()
    cols = df.columns
    if isinstance(cols, pd.MultiIndex):
        # prefer Adj Close, then Close
        for prefer in ["Adj Close", "Close"]:
            if prefer in cols.levels[0]:
                return df[prefer].copy()
        # fallback: try to get level with tickers if possible
        try:
            # try to select the last level as prices
            return df.xs("Close", level=0, axis=1).copy()
        except Exception:
            return pd.DataFrame()
    else:
        # single-level columns (could already be tickers)
        # if 'Adj Close' or 'Close' present as column names
        if "Adj Close" in df.columns:
            return df[["Adj Close"]].rename(columns={"Adj Close": df.columns.name or "PRICE"})
        if "Close" in df.columns:
            return df[["Close"]].rename(columns={"Close": df.columns.name or "PRICE"})
        return df.copy()

def compute_annualized_return(price_df):
    """Annualized return (geometric) for each column."""
    # (end / start)^(252/period) - 1
    periods = (price_df.index[-1] - price_df.index[0]).days
    if periods <= 0:
        return pd.Series(index=price_df.columns, dtype=float)
    total_ret = (price_df.iloc[-1] / price_df.iloc[0]) - 1
    annual_factor = 252 / (price_df.shape[0] if price_df.shape[0] > 0 else 252)
    # approximate annualization using sqrt-scaling on returns via daily mean compounding can be noisy; use geometric approximation
    ann_return = (1 + total_ret) ** (annual_factor) - 1
    return ann_return

def compute_annualized_volatility(price_df):
    daily = price_df.pct_change().dropna()
    ann_vol = daily.std() * np.sqrt(252)
    return ann_vol

def compute_rolling_sharpe(price_df, window=SHARPE_WINDOW):
    daily = price_df.pct_change().dropna()
    # rolling mean (daily) annualized and rolling std (daily) annualized
    roll_mean = daily.rolling(window).mean() * 252
    roll_std = daily.rolling(window).std() * np.sqrt(252)
    roll_sharpe = roll_mean / roll_std
    # return last available rolling value per column
    return roll_sharpe.iloc[-1]

def compute_rolling_beta(asset_prices, nifty_series, window=90):
    if nifty_series is None:
        return pd.Series([np.nan] * len(asset_prices), index=asset_prices.index)

    # daily returns
    asset_ret = asset_prices.pct_change().dropna()
    nifty_ret = nifty_series.pct_change().dropna()

    # align both
    df = pd.concat([asset_ret, nifty_ret], axis=1).dropna()
    df.columns = ["asset", "nifty"]     # <-- now correct because only two columns

    roll_cov = df["asset"].rolling(window).cov(df["nifty"])
    roll_var = df["nifty"].rolling(window).var()

    beta_vals = []
    for cov, var in zip(roll_cov, roll_var):
        if var is not None and not np.isnan(var) and var != 0:
            beta_vals.append(cov / var)
        else:
            beta_vals.append(np.nan)

    beta_series = pd.Series(beta_vals, index=df.index)

    # reindex back to original
    return beta_series.reindex(asset_prices.index)



def compute_metrics_table(price_df, nifty_series):
    metrics = {}

    for etf in price_df.columns:
        series = price_df[etf]

        # returns
        ret = series.pct_change().dropna()

        # annualized return
        ann_return = (1 + ret.mean()) ** 252 - 1

        # annualized vol
        ann_vol = ret.std() * np.sqrt(252)

        # sharpe
        sharpe = ann_return / ann_vol if ann_vol != 0 else np.nan

        # rolling beta (new!)
        beta_series = compute_rolling_beta(series, nifty_series, window=90)
        beta_last = beta_series.dropna().iloc[-1] if beta_series.dropna().size > 0 else np.nan

        metrics[etf] = [
            ann_return,
            ann_vol,
            sharpe,
            beta_last
        ]

    return pd.DataFrame(metrics, index=["Annualized Return", "Annualized Vol", "Sharpe Ratio", "Beta vs NIFTYBEES"])


# ----------------------------
# Layout
# ----------------------------
left_col, mid_col, right_col = st.columns((1, 0.05, 1))

st.title("ETF Tracker — Metrics & Factors")
st.caption("Left: user-selected ETFs and metrics. Right: fixed factor panel.")
st.divider()

# ----------------------------
# LEFT COLUMN (Persistent)
# ----------------------------
with left_col:
    st.header("Left — Controls & Metrics")

    default_end = datetime.today().date()
    default_start = default_end - timedelta(days=730)

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", default_start, key="start_date")
    with col2:
        end_date = st.date_input("End Date", default_end, key="end_date")

    # validation
    if end_date < start_date:
        st.error("End Date cannot be earlier than Start Date.")
        st.stop()
    if start_date > datetime.today().date() or end_date > datetime.today().date():
        st.error("Dates cannot be in the future.")
        st.stop()

    # tickers selection
    master_tickers = [
        "NIFTYBEES.NS", "NEXT50IETF.NS", "BANKBEES.NS", "MASPTOP50.NS", "MON100.NS", "MAFANG.NS",
        "MONQ50.NS", "GOLDBEES.NS", "SILVERETF.NS", "COMMOIETF.NS", "HDFCQUAL.NS", "LOWVOLIETF.NS",
        "MOMOMENTUM.NS", "HDFCGROWTH.NS", "MOVALUE.NS", "SMALLCAP.NS", "MIDCAPETF.NS", "TOP100CASE.NS",
        "ALPHA.NS", "AONETOTAL.NS", "MULTICAP.NS", "MNC.NS", "CONS.NS", "ITIETF.NS", "FMCGIETF.NS",
        "OILIETF.NS", "HEALTHIETF.NS", "EVINDIA.NS", "METAL.NS", "AUTOIETF.NS", "MOREALTY.NS",
        "DIVOPPBEES.NS", "LTGILTBEES.NS", "GSEC5IETF.NS", "CPSEETF.NS", "MAKEINDIA.NS", "LIQUIDCASE.NS",
        "LIQUIDBEES.NS", "VYM", "SCHD", "BBUS", "IYF", "VTI", "JGRO", "MTUM", "QUAL", "JCTR", "SPY",
        "VOO", "USMV", "FEZ", "BBEU"
    ]
    default_tickers = ["MASPTOP50.NS", "NIFTYBEES.NS", "GOLDBEES.NS", "LOWVOLIETF.NS", "MOMOMENTUM.NS", "MOVALUE.NS"]
    selected_tickers = st.multiselect("Select ETFs to analyze", master_tickers, default=default_tickers)

    if not selected_tickers:
        st.error("Please select at least one ETF.")
        st.stop()

    st.divider()

    # Download selected tickers (store in session_state for persistence)
    yf_start = start_date.strftime("%Y-%m-%d")
    yf_end = (end_date + timedelta(days=1)).strftime("%Y-%m-%d")

    # Try to download selected tickers
    raw = yf.download(selected_tickers, start=yf_start, end=yf_end, progress=False)
    prices = safe_extract_prices(raw)
    # if single column and single ticker, set correct column name
    if not prices.empty and len(prices.columns) == 1 and len(selected_tickers) == 1:
        prices.columns = selected_tickers

    # Always ensure NIFTYBEES.NS is downloaded in background for beta calculation
    try:
        if "NIFTYBEES.NS" not in selected_tickers:
            raw_nifty = yf.download("NIFTYBEES.NS", start=yf_start, end=yf_end, progress=False)
            nifty_prices = safe_extract_prices(raw_nifty)
            if not nifty_prices.empty:
                # single column rename
                if len(nifty_prices.columns) == 1:
                    nifty_prices.columns = ["NIFTYBEES.NS"]
        else:
            # if user already selected it, use it from prices
            if "NIFTYBEES.NS" in prices.columns:
                nifty_prices = prices[["NIFTYBEES.NS"]].copy()
            else:
                raw_nifty = yf.download("NIFTYBEES.NS", start=yf_start, end=yf_end, progress=False)
                nifty_prices = safe_extract_prices(raw_nifty)
                if len(nifty_prices.columns) == 1:
                    nifty_prices.columns = ["NIFTYBEES.NS"]
    except Exception:
        nifty_prices = pd.DataFrame()

    # Store in session_state for persistence (keeps LHS stable across reruns)
    st.session_state.prices = prices
    st.session_state.nifty_prices = nifty_prices

    # --- Left cumulative performance chart ---
    st.markdown("### Cumulative Performance (Selected ETFs)")
    if prices.empty:
        st.info("No price data available for the selected tickers/dates.")
    else:
        cum = prices.pct_change().add(1).cumprod().sub(1)
        fig = go.Figure()
        for t in cum.columns:
            fig.add_trace(go.Scatter(x=cum.index, y=cum[t], mode="lines", name=t))
        fig.update_layout(template="plotly_dark", hovermode="x unified", height=380)
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # --- Summary stats table (metrics) ---
    st.markdown("### Summary Statistics (metrics used for scatter & heatmap)")
    metrics_table = compute_metrics_table(prices, nifty_series=(nifty_prices.iloc[:,0] if not nifty_prices.empty else None))
    if metrics_table.empty:
        st.info("Not enough data to compute metrics.")
    else:
        # Show nicely formatted table
        st.dataframe(metrics_table.round(2).style.format("{:.2f}"), use_container_width=True)

    st.divider()

    # --- Correlation heatmap of the metrics (original colors & heatmap feel) ---
    st.subheader("ETF Correlation (Dark Mode)")

    try:
        corr = returns[selected_tickers].corr().round(2)
    
        fig = px.imshow(
            corr,
            text_auto=True,
            color_continuous_scale=["#1b3368", "black", "#7c2f57"],  # same color theme as before
            aspect="auto",
        )
    
        fig.update_layout(
            template="plotly_dark",
            margin=dict(l=50, r=50, t=50, b=50),
            coloraxis_colorbar=dict(title="Corr"),
        )
    
        fig.update_xaxes(side="top")
    
        st.plotly_chart(fig, use_container_width=True)
    
    except Exception as e:
        st.error(f"Error creating correlation heatmap: {e}")
    
    
    


    
    st.subheader("Scatter Plot: Compare Metrics")

    try:
        # Prepare metrics dataframe for scatter use
        df_scatter = metrics_table.T.copy()
        df_scatter["ETF"] = df_scatter.index
    
        # Identify numeric metrics available
        numeric_metrics = df_scatter.select_dtypes(include=["float", "int"]).columns.tolist()
    
        # Remove ETF column if detected
        if "ETF" in numeric_metrics:
            numeric_metrics.remove("ETF")
    
        if len(numeric_metrics) < 2:
            st.warning("Not enough numeric metrics available to generate scatter plot.")
        else:
            # Dropdown selectors
            col1, col2 = st.columns(2)
            with col1:
                x_metric = st.selectbox("Select X-axis Metric", numeric_metrics, index=0)
            with col2:
                y_metric = st.selectbox("Select Y-axis Metric", numeric_metrics, index=1)
    
            # Drop rows missing both values
            df_clean = df_scatter.dropna(subset=[x_metric, y_metric], how="any")
    
            if df_clean.shape[0] < 2:
                st.warning("Insufficient data to render scatter plot.")
            else:
                fig = px.scatter(
                    df_clean,
                    x=x_metric,
                    y=y_metric,
                    text="ETF",
                    title=f"{x_metric} vs {y_metric}",
                )
    
                # 🎨 Customize marker color + size
                fig.update_traces(
                    marker=dict(
                        size=20,        # 3x larger
                        color="#7a384c",   # custom color
                        opacity=0.95,
                    ),
                    textposition="top center"
                )
    
                st.plotly_chart(fig, use_container_width=True)
    
    except Exception as e:
        st.error(f"Error extracting metrics for scatter plot: {e}")



    st.divider()

    # ------------------------------
    # SCATTER PLOT (bottom of LHS)
    # Selectable metrics (the 4 requested)
    # ------------------------------

    
# ---------------------------
# RIGHT COLUMN (fixed factor panel)
# ----------------------------
with right_col:
    st.header("Right — Fixed Factor Panel")

    # Use same date range
    yf_start_f = start_date.strftime("%Y-%m-%d")
    yf_end_f = (end_date + timedelta(days=1)).strftime("%Y-%m-%d")

    raw_factors = yf.download(FACTOR_TICKERS, start=yf_start_f, end=yf_end_f, progress=False)
    factor_prices = safe_extract_prices(raw_factors)
    if factor_prices.empty:
        st.error("No factor data for the selected dates.")
    else:
        # reorder to defined order and rename
        ordered = [t for t in FACTOR_TICKERS if t in factor_prices.columns]
        if ordered:
            factor_prices = factor_prices[ordered]
        factor_prices = factor_prices.rename(columns=FACTOR_MAP)

        # cumulative
        factor_cum = factor_prices.pct_change().add(1).cumprod().sub(1)

        st.subheader("Cumulative Returns (Factors)")
        figf = go.Figure()
        for col in factor_cum.columns:
            figf.add_trace(go.Scatter(x=factor_cum.index, y=factor_cum[col], mode="lines", name=col))
        figf.update_layout(template="plotly_dark", height=420, hovermode="x unified")
        st.plotly_chart(figf, use_container_width=True)

        st.divider()

        st.subheader("Factor Summary Stats")
        # compute simple stats: total return, ann vol, sharpe
        ann_ret_f = compute_annualized_return(factor_prices) * 100
        ann_vol_f = compute_annualized_volatility(factor_prices) * 100
        # approximate sharpe: mean_ann / ann_vol
        daily_f = factor_prices.pct_change().dropna()
        mean_ann_f = daily_f.mean() * 252
        sharpe_f = (mean_ann_f / (daily_f.std() * np.sqrt(252))).round(2)

        factor_stats_df = pd.DataFrame({
            "Total Return (%)": ann_ret_f,
            "Annualized Volatility (%)": ann_vol_f,
            "Sharpe Ratio": sharpe_f
        })

        # reindex to pretty names
        factor_stats_df.index = [FACTOR_MAP.get(i, i) if i in FACTOR_MAP else i for i in factor_stats_df.index]
        st.dataframe(factor_stats_df.round(2).style.format("{:.2f}"), use_container_width=True)










