# app.py - robust, full ETF tracker with persistent left column and fixed-factor right column

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(layout="wide", page_title="ETF Tracker")

# -----------------------
# Helper utilities
# -----------------------
def safe_extract_prices(df):
    """
    Given a dataframe returned by yf.download, return a prices DataFrame
    with tickers as columns. Handles multiindex and single-index cases.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    cols = df.columns
    # MultiIndex (typical when multiple tickers & multiple fields)
    if isinstance(cols, pd.MultiIndex):
        # Prefer 'Adj Close' then 'Close'
        for prefer in ["Adj Close", "Close"]:
            if prefer in cols.levels[0]:
                prices = df[prefer].copy()
                return prices
        # If neither level exists, try to collapse by taking first level if that makes sense
        # Fallback: take last level
        try:
            # try to get the last level (this may be the ticker if structure differs)
            prices = df.xs(df.columns.levels[1][0], axis=1, level=1, drop_level=False)
            return prices
        except Exception:
            return pd.DataFrame()
    else:
        # Single-level columns. Could be 'Adj Close', 'Close' or ticker names.
        if "Adj Close" in df.columns:
            return df[["Adj Close"]].rename(columns={"Adj Close": df.columns.name or "PRICE"})
        if "Close" in df.columns:
            return df[["Close"]].rename(columns={"Close": df.columns.name or "PRICE"})
        # Otherwise assume columns are already tickers (e.g. single ticker download with auto-labeled col)
        return df.copy()

def compute_summary_stats(price_df):
    """Return summary stats DataFrame for given price dataframe."""
    if price_df.empty:
        return pd.DataFrame()

    # daily returns
    daily = price_df.pct_change().dropna()
    # cumulative return
    cumulative = (price_df.iloc[-1] / price_df.iloc[0]) - 1
    ann_vol = daily.std() * np.sqrt(252)
    # sharpe using mean daily returns annualized / ann_vol
    mean_ann = daily.mean() * 252
    sharpe = (mean_ann / ann_vol).replace([np.inf, -np.inf], np.nan)
    var95 = daily.quantile(0.05)

    summary = pd.DataFrame({
        "Total Return (%)": cumulative * 100,
        "Annualized Volatility (%)": ann_vol * 100,
        "Sharpe Ratio": sharpe,
        "VaR 95 (%)": var95 * 100
    })

    # Ensure consistent ordering of columns when price_df has tickers as columns
    summary.index.name = "Ticker"
    return summary

# -----------------------
# Fixed factor ETFs for RHS
# -----------------------
FACTOR_MAP = {
    "MOVALUE.NS": "Value",
    "MOMOMENTUM.NS": "Momentum",
    "HDFCQUAL.NS": "Quality",
    "HDFCGROWTH.NS": "Growth",
    "LOWVOLIETF.NS": "Low Volatility"
}
FACTOR_TICKERS = list(FACTOR_MAP.keys())

# -----------------------
# Layout: two columns
# -----------------------
left_col, mid_col, right_col = st.columns((1, 0.05, 1))

# App header (global)
st.title("ETF Tracker")
st.caption("Track ETFs — left column is persistent, right column shows fixed factor panel.")
st.divider()

# -----------------------
# LEFT COLUMN - full app controls, cumulative chart, stats, scatter (moved earlier)
# -----------------------
with left_col:
    st.markdown("### Analysis Controls")

    # Date inputs (defaults)
    default_end = datetime.today().date()
    default_start = default_end - timedelta(days=730)

    col_a, col_b = st.columns(2)
    with col_a:
        start_date = st.date_input("Start Date", default_start, key="left_start")
    with col_b:
        end_date = st.date_input("End Date", default_end, key="left_end")

    # Basic validation
    if end_date < start_date:
        st.error("🚨 End Date cannot be earlier than Start Date. Please select a valid range.")
        st.stop()

    if start_date > datetime.today().date() or end_date > datetime.today().date():
        st.error("🚨 Dates cannot be in the future.")
        st.stop()

    # ETF selection
    st.markdown("### Select ETFs")
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
    selected_tickers = st.multiselect("Choose ETFs:", master_tickers, default=default_tickers)

    if not selected_tickers:
        st.error("🚨 Please select at least one ETF to proceed.")
        st.stop()

    st.divider()

    # --- SCATTER PLOT MOVED EARLIER IN LHS ---
    st.markdown("### Scatter Plot: Compare Two Metrics (from summary)")
    # We'll compute summary after downloading; show placeholders if not ready
    # Download data for selected tickers
    yf_start = start_date.strftime("%Y-%m-%d")
    yf_end = (end_date + timedelta(days=1)).strftime("%Y-%m-%d")
    raw = yf.download(selected_tickers, start=yf_start, end=yf_end, progress=False)

    prices = safe_extract_prices(raw)
    # If single-column, ensure columns are tickers
    if prices.empty:
        st.warning("No price data returned for selected tickers/dates.")
    else:
        # If the extracted df has columns that are not the tickers (happens when single column), try rename
        # Make sure columns are the selected tickers where possible
        if len(prices.columns) == 1 and len(selected_tickers) == 1:
            prices.columns = selected_tickers
        # compute summary
        summary_stats = compute_summary_stats(prices)
        # If summary is empty or shape mismatch, handle gracefully
        if summary_stats.empty:
            st.warning("Insufficient data to compute metrics for scatter plot.")
        else:
            metrics = summary_stats.index.tolist()
            if len(metrics) >= 2:
                col1, col2 = st.columns(2)
                with col1:
                    metric_x = st.selectbox("Select Metric 1 (x)", metrics, index=0)
                with col2:
                    metric_y = st.selectbox("Select Metric 2 (y)", metrics, index=1)
                # x and y values from summary (ordered by columns/tickers)
                x_vals = summary_stats.loc[metric_x].values
                y_vals = summary_stats.loc[metric_y].values
                # labels are tickers
                labels = summary_stats.columns.tolist()

                fig_scatter = px.scatter(
                    x=x_vals,
                    y=y_vals,
                    labels={"x": metric_x, "y": metric_y},
                    title=f"Scatter: {metric_x} vs {metric_y}",
                    text=labels
                )
                fig_scatter.update_traces(textposition="top center", marker=dict(size=10))
                fig_scatter.update_layout(template="plotly_dark", height=450)
                st.plotly_chart(fig_scatter, use_container_width=True)
            else:
                st.info("Not enough metrics to build scatter plot.")

    st.divider()

    # --- CUMULATIVE PERFORMANCE CHART (LEFT) ---
    st.markdown("### Cumulative Performance (Selected ETFs)")
    if prices.empty:
        st.info("No cumulative chart available (no data).")
    else:
        cumulative = prices.pct_change().add(1).cumprod().sub(1)
        fig = go.Figure()
        for t in prices.columns:
            fig.add_trace(go.Scatter(x=cumulative.index, y=cumulative[t], mode="lines", name=t))
        fig.update_layout(template="plotly_dark", hovermode="x unified", height=420,
                          legend=dict(title="ETFs"))
        st.plotly_chart(fig, use_container_width=True)

    # --- SUMMARY STATS (LEFT) ---
    st.markdown("### ETF Summary (Selected)")
    left_summary = compute_summary_stats(prices)
    if not left_summary.empty:
        # transpose for display like earlier versions (tickers as rows)
        display_df = left_summary.T[selected_tickers].T if (set(selected_tickers) <= set(left_summary.index)) else left_summary
        # format numbers
        display_df_formatted = display_df.copy()
        # Align index order to selected_tickers where possible
        try:
            display_df_formatted = display_df.loc[selected_tickers]
        except Exception:
            pass
        # Round and format
        display_df_formatted = display_df_formatted.round(2)
        st.dataframe(display_df_formatted.style.format("{:.2f}"))
    else:
        st.info("Summary stats not available for selected tickers.")

    # Correlation matrix
    st.markdown("### Correlation (Selected ETFs)")
    try:
        corr = cumulative[selected_tickers].corr().round(2)
        st.dataframe(corr.style.format("{:.2f}"))
    except Exception:
        st.info("Correlation matrix unavailable.")

# small spacer
with mid_col:
    st.write("")

# -----------------------
# RIGHT COLUMN - Fixed factor panel (no user selection)
# -----------------------
with right_col:
    st.markdown("### Factor Dashboard (fixed list)")

    # Ensure we use the same date range selection from left
    # falling back to defaults if not present
    try:
        start_for_factors = start_date
        end_for_factors = end_date
    except Exception:
        start_for_factors = datetime.today().date() - timedelta(days=365 * 2)
        end_for_factors = datetime.today().date()

    yf_start_f = start_for_factors.strftime("%Y-%m-%d")
    yf_end_f = (end_for_factors + timedelta(days=1)).strftime("%Y-%m-%d")

    # Download factor prices
    raw_factors = yf.download(FACTOR_TICKERS, start=yf_start_f, end=yf_end_f, progress=False)

    factor_prices = safe_extract_prices(raw_factors)
    # if single-column and matches single ticker, rename
    if not factor_prices.empty and len(factor_prices.columns) == 1 and len(FACTOR_TICKERS) == 1:
        factor_prices.columns = FACTOR_TICKERS

    if factor_prices.empty:
        st.error("No factor ETF data available for the selected dates.")
    else:
        # ensure columns are in the defined order
        # if factor_prices columns are tickers, reorder; otherwise attempt to map
        cols = list(factor_prices.columns)
        # If extracted columns are ticker strings -> reorder
        try:
            ordered = [c for c in FACTOR_TICKERS if c in factor_prices.columns]
            factor_prices = factor_prices[ordered]
        except Exception:
            pass

        # compute cumulative returns (as pct change cumulative)
        factor_cum = factor_prices.pct_change().add(1).cumprod().sub(1)

        # rename tickers to pretty names for plotting & tables where possible
        rename_map = {ticker: FACTOR_MAP.get(ticker, ticker) for ticker in factor_cum.columns}
        factor_cum_renamed = factor_cum.rename(columns=rename_map)
        factor_prices_renamed = factor_prices.rename(columns=rename_map)

        # Plot cumulative returns (indexed to 0 i.e. cumulative pct)
        st.subheader("Cumulative Returns (Factors)")
        fig = go.Figure()
        for col in factor_cum_renamed.columns:
            fig.add_trace(go.Scatter(x=factor_cum_renamed.index, y=factor_cum_renamed[col],
                                     mode="lines", name=col))
        fig.update_layout(template="plotly_dark", height=450, hovermode="x unified",
                          legend=dict(title="Factors"))
        st.plotly_chart(fig, use_container_width=True)

        st.divider()

        # Summary statistics for factors
        st.subheader("Factor Summary Statistics")
        factor_summary = compute_summary_stats(factor_prices_renamed)
        if not factor_summary.empty:
            # rename index entries to pretty names if index currently tickers
            try:
                # factor_summary index is tickers; replace with pretty names
                pretty_index = [FACTOR_MAP.get(idx, idx) for idx in factor_summary.index]
                factor_summary.index = pretty_index
            except Exception:
                pass

            # Format and show
            factor_summary_display = factor_summary.round(2)
            st.dataframe(factor_summary_display.style.format("{:.2f}"))
        else:
            st.info("Unable to compute factor summary stats.")

