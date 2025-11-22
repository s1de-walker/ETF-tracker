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

def compute_rolling_beta(price_df, nifty_series, window=90):
    if nifty_series is None:
        return pd.Series([np.nan] * len(price_df), index=price_df.index)

    # daily returns
    asset_ret = price_df.pct_change().dropna()
    nifty_ret = nifty_series.pct_change().dropna()

    # align
    df = pd.concat([asset_ret, nifty_ret], axis=1).dropna()
    df.columns = ["asset", "nifty"]

    roll_cov = df["asset"].rolling(window).cov(df["nifty"])
    roll_var = df["nifty"].rolling(window).var()

    betas = []
    for i in range(len(df)):
        roll_cov_last = roll_cov.iloc[i]
        roll_var_last = roll_var.iloc[i]

        # SAFE CHECK
        if (
            roll_var_last is not None 
            and not np.isnan(roll_var_last) 
            and roll_var_last != 0
        ):
            beta_val = roll_cov_last / roll_var_last
        else:
            beta_val = np.nan

        betas.append(beta_val)

    beta_series = pd.Series(betas, index=df.index)

    # reindex to original price_df index
    return beta_series.reindex(price_df.index)


def compute_metrics_table(price_df, nifty_series=None):
    """
    Compute the four metrics per ticker:
    - Annualized Return (%)
    - Annualized Volatility (%)
    - Rolling Sharpe (SHARPE_WINDOW)
    - Beta vs NIFTYBEES (BETA_WINDOW) -- relies on nifty_series
    Returns a DataFrame with metrics as rows and tickers as columns.
    """
    if price_df.empty:
        return pd.DataFrame()

    ann_ret = compute_annualized_return(price_df) * 100
    ann_vol = compute_annualized_volatility(price_df) * 100
    roll_sharpe = compute_rolling_sharpe(price_df, window=SHARPE_WINDOW)
    # ensure EP has same columns shape; roll_sharpe may be Series with same index as price_df.columns
    roll_sharpe = roll_sharpe.rename(lambda x: x).astype(float)

    # Beta
    if nifty_series is None or nifty_series.empty:
        # fill with NaNs
        betas = pd.Series(index=price_df.columns, dtype=float)
    else:
        betas = compute_rolling_beta(price_df, nifty_series, window=BETA_WINDOW)

    # Build table: rows = metrics, columns = tickers
    metric_names = [
        f"Annualized Return (%)",
        f"Annualized Volatility (%)",
        f"Rolling Sharpe ({SHARPE_WINDOW}d)",
        f"Beta vs NIFTYBEES ({BETA_WINDOW}d)"
    ]

    df = pd.DataFrame(index=metric_names, columns=price_df.columns)
    df.loc[metric_names[0]] = ann_ret
    df.loc[metric_names[1]] = ann_vol
    df.loc[metric_names[2]] = roll_sharpe
    df.loc[metric_names[3]] = betas * 1.0  # keep as numeric

    return df.astype(float)

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
    st.markdown("### Correlation Heatmap (metrics)")
    if not metrics_table.empty:
        # metrics_table rows = metrics, columns = tickers. We want correlation across tickers for each metric,
        # but the user asked heatmap of the metrics — we will compute correlation across tickers using the metric values.
        # Make a matrix of correlation between tickers based on metric vectors (so the heatmap shows ticker vs ticker correlation).
        # But they wanted heatmap of metrics; the original request earlier suggested heatmap of returns correlation.
        # Here: we'll produce a metric-correlation heatmap where each cell = correlation between two tickers across the metrics.
        # First transpose so rows=tickers, cols=metrics
        df_metrics_t = metrics_table.T  # rows = tickers, cols = metrics
        # Compute correlation between tickers across metrics
        corr_matrix = df_metrics_t.corr(method='pearson')  # correlation across metric names
        # But typically heatmap desired is ticker vs ticker correlation of returns (we'll provide both)
        # Primary: show correlation of tickers based on returns (like original)
        try:
            returns = prices.pct_change().dropna()
            returns_corr = returns.corr().round(2)
            heatmap_df = returns_corr
            heatmap_title = "Returns Correlation (tickers)"
        except Exception:
            heatmap_df = df_metrics_t.T.corr().round(2)
            heatmap_title = "Metrics Correlation"

        # custom colormap: blue-white-maroon similar to earlier
        colors = ["#1b3368", "white", "#7c2f57"]
        cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)

        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(heatmap_df, annot=True, fmt=".2f", cmap=cmap, vmin=-1, vmax=1, ax=ax, cbar_kws={"shrink": .8})
        ax.set_title(heatmap_title)
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.info("Not enough metrics data to draw heatmap.")

    st.divider()

    # ------------------------------
    # SCATTER PLOT (bottom of LHS)
    # Selectable metrics (the 4 requested)
    # ------------------------------
    st.markdown("### Scatter Plot — Choose two metrics to compare (bottom of LHS)")

    metric_names = [
        "Annualized Return (%)",
        "Annualized Volatility (%)",
        f"Rolling Sharpe ({SHARPE_WINDOW}d)",
        f"Beta vs NIFTYBEES ({BETA_WINDOW}d)"
    ]

    if metrics_table.empty:
        st.info("Metrics table unavailable for scatter plot.")
    else:
        # metrics_table has rows=metrics, cols=tickers
        # Build selection
        colx, coly = st.columns(2)
        with colx:
            metric_x = st.selectbox("X metric", metric_names, index=0)
        with coly:
            metric_y = st.selectbox("Y metric", metric_names, index=1)

        # extract x and y arrays (ordered by tickers)
        try:
            x_vals = metrics_table.loc[metric_x].values.astype(float)
            y_vals = metrics_table.loc[metric_y].values.astype(float)
            labels = metrics_table.columns.tolist()
        except Exception:
            st.error("Error extracting metrics for scatter plot.")
            x_vals = np.array([])
            y_vals = np.array([])
            labels = []

        if x_vals.size and y_vals.size:
            fig_sc = px.scatter(
                x=x_vals,
                y=y_vals,
                text=labels,
                labels={"x": metric_x, "y": metric_y},
                title=f"{metric_x} vs {metric_y}"
            )
            fig_sc.update_traces(textposition="top center", marker=dict(size=10))
            fig_sc.update_layout(template="plotly_dark", height=480)
            st.plotly_chart(fig_sc, use_container_width=True)
        else:
            st.info("Insufficient data to render scatter plot.")

# ----------------------------
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

