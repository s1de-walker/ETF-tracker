# app.py (fixed + session_state persistence)

import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from matplotlib.colors import LinearSegmentedColormap

# Make the app widescreen
st.set_page_config(layout="wide")

st.title("ETF Tracker")

# --- layout columns
left_col, colmid, right_col = st.columns((1, 0.1, 1))

# Title (top-level, keep visible)

st.caption("Track your ETFs")
st.divider()

# initialize session state keys we will use
if "left_ready" not in st.session_state:
    st.session_state.left_ready = False
if "data" not in st.session_state:
    st.session_state.data = None
if "returns" not in st.session_state:
    st.session_state.returns = None
if "summary" not in st.session_state:
    st.session_state.summary = None
if "selected_tickers" not in st.session_state:
    st.session_state.selected_tickers = []

# ----------------------------
# LEFT COLUMN (persistent)
# ----------------------------
with left_col:
    st.markdown("### Select Time Period for Analysis")

    # Use date objects for Streamlit defaults
    default_start = datetime.today().date() - timedelta(days=730)
    default_end = datetime.today().date()

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", default_start)
    with col2:
        end_date = st.date_input("End Date", default_end)

    # Display selected period in months
    days_diff = (end_date - start_date).days
    months_diff = days_diff / 30 if days_diff >= 0 else 0
    st.caption(f"Selected Period: **{months_diff:.0f} months**")

    # validation
    error_flag = False
    if end_date < start_date:
        st.error("🚨 End Date cannot be earlier than Start Date. Please select a valid range.")
        error_flag = True
    if start_date > datetime.today().date() or end_date > datetime.today().date():
        st.error("🚨 Dates cannot be in the future. Please select a valid range.")
        error_flag = True

    # ETF list (fixed missing comma after MULTICAP.NS)
    tickers = [
        "NIFTYBEES.NS", "NEXT50IETF.NS", "BANKBEES.NS", "MASPTOP50.NS", "MON100.NS", "MAFANG.NS",
        "MONQ50.NS", "GOLDBEES.NS", "SILVERETF.NS", "COMMOIETF.NS", "HDFCQUAL.NS", "LOWVOLIETF.NS",
        "MOMOMENTUM.NS", "HDFCGROWTH.NS", "MOVALUE.NS", "SMALLCAP.NS", "MIDCAPETF.NS", "TOP100CASE.NS",
        "ALPHA.NS", "AONETOTAL.NS", "MULTICAP.NS", "MNC.NS", "CONS.NS", "ITIETF.NS", "FMCGIETF.NS",
        "OILIETF.NS", "HEALTHIETF.NS", "EVINDIA.NS", "METAL.NS", "AUTOIETF.NS", "MOREALTY.NS",
        "DIVOPPBEES.NS", "LTGILTBEES.NS", "GSEC5IETF.NS", "CPSEETF.NS", "MAKEINDIA.NS", "LIQUIDCASE.NS",
        "LIQUIDBEES.NS", "VYM", "SCHD", "BBUS", "IYF", "VTI", "JGRO", "MTUM", "QUAL", "JCTR", "SPY",
        "VOO", "USMV", "FEZ", "BBEU"
    ]

    default_tickers = ["MASPTOP50.NS", "NIFTYBEES.NS", "GOLDBEES.NS", "LOWVOLIETF.NS", "MOMOMENTUM.NS", "MON100.NS"]
    selected_tickers = st.multiselect("Choose ETFs:", tickers, default=default_tickers)

    # Check selection
    if not selected_tickers:
        st.error("🚨 Please select at least one ETF to proceed.")
        error_flag = True

    st.divider()

    # If no validation errors, download and compute left-side visuals & stats and STORE to session_state
    if not error_flag:
        # yfinance expects string dates; include end_date +1 day to make end inclusive like original code
        yf_start = start_date.strftime("%Y-%m-%d")
        yf_end = (end_date + timedelta(days=1)).strftime("%Y-%m-%d")

        # Download
        try:
            data = yf.download(selected_tickers, start=yf_start, end=yf_end)["Close"]
        except Exception as e:
            st.error("Error downloading data: " + str(e))
            data = pd.DataFrame()

        if data.empty:
            st.warning("No data returned for selected tickers/dates.")
            st.session_state.left_ready = False
        else:
            # calculate compounded returns for cumulative performance
            returns = data.pct_change().add(1).cumprod() - 1
            daily_returns = data.pct_change()

            # Beta vs NIFTYBEES.NS (if present)
            if "NIFTYBEES.NS" in selected_tickers and "NIFTYBEES.NS" in data.columns:
                cov_with_n50 = daily_returns.cov().loc[selected_tickers, "NIFTYBEES.NS"]
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

            # Filter to only selected tickers (keeps order)
            filtered_summary_stats = summary_stats[selected_tickers]

            # Save to session_state using the same variable names you'll read later
            st.session_state.start_date = start_date
            st.session_state.end_date = end_date
            st.session_state.data = data
            st.session_state.returns = returns
            st.session_state.daily_returns = daily_returns
            st.session_state.summary = filtered_summary_stats
            st.session_state.selected_tickers = selected_tickers
            st.session_state.left_ready = True

            # --- Present left-side visuals (cumulative performance)
            if not returns.empty:
                st.markdown("### Cumulative Performance")
                fig = go.Figure()
                for ticker in selected_tickers:
                    if ticker in returns.columns:
                        fig.add_trace(go.Scatter(
                            x=returns.index,
                            y=returns[ticker],
                            mode="lines",
                            name=ticker,
                            line=dict(width=2)
                        ))
                fig.update_layout(
                    template="plotly_dark",
                    hovermode="x unified",
                    legend=dict(title="ETFs", bgcolor="rgba(0,0,0,0.5)"),
                    margin=dict(l=40, r=40, t=40, b=40),
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)

            # show metrics and table
            if not filtered_summary_stats.empty:
                st.subheader("ETF Summary")
                best_performer = filtered_summary_stats.loc["Total Return (%)"].idxmax()
                best_performer_value = filtered_summary_stats.loc["Total Return (%)", best_performer]
                most_volatile = filtered_summary_stats.loc["Annualized Volatility (%)"].idxmax()
                most_volatile_value = filtered_summary_stats.loc["Annualized Volatility (%)", most_volatile]
                best_sharpe = filtered_summary_stats.loc["Sharpe Ratio"].idxmax()
                best_sharpe_value = filtered_summary_stats.loc["Sharpe Ratio", best_sharpe]

                col1_metric, col2_metric, col3_metric = st.columns(3)
                col1_metric.metric("🚀 Highest Return ETF", best_performer, f"{best_performer_value:.0f}%")
                col2_metric.metric("⚡ Most Volatile ETF", most_volatile, f"{most_volatile_value:.0f}%")
                col3_metric.metric("🎯 Best Sharpe Ratio", best_sharpe, f"{best_sharpe_value:.1f}")

                st.dataframe(filtered_summary_stats.T.style.format("{:.1f}"))

            # correlation matrix
            try:
                correlation_matrix = returns[selected_tickers].corr().round(2)
                colors = ["#1b3368", "white", "#7c2f57"]
                cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)
                st.subheader("ETF Correlation")
                st.dataframe(correlation_matrix.style.format("{:.2f}").background_gradient(cmap=cmap, axis=None, vmin=-1, vmax=1))
            except Exception:
                # If correlation fails, just skip
                pass

# small spacer column in middle
with colmid:
    st.write("")

# ----------------------------
# RIGHT COLUMN (dynamic)
# ----------------------------
with right_col:
    st.write("")  # keep layout spacing
    st.markdown("### Right Panel (dynamic)")

    if not st.session_state.left_ready:
        st.info("Select date range & ETFs on the left to enable right-panel analysis.")
    else:
        # Read from session_state
        data = st.session_state.data
        returns = st.session_state.returns
        daily_returns = st.session_state.daily_returns
        summary = st.session_state.summary
        selected_tickers = st.session_state.selected_tickers

        # Show dates (these come from left)
        st.write("Selected start:", st.session_state.start_date)
        st.write("Selected end:", st.session_state.end_date)
        st.write("Selected tickers:", ", ".join(selected_tickers))

        # Example dynamic content: scatter plot compare two metrics from summary
        if summary is not None and not summary.empty:
            st.markdown("### Scatter Plot: Compare Two Metrics")
            metrics = summary.index.tolist()
            col_a, col_b = st.columns(2)
            with col_a:
                metric_1 = st.selectbox("Select Metric 1 (x)", metrics, index=0)
            with col_b:
                metric_2 = st.selectbox("Select Metric 2 (y)", metrics, index=1 if len(metrics) > 1 else 0)

            x_values = summary.loc[metric_1].values
            y_values = summary.loc[metric_2].values

            fig_scatter = px.scatter(x=x_values, y=y_values, title=f"Scatter Plot: {metric_1} vs {metric_2}")
            fig_scatter.update_layout(template="plotly_dark", xaxis_title='', yaxis_title='', height=500)
            fig_scatter.update_traces(marker=dict(size=10), hovertemplate='%{text}', text=summary.columns)
            st.plotly_chart(fig_scatter, use_container_width=True)

        # Example dynamic analysis: simple VaR table using daily_returns
        if daily_returns is not None and not daily_returns.empty:
            st.markdown("### Simple VaR (Historical)")
            var_95 = daily_returns.quantile(0.05) * 100
            var_df = var_95.loc[selected_tickers].to_frame(name="VaR 95 (%)")
            st.table(var_df.round(2))


