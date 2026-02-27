"""
Predictive Inventory & Procurement Intelligence Platform
Hackathon MVP — PDR v2
"""

import streamlit as st
import pandas as pd

from utils.data_processor import (
    load_sales_excel,
    get_anchor_customers,
    predict_reorder,
    process_bom,
    forecast_raw_materials,
)

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Procurement Intelligence | MSME",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
def _load_css():
    try:
        with open("style.css") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass

_load_css()

# ── Session State ─────────────────────────────────────────────────────────────
for key in ["sales_df", "bom_df", "anchor_df", "predictions_df", "forecast_df", "processed"]:
    if key not in st.session_state:
        st.session_state[key] = None

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏭 Data Ingestion")
    st.markdown("---")

    st.markdown("### 📂 Sales Register")
    sales_file = st.file_uploader("Upload Sales Excel (.xlsx)", type=["xlsx", "xls"], key="sales_upload")

    st.markdown("### 📂 Bill of Materials")
    bom_file = st.file_uploader("Upload BOM Excel (.xlsx)", type=["xlsx", "xls"], key="bom_upload")

    st.markdown("---")
    run_btn = st.button("▶ Process & Run Insights", use_container_width=True)

    if run_btn:
        if not sales_file:
            st.error("Please upload a Sales Register first.")
        else:
            with st.spinner("Processing data..."):
                # --- Load Sales ---
                df, err = load_sales_excel(sales_file)
                if err:
                    st.error(f"Sales ingestion failed: {err}")
                    st.session_state.sales_df = None
                else:
                    st.session_state.sales_df = df
                    st.success(f"Sales loaded: {len(df):,} clean rows")

                # --- Load BOM ---
                if bom_file:
                    bom_df, berr = process_bom(bom_file)
                    if berr:
                        st.warning(f"BOM issue: {berr}")
                        st.session_state.bom_df = None
                    else:
                        st.session_state.bom_df = bom_df
                        st.success(f"BOM loaded: {len(bom_df)} entries")

                # --- Run Analytics ---
                if st.session_state.sales_df is not None:
                    anchor_df, aerr = get_anchor_customers(st.session_state.sales_df)
                    st.session_state.anchor_df = anchor_df

                    pred_df, perr = predict_reorder(st.session_state.sales_df, anchor_df)
                    st.session_state.predictions_df = pred_df

                    if st.session_state.bom_df is not None and not anchor_df.empty:
                        fc_df, fcerr = forecast_raw_materials(
                            st.session_state.sales_df,
                            anchor_df,
                            pred_df,
                            st.session_state.bom_df
                        )
                        st.session_state.forecast_df = fc_df

                    st.session_state.processed = True

# ── Main Dashboard ────────────────────────────────────────────────────────────
st.markdown("# 🏭 Predictive Inventory & Procurement Dashboard")
st.markdown("*Turning unstructured MSME Excel data into procurement intelligence.*")
st.markdown("---")

if not st.session_state.processed:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**Step 1**\nUpload your Sales Register Excel from the sidebar. Any format accepted.")
    with col2:
        st.info("**Step 2**\n(Optional) Upload your Bill of Materials to enable raw material forecasting.")
    with col3:
        st.info("**Step 3**\nClick **Process & Run Insights** to generate demand predictions.")
    st.stop()

# ─── Section 1: Cleaned Data Preview ─────────────────────────────────────────
st.markdown("## 1. Cleaned Data Preview")
if st.session_state.sales_df is not None and not st.session_state.sales_df.empty:
    df = st.session_state.sales_df
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Rows Loaded", f"{len(df):,}")
    c2.metric("Unique Customers", f"{df['customer'].nunique():,}")
    c3.metric("Date Range", f"{df['date'].min().strftime('%d %b %Y')} → {df['date'].max().strftime('%d %b %Y')}")
    st.dataframe(df.head(20), use_container_width=True)
else:
    st.warning("No clean sales data available. Check that your file has Date, Customer and Quantity columns.")

st.markdown("---")

# ─── Section 2: Anchor Customers ─────────────────────────────────────────────
st.markdown("## 2. Anchor Customers")
if st.session_state.anchor_df is not None and not st.session_state.anchor_df.empty:
    anchor = st.session_state.anchor_df.copy()
    anchor.columns = ['Customer', 'Total Quantity', 'Order Count', 'Contribution %']
    st.markdown("*Top revenue-driving customers ranked by total quantity ordered.*")
    st.dataframe(anchor.style.format({'Total Quantity': '{:,.0f}', 'Contribution %': '{:.1f}%'}), use_container_width=True)

    # Bar chart
    try:
        st.bar_chart(anchor.set_index('Customer')['Total Quantity'])
    except Exception:
        pass
else:
    st.info("Anchor customers will appear here after processing sales data.")

st.markdown("---")

# ─── Section 3: Reorder Prediction Panel ─────────────────────────────────────
st.markdown("## 3. Reorder Prediction Panel")
if st.session_state.predictions_df is not None and not st.session_state.predictions_df.empty:
    pred = st.session_state.predictions_df

    st.markdown("*Customers with fewer than 3 historical orders are excluded from predictions.*")

    # Colour confidence score
    def colour_confidence(val):
        if val >= 70:
            return 'color: green; font-weight: bold'
        elif val >= 40:
            return 'color: orange; font-weight: bold'
        else:
            return 'color: red; font-weight: bold'

    try:
        styled = pred.style.applymap(colour_confidence, subset=['Confidence %'])
        st.dataframe(styled, use_container_width=True)
    except Exception:
        st.dataframe(pred, use_container_width=True)

    # Summary metrics
    if len(pred) > 0:
        c1, c2, c3 = st.columns(3)
        c1.metric("Customers Predicted", len(pred))
        c2.metric("Avg Reorder Interval", f"{pred['Avg Interval (Days)'].mean():.0f} days")
        c3.metric("Avg Confidence", f"{pred['Confidence %'].mean():.1f}%")
else:
    st.info("Predictions require anchor customers with at least 3 historical orders. Check your date column is parsed correctly.")

st.markdown("---")

# ─── Section 4: BOM + Raw Material Forecast ──────────────────────────────────
st.markdown("## 4. Raw Material Forecast")

if st.session_state.bom_df is not None and not st.session_state.bom_df.empty:
    with st.expander("View BOM Mapping", expanded=False):
        st.dataframe(st.session_state.bom_df, use_container_width=True)

    if st.session_state.forecast_df is not None and not st.session_state.forecast_df.empty:
        fc = st.session_state.forecast_df

        st.markdown("*Projected raw material requirement based on anchor customer predicted order quantities × BOM.*")
        st.dataframe(fc, use_container_width=True)

        try:
            st.bar_chart(fc.set_index('Raw Material')['Projected Requirement'])
        except Exception:
            pass
    else:
        st.info("Raw material forecast will appear once both Sales data and BOM are loaded.")
else:
    st.info("Upload a BOM file from the sidebar to enable raw material forecasting.")
