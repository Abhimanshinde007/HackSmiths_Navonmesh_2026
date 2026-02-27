"""
Predictive Inventory & Procurement Intelligence Platform
Full MVP with Tally PDF Support — Streamlit Frontend
"""

import streamlit as st
import pandas as pd

from utils.data_processor import (
    ingest_sales_excel, ingest_sales_pdf,
    get_anchor_customers, predict_reorder,
    ingest_inout_excel, ingest_inout_pdf, compute_stock,
    ingest_purchase_excel, ingest_purchase_pdf, compute_purchase_summary,
    material_outlook,
)

# ── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="MSME Procurement Intelligence",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────
try:
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    pass

# ── Session State ─────────────────────────────────────────────
_keys = ["sales_df", "inout_df", "purchase_df",
         "anchor_df", "predictions_df", "stock_df",
         "purchase_summary", "outlook_df", "processed",
         "sales_qty_label"]
for k in _keys:
    if k not in st.session_state:
        st.session_state[k] = None

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏭 MSME Data Ingestion")
    st.markdown("---")
    st.caption("Accepted formats: **.xlsx / .xls / .pdf** (text-based Tally exports)")

    st.markdown("### 📊 Sales Register")
    sales_file = st.file_uploader("Upload Sales Register", type=["xlsx", "xls", "pdf"], key="sales_up")

    st.markdown("### 📦 Inward / Outward Register")
    inout_file = st.file_uploader("Upload Inward / Outward Register", type=["xlsx", "xls", "pdf"], key="inout_up")

    st.markdown("### 🛒 Purchase Register")
    purchase_file = st.file_uploader("Upload Purchase Register", type=["xlsx", "xls", "pdf"], key="purch_up")

    st.markdown("---")
    run_btn = st.button("▶ Process & Run Insights", use_container_width=True)

    if run_btn:
        with st.spinner("Processing all registers..."):

            # ── Sales ──────────────────────────────────────────
            if sales_file:
                try:
                    is_pdf = sales_file.name.lower().endswith(".pdf")
                    if is_pdf:
                        df, err, qty_label = ingest_sales_pdf(sales_file.read())
                    else:
                        df, err, qty_label = ingest_sales_excel(sales_file)
                    if err:
                        st.error(f"Sales: {err}")
                        st.session_state.sales_df = None
                    else:
                        st.session_state.sales_df = df
                        st.session_state.sales_qty_label = qty_label or 'Value'
                        anchor, aerr = get_anchor_customers(df)
                        st.session_state.anchor_df = anchor
                        if aerr:
                            st.warning(f"Anchor customers: {aerr}")
                        pred, perr = predict_reorder(df, anchor)
                        st.session_state.predictions_df = pred
                        st.success(f"Sales loaded: {len(df):,} rows | {df['customer'].nunique()} customers")
                except Exception as e:
                    st.error(f"Sales processing error: {e}")

            # ── Inward / Outward ────────────────────────────────
            if inout_file:
                try:
                    is_pdf = inout_file.name.lower().endswith(".pdf")
                    if is_pdf:
                        df_io, err = ingest_inout_pdf(inout_file.read())
                    else:
                        df_io, err = ingest_inout_excel(inout_file)
                    if err:
                        st.error(f"Inward/Outward: {err}")
                        st.session_state.inout_df = None
                    else:
                        st.session_state.inout_df = df_io
                        stock, serr = compute_stock(df_io)
                        st.session_state.stock_df = stock
                        st.success(f"Stock register loaded: {df_io['material'].nunique()} materials")
                except Exception as e:
                    st.error(f"Inward/Outward processing error: {e}")

            # ── Purchase ────────────────────────────────────────
            if purchase_file:
                try:
                    is_pdf = purchase_file.name.lower().endswith(".pdf")
                    if is_pdf:
                        df_p, err = ingest_purchase_pdf(purchase_file.read())
                    else:
                        df_p, err = ingest_purchase_excel(purchase_file)
                    if err:
                        st.error(f"Purchase: {err}")
                        st.session_state.purchase_df = None
                    else:
                        st.session_state.purchase_df = df_p
                        summary, _ = compute_purchase_summary(df_p)
                        st.session_state.purchase_summary = summary
                        st.success(f"Purchase register loaded: {len(df_p):,} rows")
                except Exception as e:
                    st.error(f"Purchase processing error: {e}")

            # ── Material Outlook ────────────────────────────────
            if st.session_state.stock_df is not None and not st.session_state.stock_df.empty:
                outlook, oerr = material_outlook(
                    st.session_state.stock_df,
                    st.session_state.predictions_df,
                    st.session_state.sales_df
                )
                st.session_state.outlook_df = outlook

            if sales_file or inout_file or purchase_file:
                st.session_state.processed = True

# ── Main Dashboard ────────────────────────────────────────────
st.markdown("# 🏭 Predictive Inventory & Procurement Dashboard")
st.markdown("*MSME-grade procurement intelligence — no ERP required.*")
st.markdown("---")

if not st.session_state.processed:
    st.info("Upload at least one register from the sidebar and click **Process & Run Insights** to begin.")
    st.stop()

# ── Three-column dashboard ────────────────────────────────────
col_sales, col_stock, col_outlook = st.columns(3)

# ─── Column 1: Sales Insights ─────────────────────────────────
with col_sales:
    st.markdown("### 📊 Sales Insights")

    if st.session_state.sales_df is not None and not st.session_state.sales_df.empty:
        df = st.session_state.sales_df
        st.metric("Total Clean Rows", f"{len(df):,}")
        st.metric("Unique Customers", f"{df['customer'].nunique():,}")
        dr = f"{df['date'].min().strftime('%d %b %y')} → {df['date'].max().strftime('%d %b %y')}"
        st.caption(f"Date range: {dr}")

        if st.session_state.anchor_df is not None and not st.session_state.anchor_df.empty:
            qty_col_label = st.session_state.get('sales_qty_label', 'Value')
            st.markdown("**Anchor Customers**")
            st.caption(f"📌 Ranked by: **{qty_col_label}**")
            anchor = st.session_state.anchor_df.copy()
            anchor.columns = ['Customer', qty_col_label, 'Orders', 'Share %']
            st.dataframe(anchor.style.format({qty_col_label: '{:,.0f}', 'Share %': '{:.1f}%'}),
                        use_container_width=True, height=220)

        if st.session_state.predictions_df is not None and not st.session_state.predictions_df.empty:
            st.markdown("**Reorder Predictions**")
            pred = st.session_state.predictions_df

            def _conf_colour(val):
                if val >= 70: return 'color: green; font-weight:bold'
                elif val >= 40: return 'color: orange'
                return 'color: red'

            try:
                styled = pred.style.applymap(_conf_colour, subset=['Confidence %'])
                st.dataframe(styled, use_container_width=True, height=250)
            except Exception:
                st.dataframe(pred, use_container_width=True, height=250)
        else:
            st.info("Predictions require ≥ 3 orders per customer.")
    else:
        st.info("Upload a Sales Register to see insights.")

# ─── Column 2: Stock Status ───────────────────────────────────
with col_stock:
    st.markdown("### 📦 Stock Status")

    if st.session_state.stock_df is not None and not st.session_state.stock_df.empty:
        stock = st.session_state.stock_df.copy()
        low_stock = stock[stock['current_stock'] <= 0]
        st.metric("Total Materials", len(stock))
        st.metric("Low / Zero Stock", len(low_stock), delta_color="inverse")

        def _highlight_low(row):
            colour = 'background-color: #ffe0e0' if row['current_stock'] <= 0 else ''
            return [colour] * len(row)

        st.dataframe(
            stock.style.apply(_highlight_low, axis=1)
                       .format({'total_inward': '{:,.0f}', 'total_outward': '{:,.0f}', 'current_stock': '{:,.0f}'}),
            use_container_width=True, height=400
        )
    else:
        st.info("Upload an Inward/Outward Register to see stock status.")

    if st.session_state.purchase_summary is not None and not st.session_state.purchase_summary.empty:
        st.markdown("**Purchase Summary**")
        st.dataframe(st.session_state.purchase_summary, use_container_width=True, height=200)

# ─── Column 3: Material Outlook ────────────────────────────────
with col_outlook:
    st.markdown("### 🔮 Material Outlook")

    if st.session_state.outlook_df is not None and not st.session_state.outlook_df.empty:
        outlook = st.session_state.outlook_df

        procure_count = outlook[outlook['Advisory'].str.contains('Prepare', na=False)].shape[0] if 'Advisory' in outlook.columns else 0
        monitor_count = outlook[outlook['Advisory'].str.contains('Monitor', na=False)].shape[0] if 'Advisory' in outlook.columns else 0

        c1, c2 = st.columns(2)
        c1.metric("🔴 Procure Soon", procure_count)
        c2.metric("🟡 Monitor", monitor_count)

        def _highlight_advisory(row):
            if 'Advisory' in row.index and 'Prepare' in str(row['Advisory']):
                return ['background-color: #fff3cd'] * len(row)
            elif 'Advisory' in row.index and 'Out' in str(row['Advisory']):
                return ['background-color: #ffe0e0'] * len(row)
            return [''] * len(row)

        st.dataframe(
            outlook.style.apply(_highlight_advisory, axis=1),
            use_container_width=True, height=420
        )
    elif st.session_state.stock_df is not None:
        st.info("Upload Sales Register + Inward/Outward to generate Material Outlook.")
    else:
        st.info("Upload registers to generate procurement advisory.")
