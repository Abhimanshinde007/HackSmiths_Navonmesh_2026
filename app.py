"""
Predictive Inventory & Procurement Intelligence Platform
Full MVP — Streamlit Frontend
Supports: Individual GST Invoice PDFs, Sales Register Excel, Purchase & Inward/Outward Registers
"""

import streamlit as st
import pandas as pd

from utils.data_processor import (
    ingest_sales_excel,
    get_anchor_customers, predict_reorder,
    ingest_inout_excel, ingest_inout_pdf, compute_stock,
    ingest_purchase_excel, ingest_purchase_pdf, compute_purchase_summary,
    material_outlook,
)
from utils.invoice_parser import ingest_multiple_invoices

# ── Load Gemini API key from Streamlit secrets (if available) ─
GEMINI_KEY = st.secrets.get("GEMINI_API_KEY", None)
if GEMINI_KEY == "paste-your-key-here":
    GEMINI_KEY = None  # treat placeholder as no key


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
         "sales_qty_label", "sales_invoice_count"]
for k in _keys:
    if k not in st.session_state:
        st.session_state[k] = None

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏭 MSME Data Ingestion")
    st.markdown("---")

    # ── Sales Section ─────────────────────────────────────────
    st.markdown("### 📊 Sales Data")
    st.caption("Choose your format:")
    sales_mode = st.radio("Sales upload type", ["Individual GST Invoice PDFs", "Sales Register (Excel)"],
                          key="sales_mode_radio", label_visibility="collapsed")

    if sales_mode == "Individual GST Invoice PDFs":
        st.caption("📄 Upload Tally GST Invoices (PDF or Converted Excel)")
        sales_pdfs = st.file_uploader("Upload Invoice Files", type=["pdf", "xlsx", "xls"],
                                       accept_multiple_files=True, key="sales_pdf_up")
        sales_excel = None

    else:
        st.caption("📊 Upload a consolidated Sales Register Excel")
        sales_excel = st.file_uploader("Upload Sales Register Excel", type=["xlsx", "xls"],
                                        key="sales_excel_up")
        sales_pdfs = None

    st.markdown("### 📦 Inward / Outward Register")
    inout_file = st.file_uploader("Upload Stock Register (.xlsx / .pdf)", type=["xlsx", "xls", "pdf"], key="inout_up")

    st.markdown("### 🛒 Purchase Register")
    purchase_file = st.file_uploader("Upload Purchase Register (.xlsx / .pdf)", type=["xlsx", "xls", "pdf"], key="purch_up")

    st.markdown("---")
    run_btn = st.button("▶ Process & Run Insights", use_container_width=True)

    if run_btn:
        with st.spinner("Processing all registers..."):

            # ── Sales: Individual Invoice PDFs ──────────────────
            if sales_mode == "Individual GST Invoice PDFs" and sales_pdfs:
                try:
                    if GEMINI_KEY:
                        st.info("🤖 Using Gemini AI to parse invoices...")
                    else:
                        st.warning("⚠ No Gemini API key — using basic parser. Add key to .streamlit/secrets.toml for AI parsing.")
                    file_list = [(f.name, f.read()) for f in sales_pdfs]
                    df, errs = ingest_multiple_invoices(file_list, api_key=GEMINI_KEY)

                    if errs:
                        for e in errs:
                            st.warning(f"⚠ {e}")

                    if df is not None and not df.empty:
                        # Drop physical qty column, use invoice amount as 'quantity' for ranking
                        # (amount = invoice value in ₹, more meaningful for MSME ranking)
                        df_std = df.drop(columns=['quantity'], errors='ignore').rename(columns={'amount': 'quantity'})

                        st.session_state.sales_df = df_std
                        st.session_state.sales_qty_label = 'Invoice Value (₹)'
                        st.session_state.sales_invoice_count = len(sales_pdfs)

                        anchor, aerr = get_anchor_customers(df_std)
                        st.session_state.anchor_df = anchor
                        pred, _ = predict_reorder(df_std, anchor)
                        st.session_state.predictions_df = pred
                        st.success(f"✅ Parsed {len(sales_pdfs)} invoices → {len(df):,} line items | {df['customer'].nunique()} customers")
                    else:
                        st.error("Could not extract data from the uploaded PDFs.")
                except Exception as e:
                    st.error(f"Invoice parsing error: {e}")

            # ── Sales: Excel Register ───────────────────────────
            elif sales_mode == "Sales Register (Excel)" and sales_excel:
                try:
                    df, err, qty_label = ingest_sales_excel(sales_excel)
                    if err:
                        st.error(f"Sales Excel: {err}")
                    else:
                        st.session_state.sales_df = df
                        st.session_state.sales_qty_label = qty_label or 'Value'
                        anchor, _ = get_anchor_customers(df)
                        st.session_state.anchor_df = anchor
                        pred, _ = predict_reorder(df, anchor)
                        st.session_state.predictions_df = pred
                        st.success(f"✅ Sales loaded: {len(df):,} rows | {df['customer'].nunique()} customers")
                except Exception as e:
                    st.error(f"Sales Excel error: {e}")

            # ── Inward / Outward ────────────────────────────────
            if inout_file:
                try:
                    if inout_file.name.lower().endswith(".pdf"):
                        df_io, err = ingest_inout_pdf(inout_file.read())
                    else:
                        df_io, err = ingest_inout_excel(inout_file)
                    if err:
                        st.error(f"Stock Register: {err}")
                    else:
                        st.session_state.inout_df = df_io
                        stock, _ = compute_stock(df_io)
                        st.session_state.stock_df = stock
                        st.success(f"✅ Stock register: {df_io['material'].nunique()} materials")
                except Exception as e:
                    st.error(f"Stock register error: {e}")

            # ── Purchase ────────────────────────────────────────
            if purchase_file:
                try:
                    if purchase_file.name.lower().endswith(".pdf"):
                        df_p, err = ingest_purchase_pdf(purchase_file.read())
                    else:
                        df_p, err = ingest_purchase_excel(purchase_file)
                    if err:
                        st.error(f"Purchase Register: {err}")
                    else:
                        st.session_state.purchase_df = df_p
                        summary, _ = compute_purchase_summary(df_p)
                        st.session_state.purchase_summary = summary
                        st.success(f"✅ Purchase register: {len(df_p):,} rows")
                except Exception as e:
                    st.error(f"Purchase register error: {e}")

            # ── Material Outlook ────────────────────────────────
            if st.session_state.stock_df is not None and not st.session_state.stock_df.empty:
                outlook, _ = material_outlook(
                    st.session_state.stock_df,
                    st.session_state.predictions_df,
                    st.session_state.sales_df
                )
                st.session_state.outlook_df = outlook

            if (sales_pdfs or sales_excel or inout_file or purchase_file):
                st.session_state.processed = True

# ── Main Dashboard ────────────────────────────────────────────
st.markdown("# 🏭 Predictive Inventory & Procurement Dashboard")
st.markdown("*MSME-grade procurement intelligence — no ERP required.*")
st.markdown("---")

if not st.session_state.processed:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**Step 1 — Sales Data**\nUpload individual Tally GST Invoice PDFs (multiple at once) OR a Sales Register Excel from the sidebar.")
    with col2:
        st.info("**Step 2 — Stock Register**\n(Optional) Upload Inward/Outward register to track current stock levels.")
    with col3:
        st.info("**Step 3 — Run Insights**\nClick Process to generate anchor customer predictions and material outlook.")
    st.stop()

# ── Three-column dashboard ────────────────────────────────────
col_sales, col_stock, col_outlook = st.columns(3)

# ─── Column 1: Sales Insights ─────────────────────────────────
with col_sales:
    st.markdown("### 📊 Sales Insights")

    if st.session_state.sales_df is not None and not st.session_state.sales_df.empty:
        df = st.session_state.sales_df
        inv_count = st.session_state.sales_invoice_count
        if inv_count:
            st.metric("Invoices Parsed", inv_count)
        st.metric("Line Items", f"{len(df):,}")
        st.metric("Unique Customers", f"{df['customer'].nunique():,}")
        if 'date' in df.columns and df['date'].notna().any():
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
                st.dataframe(pred.style.applymap(_conf_colour, subset=['Confidence %']),
                             use_container_width=True, height=250)
            except Exception:
                st.dataframe(pred, use_container_width=True, height=250)
        else:
            st.info("Predictions require ≥ 3 orders per customer.")

        # Show raw invoice data in expander
        if 'invoice_no' in df.columns:
            with st.expander("View All Invoice Data"):
                st.dataframe(df, use_container_width=True, height=300)
    else:
        st.info("Upload Sales data to see insights.")

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

        try:
            st.dataframe(
                stock.style.apply(_highlight_low, axis=1)
                           .format({'total_inward': '{:,.0f}', 'total_outward': '{:,.0f}', 'current_stock': '{:,.0f}'}),
                use_container_width=True, height=360
            )
        except Exception:
            st.dataframe(stock, use_container_width=True, height=360)
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
        advisory_col = [c for c in outlook.columns if 'advisory' in c.lower()]

        if advisory_col:
            c1, c2 = st.columns(2)
            procure = outlook[outlook[advisory_col[0]].str.contains('Prepare', na=False)].shape[0]
            monitor = outlook[outlook[advisory_col[0]].str.contains('Monitor', na=False)].shape[0]
            c1.metric("🔴 Procure Soon", procure)
            c2.metric("🟡 Monitor", monitor)

        def _highlight_advisory(row):
            a = [c for c in row.index if 'advisory' in c.lower()]
            if a and 'Prepare' in str(row[a[0]]):
                return ['background-color: #fff3cd'] * len(row)
            elif a and 'Out' in str(row[a[0]]):
                return ['background-color: #ffe0e0'] * len(row)
            return [''] * len(row)

        try:
            st.dataframe(outlook.style.apply(_highlight_advisory, axis=1),
                         use_container_width=True, height=420)
        except Exception:
            st.dataframe(outlook, use_container_width=True, height=420)
    elif st.session_state.stock_df is not None:
        st.info("Upload Sales + Inward/Outward data together to generate Material Outlook.")
    else:
        st.info("Upload registers to generate procurement advisory.")
