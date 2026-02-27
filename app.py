import streamlit as st
import pandas as pd

from utils.data_processor import (
    ingest_sales_excel,
    ingest_purchase_excel,
    ingest_inward_excel,
    ingest_outward_excel,
    combine_stock_registers,
    get_anchor_customers,
    predict_reorder,
    compute_stock,
    material_outlook,
)

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Inventory Intelligence Engine",
    page_icon="🏭",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────
# STYLING
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Background */
.stApp { background-color: #F0F2F6; }

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #0B3D91;
    color: white;
}
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label { color: white !important; }
[data-testid="stSidebar"] .stFileUploader label { color: white !important; }
[data-testid="stSidebar"] .stButton>button {
    background-color: #F4B400;
    color: #0B3D91;
    font-weight: 700;
    border: none;
    border-radius: 6px;
    padding: 0.5rem 1rem;
    width: 100%;
}
[data-testid="stSidebar"] .stButton>button:hover {
    background-color: #e0a200;
    color: #0B3D91;
}

/* KPI Cards */
.kpi-row { display: flex; gap: 16px; margin-bottom: 24px; flex-wrap: wrap; }
.kpi-card {
    background: white;
    border-radius: 10px;
    padding: 18px 24px;
    flex: 1;
    min-width: 160px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    border-left: 5px solid #0B3D91;
}
.kpi-card .kpi-label { font-size: 12px; font-weight: 600; color: #666; text-transform: uppercase; letter-spacing: 0.05em; }
.kpi-card .kpi-value { font-size: 32px; font-weight: 700; color: #0B3D91; margin-top: 4px; }
.kpi-card .kpi-value.accent { color: #F4B400; }

/* Panel headers */
.panel-header {
    background: #0B3D91;
    color: white;
    padding: 10px 16px;
    border-radius: 8px 8px 0 0;
    font-weight: 600;
    font-size: 14px;
    letter-spacing: 0.04em;
}

/* Status badges */
.badge-buy  { background:#d32f2f; color:white; padding:2px 10px; border-radius:12px; font-size:12px; font-weight:600; }
.badge-prep { background:#F4B400; color:#0B3D91; padding:2px 10px; border-radius:12px; font-size:12px; font-weight:600; }
.badge-mon  { background:#388e3c; color:white; padding:2px 10px; border-radius:12px; font-size:12px; font-weight:600; }

/* Divider */
hr { border: none; border-top: 1px solid #ddd; margin: 12px 0; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────
for key in ['sales_df', 'purchase_df', 'stock_df',
            'anchor_df', 'predictions_df', 'stock_summary',
            'outlook_df', 'processed']:
    if key not in st.session_state:
        st.session_state[key] = None

if 'processed' not in st.session_state:
    st.session_state.processed = False


# ─────────────────────────────────────────────────────────────
# SIDEBAR — UPLOAD PANELS
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("# 🏭 Inventory Intelligence")
    st.markdown("*MSME-grade procurement intelligence*")
    st.markdown("---")

    # Sales Bills
    st.markdown("### 📊 Sales Bills")
    st.caption("Upload one or more Excel sales bills")
    sales_files = st.file_uploader(
        "Sales Bills", type=["xlsx", "xls"],
        accept_multiple_files=True, key="sales_up"
    )

    st.markdown("### 🛒 Purchase Bills")
    st.caption("Upload one or more Excel purchase bills")
    purchase_files = st.file_uploader(
        "Purchase Bills", type=["xlsx", "xls"],
        accept_multiple_files=True, key="purch_up"
    )

    st.markdown("### 📥 Inward Register")
    st.caption("Materials received from suppliers")
    inward_files = st.file_uploader(
        "Inward Register", type=["xlsx", "xls"],
        accept_multiple_files=True, key="inward_up"
    )

    st.markdown("### 📤 Outward Register")
    st.caption("Goods dispatched to customers")
    outward_files = st.file_uploader(
        "Outward Register", type=["xlsx", "xls"],
        accept_multiple_files=True, key="outward_up"
    )

    st.markdown("---")
    run_btn = st.button("▶ Process & Generate Insights", use_container_width=True)


# ─────────────────────────────────────────────────────────────
# PROCESSING
# ─────────────────────────────────────────────────────────────
if run_btn:
    if not sales_files and not purchase_files and not stock_file:
        st.warning("Please upload at least one file before processing.")
    else:
        with st.spinner("Processing files..."):

            # ── Sales ────────────────────────────────────────
            if sales_files:
                df_s, errs_s = ingest_sales_excel(sales_files)
                for e in errs_s:
                    st.sidebar.error(f"⚠ {e}")
                if df_s is not None and not df_s.empty:
                    st.session_state.sales_df = df_s
                    anchor, aerr = get_anchor_customers(df_s)
                    if aerr:
                        st.sidebar.warning(f"Anchor: {aerr}")
                    else:
                        st.session_state.anchor_df = anchor
                        pred, perr = predict_reorder(df_s, anchor)
                        if not perr:
                            st.session_state.predictions_df = pred
                    st.sidebar.success(f"✅ Sales: {len(df_s):,} rows | {df_s['customer'].nunique()} customers")
                else:
                    if not errs_s:
                        st.sidebar.warning("Sales: No valid rows extracted.")

            # ── Purchases ────────────────────────────────────
            if purchase_files:
                df_p, errs_p = ingest_purchase_excel(purchase_files)
                for e in errs_p:
                    st.sidebar.error(f"⚠ {e}")
                if df_p is not None and not df_p.empty:
                    st.session_state.purchase_df = df_p
                    st.sidebar.success(f"✅ Purchase: {len(df_p):,} rows")

            # -- Inward Register (materials received) ----------
            inward_df, outward_df = pd.DataFrame(), pd.DataFrame()
            if inward_files:
                inward_df, errs_in = ingest_inward_excel(list(inward_files))
                for e in errs_in:
                    st.sidebar.error(f"IN: {e}")
                if not inward_df.empty:
                    st.sidebar.success(f"✅ Inward: {len(inward_df):,} rows | {inward_df['material'].nunique()} materials")

            # -- Outward Register (goods dispatched) -----------
            if outward_files:
                outward_df, errs_out = ingest_outward_excel(list(outward_files))
                for e in errs_out:
                    st.sidebar.error(f"OUT: {e}")
                if not outward_df.empty:
                    st.sidebar.success(f"✅ Outward: {len(outward_df):,} rows | {outward_df['material'].nunique()} materials")

            if not inward_df.empty or not outward_df.empty:
                combined = combine_stock_registers(inward_df, outward_df)
                st.session_state.stock_df = combined
                stk, serr = compute_stock(combined)
                if not serr:
                    st.session_state.stock_summary = stk
                    out_df, oerr = material_outlook(
                        combined,
                        st.session_state.predictions_df,
                        st.session_state.sales_df
                    )
                    if not oerr:
                        st.session_state.outlook_df = out_df


            st.session_state.processed = True
        st.rerun()


# ─────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────
st.markdown("# 🏭 Inventory Intelligence Engine")
st.markdown("*Predictive procurement analytics for MSMEs — no ERP required*")
st.markdown("---")

if not st.session_state.processed:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**Step 1 → Sales Bills**\nUpload your Excel sales invoices in the sidebar.")
    with col2:
        st.info("**Step 2 → Purchase Bills**\nUpload your Excel purchase invoices.")
    with col3:
        st.info("**Step 3 → Stock Register**\nUpload your inward/outward register, then click Process.")
    st.stop()


# ─────────────────────────────────────────────────────────────
# KPI CARDS
# ─────────────────────────────────────────────────────────────
sales_df     = st.session_state.sales_df
anchor_df    = st.session_state.anchor_df
stock_df     = st.session_state.stock_df
purchase_df  = st.session_state.purchase_df
predictions  = st.session_state.predictions_df
stock_sum    = st.session_state.stock_summary
outlook_df   = st.session_state.outlook_df

total_customers  = sales_df['customer'].nunique() if sales_df is not None and not sales_df.empty else 0
anchor_count     = len(anchor_df) if anchor_df is not None and not anchor_df.empty else 0
total_materials  = stock_df['material'].nunique() if stock_df is not None and not stock_df.empty else 0
stock_value_str  = f"₹{purchase_df['quantity'].sum() * purchase_df['rate'].dropna().mean():,.0f}" \
                   if purchase_df is not None and not purchase_df.empty and 'rate' in purchase_df.columns else "N/A"

st.markdown(f"""
<div class="kpi-row">
  <div class="kpi-card">
    <div class="kpi-label">Total Customers</div>
    <div class="kpi-value">{total_customers}</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">Anchor Customers</div>
    <div class="kpi-value accent">{anchor_count}</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">Total Materials</div>
    <div class="kpi-value">{total_materials}</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">Predicted Reorders</div>
    <div class="kpi-value">{len(predictions) if predictions is not None and not predictions.empty else 0}</div>
  </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# 3-COLUMN DASHBOARD
# ─────────────────────────────────────────────────────────────
col_sales, col_stock, col_outlook = st.columns(3)

# ── Column 1: Sales Intelligence ─────────────────────────────
with col_sales:
    st.markdown('<div class="panel-header">📊 Sales Intelligence</div>', unsafe_allow_html=True)
    st.markdown("")

    if sales_df is not None and not sales_df.empty:
        st.metric("Invoices / Line Items", f"{sales_df['source'].nunique() if 'source' in sales_df.columns else '—'} files / {len(sales_df):,}")
        st.metric("Date Range",
                  f"{sales_df['date'].min().strftime('%d %b %y')} → {sales_df['date'].max().strftime('%d %b %y')}"
                  if sales_df['date'].notna().any() else "—")

        if anchor_df is not None and not anchor_df.empty:
            st.markdown("**Anchor Customers** (Top 5 by Volume)")
            st.dataframe(
                anchor_df.style.format({'Total Qty': '{:,.0f}', 'Share %': '{:.1f}%'}),
                use_container_width=True, height=220
            )

        if predictions is not None and not predictions.empty:
            st.markdown("**Reorder Predictions**")
            def _conf_color(val):
                if val >= 70: return 'color: #2e7d32; font-weight:bold'
                elif val >= 40: return 'color: #e65100'
                return 'color: #c62828'
            try:
                st.dataframe(
                    predictions.style.applymap(_conf_color, subset=['Confidence %']),
                    use_container_width=True, height=260
                )
            except Exception:
                st.dataframe(predictions, use_container_width=True, height=260)
        else:
            st.info("Predictions need ≥2 order dates per customer.")

        # Products breakdown
        if 'product' in sales_df.columns:
            with st.expander("📋 Product Line Items"):
                prod_agg = (sales_df.groupby('product')['quantity']
                            .sum().reset_index()
                            .sort_values('quantity', ascending=False)
                            .rename(columns={'product': 'Product', 'quantity': 'Total Qty'}))
                st.dataframe(prod_agg.style.format({'Total Qty': '{:,.1f}'}),
                             use_container_width=True, height=280)
    else:
        st.info("Upload Sales Bills to see insights.")

# ── Column 2: Stock Status ────────────────────────────────────
with col_stock:
    st.markdown('<div class="panel-header">📦 Stock Position</div>', unsafe_allow_html=True)
    st.markdown("")

    if stock_sum is not None and not stock_sum.empty:
        low = stock_sum[stock_sum['Low Stock'] == True]
        st.metric("Materials Tracked", len(stock_sum))
        st.metric("Low Stock Alerts", len(low), delta=f"-{len(low)}" if len(low) > 0 else None,
                  delta_color="inverse")

        def _highlight_low(row):
            style = 'background-color: #fff3e0; color: #e65100' if row['Low Stock'] else ''
            return [style] * len(row)

        st.markdown("**Stock Movement Summary**")
        try:
            st.dataframe(
                stock_sum.drop(columns=['Low Stock'])
                .style.apply(_highlight_low, axis=1),
                use_container_width=True, height=360
            )
        except Exception:
            st.dataframe(stock_sum.drop(columns=['Low Stock'], errors='ignore'),
                         use_container_width=True, height=360)

        if len(low) > 0:
            st.warning(f"⚠ {len(low)} material(s) are below 20% of stored inward quantity.")
    else:
        st.info("Upload Stock Register to see stock position.")

    if purchase_df is not None and not purchase_df.empty:
        st.markdown("**Recent Purchases**")
        show_cols = [c for c in ['date', 'supplier', 'material', 'quantity', 'rate'] if c in purchase_df.columns]
        with st.expander("View Purchase Data"):
            st.dataframe(purchase_df[show_cols].rename(columns=str.title),
                         use_container_width=True, height=260)


# ── Column 3: Material Outlook ────────────────────────────────
with col_outlook:
    st.markdown('<div class="panel-header">🔮 Material Outlook</div>', unsafe_allow_html=True)
    st.markdown("")

    if outlook_df is not None and not outlook_df.empty:
        buy_soon = outlook_df[outlook_df['Advisory'] == 'Buy Soon']
        prepare  = outlook_df[outlook_df['Advisory'] == 'Prepare']
        monitor  = outlook_df[outlook_df['Advisory'] == 'Monitor']

        st.metric("Buy Soon 🔴", len(buy_soon))
        st.metric("Prepare 🟡", len(prepare))
        st.metric("Monitor 🟢", len(monitor))

        st.markdown("**Advisory by Material**")

        def _advisory_style(val):
            if val == 'Buy Soon': return 'color:#c62828; font-weight:bold'
            if val == 'Prepare':  return 'color:#e65100; font-weight:bold'
            return 'color:#2e7d32'

        try:
            st.dataframe(
                outlook_df.style.applymap(_advisory_style, subset=['Advisory'])
                .format({'Current Stock': '{:,.1f}', 'Est. Demand': '{:,.1f}'}),
                use_container_width=True, height=400
            )
        except Exception:
            st.dataframe(outlook_df, use_container_width=True, height=400)

        st.markdown("---")
        st.caption("**Advisory Logic:**\n- 🔴 Buy Soon = Est. demand > current stock\n- 🟡 Prepare = Demand approaching stock\n- 🟢 Monitor = Stock adequate")
    else:
        st.info("Upload Stock Register to see material advisory.")
