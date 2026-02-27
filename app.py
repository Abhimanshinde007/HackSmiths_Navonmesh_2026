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
    ingest_bom_excel,
    compute_material_requirements,
    load_data,
    save_data,
    clear_data,
    get_commodity_rates
)

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ERP Intelligence Engine",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────────────────────────
# STYLING
# ─────────────────────────────────────────────────────────────
try:
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except Exception:
    pass

# ─────────────────────────────────────────────────────────────
# SESSION STATE & PERSISTENCE
# ─────────────────────────────────────────────────────────────
# Attempt to load saved data first
loaded = load_data() if 'loaded_once' not in st.session_state else {}
st.session_state.loaded_once = True

for key in ['sales_df', 'purchase_df', 'stock_df',
            'anchor_df', 'predictions_df', 'stock_summary',
            'outlook_df', 'bom_df', 'requirements_df']:
    if key not in st.session_state:
        st.session_state[key] = loaded.get(key, None)

if 'processed' not in st.session_state:
    # If we successfully loaded sales or stock, flag as processed
    st.session_state.processed = (loaded.get('sales_df') is not None) or (loaded.get('stock_df') is not None)


# ─────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────
st.markdown("# 🏭 Enterprise Inventory Intelligence")
st.markdown("*Predictive procurement & stock analytics for MSMEs — ERP grade*")
st.markdown("---")

tab_dash, tab_bom, tab_commodity, tab_ingest = st.tabs([
    "📊 Executive Dashboard", 
    "🧾 BOM & Procurement", 
    "📈 Commodity Insights",
    "⚙️ Data Ingestion"
])

# ─────────────────────────────────────────────────────────────
# TAB 3: DATA INGESTION
# ─────────────────────────────────────────────────────────────
with tab_ingest:
    st.markdown('<div class="panel-header">📥 Enterprise Data Ingestion</div>', unsafe_allow_html=True)
    st.markdown("Upload your Excel records to continuously update the intelligence engine.")
    st.markdown("<br>", unsafe_allow_html=True)
    
    col_up1, col_up2, col_up3 = st.columns(3)
    
    with col_up1:
        st.markdown("### 📊 Sales Bills")
        st.caption("Upload Excel sales format invoices")
        sales_files = st.file_uploader("Sales Bills", type=["xlsx", "xls"], accept_multiple_files=True, key="sales_up")
        
        st.markdown("### 🛒 Purchase Bills")
        st.caption("Upload Excel purchase invoices")
        purchase_files = st.file_uploader("Purchase Bills", type=["xlsx", "xls"], accept_multiple_files=True, key="purch_up")
        
    with col_up2:
        st.markdown("### 📥 Inward Register")
        st.caption("Materials received from suppliers")
        inward_files = st.file_uploader("Inward Register", type=["xlsx", "xls"], accept_multiple_files=True, key="inward_up")
        
        st.markdown("### 📤 Outward Register")
        st.caption("Goods dispatched to customers")
        outward_files = st.file_uploader("Outward Register", type=["xlsx", "xls"], accept_multiple_files=True, key="outward_up")
        
    with col_up3:
        st.markdown("### 📋 Bill of Materials")
        st.caption("Product to raw material mapping")
        bom_file = st.file_uploader("BOM File", type=["xlsx", "xls"], key="bom_up")
        
        st.markdown("<br>", unsafe_allow_html=True)
        run_btn = st.button("▶ Process & Generate Insights", type="primary", use_container_width=True)
        st.markdown("<br>", unsafe_allow_html=True)
        
        if st.button("🔄 Reset All Data", help="Clear all stored data and refresh app", use_container_width=True):
            clear_data()
            for key in ['sales_df', 'purchase_df', 'stock_df', 'anchor_df', 'predictions_df', 'stock_summary', 'outlook_df', 'bom_df', 'requirements_df']:
                st.session_state[key] = None
            st.session_state.processed = False
            st.session_state.loaded_once = False
            st.rerun()

    if run_btn:
        if not sales_files and not purchase_files and not inward_files and not outward_files and not bom_file:
            st.warning("Please upload at least one file before processing.")
        else:
            with st.spinner("Processing files..."):
                st.markdown("---")
                
                # ── Sales ────────────────────────────────────────
                if sales_files:
                    df_s, errs_s = ingest_sales_excel(sales_files)
                    for e in errs_s: st.error(f"⚠ Sales Error: {e}")
                    if df_s is not None and not df_s.empty:
                        st.session_state.sales_df = df_s
                        anchor, aerr = get_anchor_customers(df_s)
                        if aerr: st.warning(f"Anchor: {aerr}")
                        else:
                            st.session_state.anchor_df = anchor
                            pred, perr = predict_reorder(df_s, anchor)
                            if not perr: st.session_state.predictions_df = pred
                        st.success(f"✅ Sales: {len(df_s):,} rows | {df_s['customer'].nunique()} customers")
                    else:
                        if not errs_s: st.warning("Sales: No valid rows extracted.")
                
                # ── Purchases ────────────────────────────────────
                if purchase_files:
                    df_p, errs_p = ingest_purchase_excel(purchase_files)
                    for e in errs_p: st.error(f"⚠ Purchase Error: {e}")
                    if df_p is not None and not df_p.empty:
                        st.session_state.purchase_df = df_p
                        st.success(f"✅ Purchase: {len(df_p):,} rows")
                
                # -- Inward / Outward --------------------------
                inward_df, outward_df = pd.DataFrame(), pd.DataFrame()
                if inward_files:
                    inward_df, errs_in = ingest_inward_excel(list(inward_files))
                    for e in errs_in: st.error(f"⚠ IN Error: {e}")
                    if not inward_df.empty:
                        st.success(f"✅ Inward: {len(inward_df):,} rows | {inward_df['material'].nunique()} materials")
                
                if outward_files:
                    outward_df, errs_out = ingest_outward_excel(list(outward_files))
                    for e in errs_out: st.error(f"⚠ OUT Error: {e}")
                    if not outward_df.empty:
                        st.success(f"✅ Outward: {len(outward_df):,} rows | {outward_df['material'].nunique()} materials")
                
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

                # -- BOM + Material Requirements --------------------
                if bom_file:
                    bom_df, bom_err = ingest_bom_excel(bom_file)
                    if bom_err: st.error(f"⚠ BOM Error: {bom_err}")
                    elif not bom_df.empty:
                        st.session_state.bom_df = bom_df
                        st.success(f"✅ BOM: {len(bom_df)} products loaded")
                        req_df, req_err = compute_material_requirements(
                            st.session_state.predictions_df,
                            bom_df,
                            st.session_state.stock_summary,
                        )
                        if not req_err:
                            st.session_state.requirements_df = req_df
                
                # Persist to local disk
                saved_count = save_data(st.session_state)
                if saved_count > 0:
                    st.success(f"💾 Securely cached {saved_count} datasets locally for persistence.")

                st.session_state.processed = True
            st.rerun()


# ─────────────────────────────────────────────────────────────
# TAB 1: EXECUTIVE DASHBOARD
# ─────────────────────────────────────────────────────────────
with tab_dash:
    if not st.session_state.processed:
        st.info("ℹ️ Your dashboard is empty. Please navigate to the **⚙️ Data Ingestion** tab to upload your records.")
        st.stop()

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
    pred_count       = len(predictions) if predictions is not None and not predictions.empty else 0

    st.markdown(f"""
    <div class="kpi-row">
      <div class="kpi-card" style="border-left: 4px solid #0B3D91;">
        <div class="kpi-label">Active Customers</div>
        <div class="kpi-value">{total_customers}</div>
      </div>
      <div class="kpi-card" style="border-left: 4px solid #F4B400;">
        <div class="kpi-label">Anchor Clients</div>
        <div class="kpi-value">{anchor_count}</div>
      </div>
      <div class="kpi-card" style="border-left: 4px solid #5A6772;">
        <div class="kpi-label">SKUs Tracked</div>
        <div class="kpi-value">{total_materials}</div>
      </div>
      <div class="kpi-card" style="border-left: 4px solid #2e7d32;">
        <div class="kpi-label">Expected Reorders</div>
        <div class="kpi-value">{pred_count}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)
    
    
    # -- DASH SECTION 1 --
    with st.container():
        st.markdown('<div class="panel-header">📊 Sales Intelligence</div>', unsafe_allow_html=True)
        if sales_df is not None and not sales_df.empty:
            if 'date' in sales_df.columns and 'quantity' in sales_df.columns:
                trend_df = sales_df.dropna(subset=['date', 'quantity'])
                if not trend_df.empty:
                    st.markdown("**Sales Volume Trend**")
                    daily = trend_df.groupby('date')['quantity'].sum().reset_index()
                    st.line_chart(daily.set_index('date'))

            if anchor_df is not None and not anchor_df.empty:
                st.markdown("**Core Revenue Drivers (Anchor Clients)**")
                st.dataframe(
                    anchor_df.assign(**{"SR. NO.": range(1, len(anchor_df)+1)}).set_index("SR. NO.")
                    .style.format({'Total Qty': '{:,.0f}', 'Share %': '{:.1f}%'}),
                    use_container_width=True, height=220
                )
            if predictions is not None and not predictions.empty:
                st.markdown("**Predicted Reorder Pipeline**")
                def _conf_color(val):
                    if val >= 70: return 'color: #2e7d32; font-weight:bold'
                    elif val >= 40: return 'color: #e65100'
                    return 'color: #c62828'
                st.dataframe(
                    predictions.assign(**{"SR. NO.": range(1, len(predictions)+1)}).set_index("SR. NO.")
                    .style.applymap(_conf_color, subset=['Confidence %']),
                    use_container_width=True, height=260
                )
            else:
                st.info("Predictions need ≥2 order dates per customer.")
        else:
            st.info("Awaiting Sales Data.")

    # -- DASH SECTION 2 --
    with st.container():
        st.markdown('<div class="panel-header">📦 Stock Position</div>', unsafe_allow_html=True)
        if stock_sum is not None and not stock_sum.empty:
            low = stock_sum[stock_sum['Low Stock'] == True]
            if len(low) > 0:
                st.markdown(f'<div class="warning-box">⚠ <b>{len(low)} material(s)</b> are critically low (<20% of inward).</div>', unsafe_allow_html=True)
            
            st.markdown("**Raw Material Stock Movement**")
            def _highlight_low(row):
                style = 'background-color: #fdf5f6; color: #842029' if row['Low Stock'] else ''
                return [style] * len(row)
            try:
                st.dataframe(
                    stock_sum.drop(columns=['Low Stock'])
                    .assign(**{"SR. NO.": range(1, len(stock_sum)+1)}).set_index("SR. NO.")
                    .style.apply(_highlight_low, axis=1),
                    use_container_width=True, height=360
                )
            except Exception:
                st.dataframe(stock_sum, use_container_width=True, height=360)
        else:
            st.info("Awaiting Stock Register Data.")

    # -- DASH SECTION 3 --
    with st.container():
        st.markdown('<div class="panel-header">🔮 Ad-Hoc Supply Outlook</div>', unsafe_allow_html=True)
        if outlook_df is not None and not outlook_df.empty:
            st.markdown("**Material Depletion Advisory**")
            def _advisory_style(val):
                if val == 'Buy Soon': return 'background-color: #f8d7da; color:#842029; font-weight:bold'
                if val == 'Prepare':  return 'background-color: #fff3cd; color:#664d03; font-weight:bold'
                return 'background-color: #d1e6dd; color:#0f5132'
            st.dataframe(
                outlook_df.assign(**{"SR. NO.": range(1, len(outlook_df)+1)}).set_index("SR. NO.")
                .style.applymap(_advisory_style, subset=['Advisory'])
                .format({'Current Stock': '{:,.1f}', 'Est. Demand': '{:,.1f}'}),
                use_container_width=True, height=400
            )
        else:
            st.info("Awaiting combined stock and sales data.")


# ─────────────────────────────────────────────────────────────
# TAB 2: BOM & PROCUREMENT
# ─────────────────────────────────────────────────────────────
with tab_bom:
    requirements_df = st.session_state.get('requirements_df')
    bom_df          = st.session_state.get('bom_df')

    st.markdown('<div class="panel-header">🧾 BOM-Driven Procurement Engine</div>', unsafe_allow_html=True)
    st.markdown("Automated raw material demand planning driven by predicted customer orders.")

    if requirements_df is not None and not requirements_df.empty:
        col_kpi1, col_kpi2 = st.columns(2)
        buy_now_count = len(requirements_df[requirements_df['Status'] == 'BUY NOW'])
        prepare_count = len(requirements_df[requirements_df['Status'] == 'PREPARE'])

        with col_kpi1:
            if buy_now_count > 0:
                st.markdown(f'<div class="warning-box" style="background-color:#f8d7da; border-color:#f5c2c7; color:#842029; padding:16px;">🚨 <b>ACTION REQUIRED:</b> {buy_now_count} materials must be procured immediately to meet predicted orders.</div>', unsafe_allow_html=True)
            elif prepare_count > 0:
                st.markdown(f'<div class="warning-box" style="padding:16px;">⚠️ <b>HEADS UP:</b> {prepare_count} materials are approaching low-stock against predicted orders.</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="warning-box" style="background-color:#d1e6dd; border-color:#badbcc; color:#0f5132; padding:16px;">✅ <b>STOCK HEALTHY:</b> All materials sufficient for predicted upcoming orders.</div>', unsafe_allow_html=True)

        def _req_style(val):
            if val == 'BUY NOW':   return 'background-color:#FF3B30; color:white; font-weight:bold'
            if val == 'PREPARE':   return 'background-color:#FF9500; color:white; font-weight:bold'
            if val == 'CHECK STOCK': return 'background-color:#FFD60A; color:black; font-weight:bold'
            if val == 'OK':        return 'background-color:#34C759; color:white; font-weight:bold'
            return ''

        try:
            st.dataframe(
                requirements_df.assign(**{"SR. NO.": range(1, len(requirements_df)+1)}).set_index("SR. NO.")
                .style.applymap(_req_style, subset=['Status']),
                use_container_width=True,
                height=500
            )
        except Exception:
            st.dataframe(requirements_df, use_container_width=True, height=500)

    elif bom_df is not None and not bom_df.empty:
        st.info("BOM loaded. Upload Sales Bills to generate reorder predictions and compute material requirements.")
    else:
        st.info("Upload your **BOM file** in the Data Ingestion tab to activate Procurement Alerts.")

# ─────────────────────────────────────────────────────────────
# TAB 3: COMMODITY INSIGHTS
# ─────────────────────────────────────────────────────────────
with tab_commodity:
    st.markdown('<div class="panel-header">📈 Commodity Market Intelligence</div>', unsafe_allow_html=True)
    st.markdown("Live 90-day futures tracking and 30-day algorithmic forecasts for raw materials. (Source: Yahoo Finance)")
    
    with st.spinner("Fetching live market futures..."):
        comm_data, comm_err = get_commodity_rates()
    
    if comm_err:
        st.error(comm_err)
    elif comm_data:
        c1, c2 = st.columns(2)
        cols = [c1, c2]
        for idx, (commodity, data) in enumerate(comm_data.items()):
            with cols[idx]:
                st.markdown(f"### {commodity} (INR / KG)")
                
                st.metric("Current Rate", 
                          f"₹{data['current_price']:.2f}",
                          delta=data['trend'], 
                          delta_color="normal" if "UP" in data['trend'] else "inverse" if "DOWN" in data['trend'] else "off")
                
                st.caption(f"30-Day High: ₹{data['high_30d']:.2f} | 30-Day Low: ₹{data['low_30d']:.2f}")
                
                 # Line chart historical
                st.markdown("**Historical Price (90 Days)**")
                hist_df = data['history']
                st.line_chart(hist_df.set_index('Date')['Price_INR_KG'])
                
                st.markdown("**Algorithmic Forecast (Next 30 Days)**")
                st.line_chart(data['forecast'])
