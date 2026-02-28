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
    get_commodity_rates,
    load_company,
    save_company
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
# STYLING & THEME TOGGLE LOGIC
# ─────────────────────────────────────────────────────────────
import os
CONFIG_FILE = ".streamlit/config.toml"

def get_current_theme():
    if not os.path.exists(CONFIG_FILE):
        return "light"
    with open(CONFIG_FILE, "r") as f:
        content = f.read()
        if 'base="dark"' in content or "base = 'dark'" in content:
            return "dark"
    return "light"

current_theme = get_current_theme()
css_file = "style_dark.css" if current_theme == "dark" else "style_light.css"

try:
    with open(css_file) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except Exception:
    pass

# ─────────────────────────────────────────────────────────────
# SESSION STATE & PERSISTENCE
# ─────────────────────────────────────────────────────────────
# Load from disk to ensure data survives a page reload (F5)
loaded = load_data() if 'loaded_once' not in st.session_state else {}
st.session_state.loaded_once = True

for key in ['sales_df', 'purchase_df', 'stock_df',
            'anchor_df', 'predictions_df', 'stock_summary',
            'outlook_df', 'bom_df', 'requirements_df', 'process_logs']:
    if key not in st.session_state:
        st.session_state[key] = loaded.get(key, None)

if 'processed' not in st.session_state:
    st.session_state.processed = (loaded.get('sales_df') is not None) or (loaded.get('stock_df') is not None)


# ─────────────────────────────────────────────────────────────
# HEADER & THEME TOGGLE
# ─────────────────────────────────────────────────────────────

def toggle_theme():
    new_base = "dark" if current_theme == "light" else "light"
    new_primary = "#6366f1" if new_base == "dark" else "#0B3D91"
    new_bg = "#0b1121" if new_base == "dark" else "#f4f6f9"
    new_sec_bg = "#1e293b" if new_base == "dark" else "#ffffff"
    new_text = "#e2e8f0" if new_base == "dark" else "#333333"

    new_content = f"""[theme]
base="{new_base}"
primaryColor = "{new_primary}"
backgroundColor = "{new_bg}"
secondaryBackgroundColor = "{new_sec_bg}"
textColor = "{new_text}"
font = "sans serif"

[client]
toolbarMode = "minimal"
"""
    os.makedirs(".streamlit", exist_ok=True)
    with open(CONFIG_FILE, "w") as f:
        f.write(new_content)
    
    # Force streamlit to recognize the new config immediately via internal API
    try:
        st._config.set_option('theme.base', new_base)
        st._config.set_option('theme.primaryColor', new_primary)
        st._config.set_option('theme.backgroundColor', new_bg)
        st._config.set_option('theme.secondaryBackgroundColor', new_sec_bg)
        st._config.set_option('theme.textColor', new_text)
    except Exception:
        pass
        
    st.rerun()

col_title, col_btn1, col_btn2 = st.columns([0.75, 0.12, 0.13])

# ─────────────────────────────────────────────────────────────
# COMPANY ONBOARDING & HEADER
# ─────────────────────────────────────────────────────────────
company_name = load_company()
if company_name:
    st.session_state.company_name = company_name

if not st.session_state.get('company_name'):
    st.markdown("<br><br><h1 style='text-align: center;'>Welcome to Smart Factory Operations</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #64748b;'>Please configure your workspace to continue.</p><br>", unsafe_allow_html=True)
    
    col_a, col_b, col_c = st.columns([1,2,1])
    with col_b:
        with st.form("onboarding_form"):
            c_name = st.text_input("Enter Company Name", placeholder="e.g. Acme Industries Ltd.")
            submitted = st.form_submit_button("Launch Dashboard", use_container_width=True)
            if submitted and c_name.strip():
                save_company(c_name.strip())
                st.session_state.company_name = c_name.strip()
                st.rerun()
    st.stop()

with col_title:
    st.markdown(f"# 🏭 {st.session_state.company_name} | Smart Factory Operations")
    st.markdown("*Precision Inventory & Material Tracking System*")
with col_btn1:
    st.markdown("<br>", unsafe_allow_html=True)
    if current_theme == "light":
        if st.button("🌙 Dark Mode", use_container_width=True):
            toggle_theme()
    else:
        if st.button("☀️ Light Mode", use_container_width=True):
            toggle_theme()
            
with col_btn2:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔄 Refresh Data", help="Reload data from cache and refresh views", use_container_width=True):
        st.session_state.loaded_once = False
        st.rerun()
st.markdown("---")

# ─────────────────────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ─────────────────────────────────────────────────────────────
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = "Executive Dashboard" if st.session_state.processed else "Data Ingestion"

st.sidebar.markdown(f"**{st.session_state.company_name}**")
st.sidebar.markdown("---")
selected_tab = st.sidebar.radio("Navigation", [
    "Data Ingestion",
    "Executive Dashboard", 
    "Material Requirements", 
    "Commodity Insights"
], index=["Data Ingestion", "Executive Dashboard", "Material Requirements", "Commodity Insights"].index(st.session_state.active_tab))

if selected_tab != st.session_state.active_tab:
    st.session_state.active_tab = selected_tab
    st.rerun()

# ─────────────────────────────────────────────────────────────
# TAB: DATA INGESTION
# ─────────────────────────────────────────────────────────────
if st.session_state.active_tab == "Data Ingestion":
    st.markdown('<div class="flex-header"><h2>📥 Enterprise Data Ingestion</h2></div>', unsafe_allow_html=True)
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
        st.caption("*Required Columns: Product Name, Copper Weight, Lamination Weight*")
        bom_file = st.file_uploader("BOM File", type=["xlsx", "xls"], key="bom_up")
        
        st.markdown("<br>", unsafe_allow_html=True)
        run_btn = st.button("▶ Process & Generate Insights", type="primary", use_container_width=True)
        st.markdown("<br>", unsafe_allow_html=True)
        
        
        if st.button("🔄 Reset All Data", help="Clear all stored data and refresh app", use_container_width=True):
            with st.spinner("Purging database & resetting workspace..."):
                import time
                time.sleep(1.5) # Simulate database purge for visual feedback
                clear_data()
                for key in ['sales_df', 'purchase_df', 'stock_df', 'anchor_df', 'predictions_df', 'stock_summary', 'outlook_df', 'bom_df', 'requirements_df', 'process_logs']:
                    st.session_state[key] = None
                st.session_state.processed = False
                st.session_state.loaded_once = False
                st.session_state.company_name = None
            st.rerun()
            
    # -- Display Persisted Logs --
    if st.session_state.get('process_logs'):
        st.markdown("### 📝 Last Processing Logs")
        for log_type, log_msg in st.session_state.process_logs:
            if log_type == 'error':
                st.error(log_msg)
            elif log_type == 'warning':
                st.warning(log_msg)
            else:
                st.success(log_msg)
        st.markdown("---")

    if run_btn:
        if not sales_files and not purchase_files and not inward_files and not outward_files and not bom_file:
            st.warning("Please upload at least one file before processing.")
        else:
            with st.spinner("Processing files..."):
                st.markdown("---")
                new_logs = []
                
                # ── Sales ────────────────────────────────────────
                if sales_files:
                    df_s, errs_s = ingest_sales_excel(sales_files)
                    for e in errs_s: new_logs.append(('error', f"⚠ Sales Error: {e}"))
                    if df_s is not None and not df_s.empty:
                        st.session_state.sales_df = df_s
                        anchor, aerr = get_anchor_customers(df_s)
                        if aerr: new_logs.append(('warning', f"Anchor: {aerr}"))
                        else:
                            st.session_state.anchor_df = anchor
                            pred, perr = predict_reorder(df_s, anchor)
                            if not perr: st.session_state.predictions_df = pred
                        new_logs.append(('success', f"✅ Sales: {len(df_s):,} rows | {df_s['customer'].nunique()} customers"))
                    else:
                        if not errs_s: new_logs.append(('warning', "Sales: No valid rows extracted."))
                
                # ── Purchases ────────────────────────────────────
                if purchase_files:
                    df_p, errs_p = ingest_purchase_excel(purchase_files)
                    for e in errs_p: new_logs.append(('error', f"⚠ Purchase Error: {e}"))
                    if df_p is not None and not df_p.empty:
                        st.session_state.purchase_df = df_p
                        new_logs.append(('success', f"✅ Purchase: {len(df_p):,} rows"))
                
                # -- Inward / Outward --------------------------
                inward_df, outward_df = pd.DataFrame(), pd.DataFrame()
                if inward_files:
                    inward_df, errs_in = ingest_inward_excel(list(inward_files))
                    for e in errs_in: new_logs.append(('error', f"⚠ IN Error: {e}"))
                    if not inward_df.empty:
                        new_logs.append(('success', f"✅ Inward: {len(inward_df):,} rows | {inward_df['material'].nunique()} materials"))
                
                if outward_files:
                    outward_df, errs_out = ingest_outward_excel(list(outward_files))
                    for e in errs_out: new_logs.append(('error', f"⚠ OUT Error: {e}"))
                    if not outward_df.empty:
                        new_logs.append(('success', f"✅ Outward: {len(outward_df):,} rows | {outward_df['material'].nunique()} materials"))
                
                if not inward_df.empty or not outward_df.empty:
                    combined = combine_stock_registers(inward_df, outward_df)
                    st.session_state.stock_df = combined
                    stk, serr = compute_stock(combined)
                    if not serr:
                        st.session_state.stock_summary = stk
                        out_df, locked_cap, oerr = material_outlook(
                            combined,
                            st.session_state.predictions_df,
                            st.session_state.sales_df,
                            st.session_state.get('purchase_df', None)
                        )
                        if not oerr:
                            st.session_state.outlook_df = out_df
                            st.session_state.locked_capital = locked_cap

                # -- BOM + Material Requirements --------------------
                if bom_file:
                    bom_df, bom_err = ingest_bom_excel(bom_file)
                    
                    if bom_err and "Product Name" in bom_err:
                        # Fatal error (no product names)
                        new_logs.append(('error', f"⚠ BOM Error: {bom_err}"))
                    else:
                        if bom_err:
                            # Non-fatal warning (e.g. missing weights)
                            new_logs.append(('warning', f"⚠ BOM Notice: {bom_err}"))
                        
                        if not bom_df.empty:
                            st.session_state.bom_df = bom_df
                            new_logs.append(('success', f"✅ BOM: {len(bom_df)} products loaded"))
                            req_df, req_err = compute_material_requirements(
                                st.session_state.predictions_df,
                                bom_df,
                                st.session_state.stock_summary,
                            )
                        if not req_err:
                            st.session_state.requirements_df = req_df
                
                # Persist to local disk cache for internal fast compute (but wiped on reload)
                save_data(st.session_state)

                st.session_state.process_logs = new_logs
                st.session_state.processed = True
                st.session_state.active_tab = "Executive Dashboard"
            st.rerun()


# ─────────────────────────────────────────────────────────────
# TAB 1: EXECUTIVE DASHBOARD
# ─────────────────────────────────────────────────────────────
elif st.session_state.active_tab == "Executive Dashboard":
    if not st.session_state.processed:
        st.stop()

    # -- DASH SECTION 4: SMART REQUIREMENT ENGINE (MOVED TO TOP) --
    st.markdown('<div class="flex-header"><h2>🧾 Smart Material Planning</h2></div>', unsafe_allow_html=True)
    st.markdown("Automated raw material demand planning based on upcoming customer orders.")
    
    requirements_df = st.session_state.get('requirements_df')
    bom_df          = st.session_state.get('bom_df')

    if requirements_df is not None and not requirements_df.empty:
        col_kpi1, col_kpi2 = st.columns(2)
        buy_now_count = len(requirements_df[requirements_df['Status'] == 'BUY NOW'])
        prepare_count = len(requirements_df[requirements_df['Status'] == 'PREPARE'])

        with col_kpi1:
            if buy_now_count > 0:
                st.markdown(f'<div class="warning-box status-crit">🚨 <b>ACTION REQUIRED:</b> {buy_now_count} materials must be procured immediately to meet predicted orders.</div>', unsafe_allow_html=True)
            elif prepare_count > 0:
                st.markdown(f'<div class="warning-box status-warn">⚠️ <b>HEADS UP:</b> {prepare_count} materials are approaching low-stock against predicted orders.</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="warning-box status-ok">✅ <b>STOCK HEALTHY:</b> All materials sufficient for predicted upcoming orders.</div>', unsafe_allow_html=True)

        def _req_style(val):
            if val == 'BUY NOW':   return 'background-color: #fca5a5; color:#fff; font-weight:700'
            if val == 'PREPARE':   return 'background-color: #fde68a; color:#b45309; font-weight:700'
            if val == 'CHECK STOCK': return 'background-color: #bfdbfe; color:#1d4ed8; font-weight:700'
            if val == 'OK':        return 'background-color: #bbf7d0; color:#15803d; font-weight:700'
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
        st.info("Upload Sales Bills to generate reorder predictions and compute material requirements.")
    else:
        st.info("Upload your **BOM file** in the Data Ingestion tab to activate Procurement Alerts.")
        
    st.markdown("---")

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
    locked_cap_val   = st.session_state.get('locked_capital', 0.0)

    # Replace raw HTML with native UI columns for proper margins
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Active Customers", total_customers)
    k2.metric("Anchor Clients", anchor_count)
    k3.metric("SKUs Tracked", total_materials)
    k4.metric("Expected Reorders", pred_count)
    if locked_cap_val > 0:
        k5.markdown(f'<div data-testid="stMetric" style="border-left: 4px solid #F57C00;"><div data-testid="stMetricLabel">Working Capital Locked</div><div data-testid="stMetricValue">₹{locked_cap_val:,.0f}</div></div>', unsafe_allow_html=True)
    else:
        k5.metric("Working Capital Locked", f"₹{locked_cap_val:,.0f}")
        
    st.markdown("---")
    
    if locked_cap_val > 0:
        with st.expander("View Excess Inventory Breakdown (Working Capital Lock)"):
            if outlook_df is not None and not outlook_df.empty:
                excess_df = outlook_df[outlook_df['locked_capital'] > 0]
                if not excess_df.empty:
                    st.dataframe(
                        excess_df[['Material', 'excess_stock', 'locked_capital']]
                        .sort_values('locked_capital', ascending=False)
                        .assign(**{"SR. NO.": range(1, len(excess_df)+1)}).set_index("SR. NO.")
                        .style.format({'locked_capital': '₹{:,.2f}'}),
                        use_container_width=True
                    )
    
    
    # -- DASH SECTION 1 --
    with st.container():
        st.markdown('<div class="flex-header"><h2>📊 Sales Intelligence</h2></div>', unsafe_allow_html=True)
        if sales_df is not None and not sales_df.empty:
            if 'date' in sales_df.columns and 'quantity' in sales_df.columns:
                trend_df = sales_df.dropna(subset=['date', 'quantity'])
                if not trend_df.empty:
                    st.markdown("**Sales Volume Trend**")
                    daily = trend_df.groupby('date')['quantity'].sum().reset_index()
                    import altair as alt
                    chart = alt.Chart(daily).mark_line(
                        point=alt.OverlayMarkDef(filled=False, fill="white", color="#0B3D91", size=60),
                        color="#0B3D91",
                        strokeWidth=2
                    ).encode(
                        x=alt.X('date:T', title='', axis=alt.Axis(grid=False, format='%b %d')),
                        y=alt.Y('quantity:Q', title='Volume', axis=alt.Axis(gridColor='#f1f5f9')),
                        tooltip=[alt.Tooltip('date:T', title='Date'), alt.Tooltip('quantity:Q', title='Volume sold')]
                    ).properties(height=250)
                    st.altair_chart(chart, use_container_width=True)

            if anchor_df is not None and not anchor_df.empty:
                st.markdown("**Core Revenue Drivers (Anchor Clients)**")
                st.dataframe(
                    anchor_df.assign(**{"SR. NO.": range(1, len(anchor_df)+1)}).set_index("SR. NO.")
                    .style.format({'Total Qty': '{:,.0f}', 'Share %': '{:.1f}%'}),
                    use_container_width=True, height=220
                )
            if predictions is not None and not predictions.empty:
                st.markdown("**Predicted Reorder Pipeline & Volatility**")
                if 'volatility_label' not in predictions.columns:
                    predictions['volatility_label'] = 'Pending Order Data'

                def _conf_color(val):
                    if pd.isna(val) or type(val) == str: return ''
                    if val >= 70: return 'background-color: #bbf7d0; color: #166534; font-weight:700' # light emerald
                    elif val >= 40: return 'background-color: #fef08a; color: #854d0e; font-weight:700' # light yellow
                    return 'background-color: #fecaca; color: #991b1b; font-weight:700' # light red
                    
                def _vol_color(val):
                    if val == "Stable": return 'color: #166534; font-weight:700'
                    elif val == "Moderate": return 'color: #854d0e; font-weight:700'
                    elif val == "Unstable": return 'color: #991b1b; font-weight:700'
                    return 'color: #9ca3af'

                st.dataframe(
                    predictions.assign(**{"SR. NO.": range(1, len(predictions)+1)}).set_index("SR. NO.")
                    .style.applymap(_conf_color, subset=['Confidence %'])
                    .applymap(_vol_color, subset=['volatility_label']),
                    use_container_width=True, height=260
                )
            else:
                st.info("Predictions need ≥2 order dates per customer.")
        else:
            st.info("Awaiting Sales Data.")

    # -- DASH SECTION 2 --
    with st.container():
        st.markdown('<div class="flex-header"><h2>📦 Stock Position</h2></div>', unsafe_allow_html=True)
        if stock_sum is not None and not stock_sum.empty:
            low = stock_sum[stock_sum['Low Stock'] == True]
            if len(low) > 0:
                st.markdown(f'<div class="warning-box">⚠ <b>{len(low)} material(s)</b> are critically low (<20% of inward).</div>', unsafe_allow_html=True)
            
            st.markdown("**Raw Material Stock Movement & Coverage**")
            if 'coverage_label' not in stock_sum.columns:
                stock_sum['coverage_label'] = 'No Recent Movement'

            def _highlight_low(row):
                style = 'background-color: #fca5a5; color: #991b1b; font-weight:700' if row['Low Stock'] else ''
                return [style] * len(row)
                
            def _cov_color(val):
                if val == "Critical": return 'background-color: #fca5a5; color: #991b1b; font-weight:700'
                elif val == "Moderate": return 'background-color: #fde68a; color: #92400e; font-weight:700'
                elif val == "Healthy": return 'background-color: #bbf7d0; color: #166534; font-weight:700'
                return ''

            try:
                disp_stock = stock_sum.drop(columns=['Low Stock']).assign(**{"SR. NO.": range(1, len(stock_sum)+1)}).set_index("SR. NO.")
                st.dataframe(
                    disp_stock.style.apply(_highlight_low, axis=1)
                    .applymap(_cov_color, subset=['coverage_label']),
                    use_container_width=True, height=360
                )
            except Exception:
                st.dataframe(stock_sum, use_container_width=True, height=360)
        else:
            st.info("Awaiting Stock Register Data.")

    # -- DASH SECTION 3 --
    with st.container():
        st.markdown('<div class="flex-header"><h2>🔮 Ad-Hoc Supply Outlook</h2></div>', unsafe_allow_html=True)
        if outlook_df is not None and not outlook_df.empty:
            st.markdown("**Material Depletion Advisory**")
            def _advisory_style(val):
                if val == 'Buy Soon': return 'background-color: #fca5a5; color: #991b1b; font-weight:700'
                if val == 'Prepare':  return 'background-color: #fde68a; color: #92400e; font-weight:700'
                return 'background-color: #bbf7d0; color: #166534'
            st.dataframe(
                outlook_df.assign(**{"SR. NO.": range(1, len(outlook_df)+1)}).set_index("SR. NO.")
                .style.applymap(_advisory_style, subset=['Advisory'])
                .format({'Current Stock': '{:,.1f}', 'Est. Demand': '{:,.1f}'}),
                use_container_width=True, height=400
            )
        else:
            st.info("Awaiting combined stock and sales data.")



# ─────────────────────────────────────────────────────────────
# TAB 2: BOM RAW MAPPING
# ─────────────────────────────────────────────────────────────
elif st.session_state.active_tab == "Material Requirements":
    st.markdown('<div class="flex-header"><h2>📋 Uploaded Bill of Materials</h2></div>', unsafe_allow_html=True)
    
    _bom = st.session_state.get('bom_df')
    if _bom is not None and not _bom.empty:
        st.dataframe(
            _bom.assign(**{"SR. NO.": range(1, len(_bom)+1)}).set_index("SR. NO."),
            use_container_width=True, height=600
        )
    else:
        st.info("Upload your **BOM file** in the Data Ingestion tab to view your raw materials mapping here.")

# ─────────────────────────────────────────────────────────────
# TAB 3: COMMODITY INSIGHTS
# ─────────────────────────────────────────────────────────────
elif st.session_state.active_tab == "Commodity Insights":
    st.markdown('<div class="flex-header"><h2>📈 Commodity Market Intelligence</h2></div>', unsafe_allow_html=True)
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
