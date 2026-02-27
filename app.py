"""
Main Streamlit Application
Predictive Inventory & Procurement Intelligence Platform for Manufacturing MSMEs
"""

import streamlit as st
import pandas as pd
import numpy as np
import os

# --- Custom Modules ---
from utils.data_processor import (
    clean_and_standardize_sales,
    rank_anchor_customers,
    predict_reorder_intervals,
    process_stock_movement,
    map_bom_to_raw_materials
)

# --- Configuration & Styling ---
st.set_page_config(
    page_title="Procurement Intelligence",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

def load_css(file_name):
    """Loads external CSS to inject custom styling."""
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

try:
    load_css("style.css")
except FileNotFoundError:
    pass # CSS file might be created later

# --- Session State Initialization ---
if "sales_df" not in st.session_state:
    st.session_state.sales_df = None
if "stock_df" not in st.session_state:
    st.session_state.stock_df = None
if "bom_df" not in st.session_state:
    st.session_state.bom_df = None

# --- Sidebar UI Flow ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3135/3135673.png", width=100) # Placeholder industrial icon
    st.title("Data Ingestion")
    
    st.header("1. Upload Sales Data")
    sales_file = st.file_uploader("Upload Sales Register (Excel)", type=["xlsx", "xls"], key="sales")
    
    st.header("2. Upload Stock Register")
    stock_file = st.file_uploader("Upload Stock Movements (Excel)", type=["xlsx", "xls"], key="stock")
    
    st.header("3. Upload BOM (Optional)")
    bom_file = st.file_uploader("Upload Bill of Materials (Excel)", type=["xlsx", "xls"], key="bom")
    
    if st.button("Process Data & Run Insights"):
        if sales_file or stock_file:
            
            # Process Sales Independently
            if sales_file:
                try:
                    raw_sales = pd.read_excel(sales_file)
                    st.session_state.sales_df = clean_and_standardize_sales(raw_sales)
                except Exception as e:
                    st.error(f"Failed to process Sales Data: {e}")
                    st.session_state.sales_df = None
            
            # Process Stock Independently
            if stock_file:
                try:
                    raw_stock = pd.read_excel(stock_file)
                    st.session_state.stock_df = process_stock_movement(raw_stock)
                except Exception as e:
                    st.error(f"Failed to process Stock Data: {e}")
                    st.session_state.stock_df = None
                    
            # Process BOM Independently
            if bom_file:
                try:
                    st.session_state.bom_df = pd.read_excel(bom_file)
                except Exception as e:
                    st.error(f"Failed to process BOM Data: {e}")
                    st.session_state.bom_df = None
                    
            st.session_state.data_processed = True
            st.success("Data loaded gracefully!")
        else:
            st.error("Please upload at least Sales or Stock data.")

# --- Main Dashboard Setup ---
st.title("🏭 Predictive Inventory & Procurement Dashboard")
st.markdown("---")

if st.session_state.get("data_processed"):
    # Analytics Variables
    total_skus = '--'
    top_customer_window = '--'
    avg_reorder = '-- days'

    anchor_customers = pd.DataFrame()
    predictions = pd.DataFrame()
    stock_movement = pd.DataFrame()

    if st.session_state.sales_df is not None and not st.session_state.sales_df.empty:
        anchor_customers = rank_anchor_customers(st.session_state.sales_df)
        if not anchor_customers.empty:
             predictions = predict_reorder_intervals(st.session_state.sales_df, anchor_customers)
             if not predictions.empty:
                  avg_reorder = str(round(predictions['Avg Interval (Days)'].mean(), 1)) + " days"
                  top_customer_window = predictions.iloc[0]['Predicted Next Order']
            
    if st.session_state.stock_df is not None and not st.session_state.stock_df.empty:
        stock_movement = st.session_state.stock_df
        if 'sku' in stock_movement.columns:
            total_skus = str(stock_movement['sku'].nunique())

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Active SKUs", total_skus)
    col2.metric("Top Client Next Reorder", top_customer_window)
    col3.metric("Stockout Risk Items", "See Tiers")
    col4.metric("Avg Reorder Interval", avg_reorder)
    
    st.markdown("---")
    
    # Visualizations & Detail Windows
    tab1, tab2, tab3 = st.tabs(["📊 Demand Predictions", "📦 Stock Velocity", "⚙️ BOM Intelligence"])
    
    with tab1:
        st.subheader("Anchor Customer Reorder Probabilities")
        if not predictions.empty:
            st.dataframe(predictions, use_container_width=True)
            
            # Simple chart of Anchor customers
            if not anchor_customers.empty:
                chart_df = anchor_customers.head(10).set_index('customer')['revenue']
                st.bar_chart(chart_df)
        else:
            st.info("Upload standard sales data with dates, customer names, and revenue to calculate predictions.")
            
    with tab2:
        st.subheader("SKU Movement & Velocity Tiers")
        if not stock_movement.empty:
            st.dataframe(stock_movement, use_container_width=True)
            if 'category' in stock_movement.columns:
                stock_counts = stock_movement['category'].value_counts()
                st.bar_chart(stock_counts)
        else:
            st.info("Upload stock register to see movement velocity.")
            
    with tab3:
        st.subheader("Raw Material Forecasting")
        if st.session_state.bom_df is not None and not st.session_state.bom_df.empty and not predictions.empty:
           bom_mapped = map_bom_to_raw_materials(st.session_state.bom_df, predictions)
           st.dataframe(bom_mapped, use_container_width=True)
           st.success("Successfully parsed expected demand against Bill of Materials.")
        else:
           st.info("Upload both Sales History and a BOM to map expected raw material needs.")

else:
    st.markdown("""
        ### Welcome to the MSME Procurement Intelligence Platform
        
        **To get started:**
        1. Upload your unstructured Sales Excel register via the sidebar.
        2. Upload your Stock tracking register.
        3. (Optional) Provide your Bill of Materials (BOM) for raw material forecasting.
        
        *This system will aggregate the data, clean it, and predict anchor customer demand without relying on heavy ERP structures.*
    """)
