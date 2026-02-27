import pandas as pd
import numpy as np

def clean_and_standardize_sales(df):
    """
    Takes an unstructured raw excel dataframe and attempts to standardize it
    for the hackathon MVP.
    """
    # 1. Drop completely empty rows or columns
    df = df.dropna(how='all', axis=0).dropna(how='all', axis=1)
    
    # Optional: If the first few rows are garbage headers, heuristic to find the real header
    # For MVP, assuming the uploaded file has headers on the first row or
    # user selects it, but we'll lowercase and strip the columns for safety.
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    
    # Attempt to find standard columns by fuzzy matching
    col_mapping = {}
    for col in df.columns:
        if any(kw in col for kw in ['date', 'time', 'dt']):
            col_mapping[col] = 'date'
        elif any(kw in col for kw in ['customer', 'client', 'buyer', 'party']):
            col_mapping[col] = 'customer'
        elif any(kw in col for kw in ['item', 'product', 'sku', 'goods']):
            col_mapping[col] = 'sku'
        elif any(kw in col for kw in ['qty', 'quantity', 'volume', 'units']):
            col_mapping[col] = 'quantity'
        elif any(kw in col for kw in ['amount', 'price', 'total', 'revenue', 'value']):
            col_mapping[col] = 'revenue'
            
    df = df.rename(columns=col_mapping)
    
    # Ensure Date format if 'date' column exists
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
    return df

def rank_anchor_customers(df):
    """
    Identifies top revenue-driving customers (Anchor Customers).
    """
    if 'customer' not in df.columns or 'revenue' not in df.columns:
        return pd.DataFrame() # Return empty if columns aren't found
        
    # Group by customer and sum revenue
    customer_revenue = df.groupby('customer')['revenue'].sum().reset_index()
    customer_revenue = customer_revenue.sort_values(by='revenue', ascending=False)
    
    # Calculate percentage contribution
    total_rev = customer_revenue['revenue'].sum()
    customer_revenue['contribution_pct'] = (customer_revenue['revenue'] / total_rev) * 100
    
    return customer_revenue

def predict_reorder_intervals(df, anchor_customers_df, top_n=5):
    """
    Calculates average reorder interval for top customers and predicts the next window.
    """
    if 'date' not in df.columns or 'customer' not in df.columns:
        return pd.DataFrame()
        
    top_customers = anchor_customers_df.head(top_n)['customer'].tolist()
    
    predictions = []
    
    for customer in top_customers:
        cust_sales = df[df['customer'] == customer].sort_values(by='date')
        
        if len(cust_sales) > 1:
            # Calculate days between orders
            cust_sales['days_since_last'] = cust_sales['date'].diff().dt.days
            avg_interval = cust_sales['days_since_last'].mean()
            std_interval = cust_sales['days_since_last'].std()
            
            last_order_date = cust_sales['date'].max()
            
            # Predict next window
            next_expected = last_order_date + pd.Timedelta(days=avg_interval)
            
            # Simple confidence score based on standard deviation (lower std = higher confidence)
            if pd.isna(std_interval) or avg_interval == 0:
                confidence = "Low"
            else:
                cv = std_interval / avg_interval # Coefficient of variation
                if cv < 0.2: confidence = "High 🟢"
                elif cv < 0.5: confidence = "Medium 🟡"
                else: confidence = "Low 🔴"
                
            predictions.append({
                'Customer': customer,
                'Last Order': last_order_date.strftime('%Y-%m-%d'),
                'Avg Interval (Days)': round(avg_interval, 1),
                'Predicted Next Order': next_expected.strftime('%Y-%m-%d'),
                'Confidence': confidence
            })
            
    return pd.DataFrame(predictions)

def process_stock_movement(df):
    """
    Tracks stock movement and identifies fast/slow moving SKUs.
    """
    # Similar aggressive cleaning
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    
    col_mapping = {}
    for col in df.columns:
        if any(kw in col for kw in ['item', 'product', 'sku', 'goods']):
            col_mapping[col] = 'sku'
        elif any(kw in col for kw in ['in', 'receipt', 'purchase', 'received']):
            col_mapping[col] = 'stock_in'
        elif any(kw in col for kw in ['out', 'issue', 'sales', 'consumed']):
            col_mapping[col] = 'stock_out'
        elif any(kw in col for kw in ['balance', 'current', 'stock']):
            col_mapping[col] = 'current_stock'
            
    df = df.rename(columns=col_mapping)
    
    # Velocity calculation if columns exist
    if 'sku' in df.columns and 'stock_out' in df.columns:
        velocity = df.groupby('sku')['stock_out'].sum().reset_index()
        velocity = velocity.sort_values(by='stock_out', ascending=False)
        
        # Categorize
        quantiles = velocity['stock_out'].quantile([0.33, 0.66])
        def categorize(val):
            if pd.isna(val) or val <= quantiles[0.33]: return 'Slow Mover'
            elif val <= quantiles[0.66]: return 'Steady'
            else: return 'Fast Mover'
            
        velocity['category'] = velocity['stock_out'].apply(categorize)
import pandas as pd
import numpy as np

def clean_and_standardize_sales(df):
    """
    Takes an unstructured raw excel dataframe and attempts to standardize it
    for the hackathon MVP.
    """
    # 1. Drop completely empty rows or columns
    df = df.dropna(how='all', axis=0).dropna(how='all', axis=1)
    
    # Optional: If the first few rows are garbage headers, heuristic to find the real header
    # For MVP, assuming the uploaded file has headers on the first row or
    # user selects it, but we'll lowercase and strip the columns for safety.
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    
    # Attempt to find standard columns by fuzzy matching
    col_mapping = {}
    for col in df.columns:
        if any(kw in col for kw in ['date', 'time', 'dt']):
            col_mapping[col] = 'date'
        elif any(kw in col for kw in ['customer', 'client', 'buyer', 'party']):
            col_mapping[col] = 'customer'
        elif any(kw in col for kw in ['item', 'product', 'sku', 'goods']):
            col_mapping[col] = 'sku'
        elif any(kw in col for kw in ['qty', 'quantity', 'volume', 'units']):
            col_mapping[col] = 'quantity'
        elif any(kw in col for kw in ['amount', 'price', 'total', 'revenue', 'value']):
            col_mapping[col] = 'revenue'
            
    df = df.rename(columns=col_mapping)
    
    # Ensure Date format if 'date' column exists
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
    return df

def rank_anchor_customers(df):
    """
    Identifies top revenue-driving customers (Anchor Customers).
    """
    if 'customer' not in df.columns or 'revenue' not in df.columns:
        return pd.DataFrame() # Return empty if columns aren't found
        
    # Group by customer and sum revenue
    customer_revenue = df.groupby('customer')['revenue'].sum().reset_index()
    customer_revenue = customer_revenue.sort_values(by='revenue', ascending=False)
    
    # Calculate percentage contribution
    total_rev = customer_revenue['revenue'].sum()
    customer_revenue['contribution_pct'] = (customer_revenue['revenue'] / total_rev) * 100
    
    return customer_revenue

def predict_reorder_intervals(df, anchor_customers_df, top_n=5):
    """
    Calculates average reorder interval for top customers and predicts the next window.
    """
    if 'date' not in df.columns or 'customer' not in df.columns:
        return pd.DataFrame()
        
    top_customers = anchor_customers_df.head(top_n)['customer'].tolist()
    
    predictions = []
    
    for customer in top_customers:
        cust_sales = df[df['customer'] == customer].sort_values(by='date')
        
        if len(cust_sales) > 1:
            # Calculate days between orders
            cust_sales['days_since_last'] = cust_sales['date'].diff().dt.days
            avg_interval = cust_sales['days_since_last'].mean()
            std_interval = cust_sales['days_since_last'].std()
            
            last_order_date = cust_sales['date'].max()
            
            # Predict next window
            next_expected = last_order_date + pd.Timedelta(days=avg_interval)
            
            # Simple confidence score based on standard deviation (lower std = higher confidence)
            if pd.isna(std_interval) or avg_interval == 0:
                confidence = "Low"
            else:
                cv = std_interval / avg_interval # Coefficient of variation
                if cv < 0.2: confidence = "High 🟢"
                elif cv < 0.5: confidence = "Medium 🟡"
                else: confidence = "Low 🔴"
                
            predictions.append({
                'Customer': customer,
                'Last Order': last_order_date.strftime('%Y-%m-%d'),
                'Avg Interval (Days)': round(avg_interval, 1),
                'Predicted Next Order': next_expected.strftime('%Y-%m-%d'),
                'Confidence': confidence
            })
            
    return pd.DataFrame(predictions)

def process_stock_movement(df):
    """
    Tracks stock movement and identifies fast/slow moving SKUs.
    """
    # Similar aggressive cleaning
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    
    col_mapping = {}
    for col in df.columns:
        if any(kw in col for kw in ['item', 'product', 'sku', 'goods']):
            col_mapping[col] = 'sku'
        elif any(kw in col for kw in ['in', 'receipt', 'purchase', 'received', 'qty', 'quantity', 'amount']):
            col_mapping[col] = 'stock_in'
        elif any(kw in col for kw in ['out', 'issue', 'sales', 'consumed']):
            col_mapping[col] = 'stock_out'
        elif any(kw in col for kw in ['balance', 'current', 'stock']):
            col_mapping[col] = 'current_stock'
            
    df = df.rename(columns=col_mapping)
    
    # Velocity calculation if columns exist
    if 'sku' in df.columns:
        # Determine tracking column (prefer outgoing sales, fallback to incoming purchases for velocity mapping)
        target_col = 'stock_out' if 'stock_out' in df.columns else 'stock_in' if 'stock_in' in df.columns else None
        
        if target_col:
            velocity = df.groupby('sku')[target_col].sum().reset_index()
            velocity = velocity.sort_values(by=target_col, ascending=False)
            
            # Categorize
            quantiles = velocity[target_col].quantile([0.33, 0.66])
            def categorize(val):
                if pd.isna(val) or val <= quantiles[0.33]: return 'Low Volume'
                elif val <= quantiles[0.66]: return 'Steady Volume'
                else: return 'High Volume'
                
            velocity['category'] = velocity[target_col].apply(categorize)
            
            # Merge back if current stock is present
            if 'current_stock' in df.columns:
                curr_stock = df.groupby('sku')['current_stock'].last().reset_index()
                velocity = pd.merge(velocity, curr_stock, on='sku', how='left')
                
            return velocity
        
    return df

def map_bom_to_raw_materials(bom_df, predicted_orders_df):
    """
    Takes predicted finished goods demand and maps it to raw material requirements via the BOM.
    """
    if bom_df is None or bom_df.empty or predicted_orders_df is None or predicted_orders_df.empty:
        return pd.DataFrame()
        
    # Standardize BOM columns
    bom_df.columns = [str(c).strip().lower().replace(" ", "_") for c in bom_df.columns]
    
    # Expected BOM columns conceptually: ['finished_good', 'raw_material', 'quantity_required']
    # If standard columns don't perfectly exist, returning the raw dataframe for demonstration
    col_mapping = {}
    for col in bom_df.columns:
        if any(kw in col for kw in ['finished', 'product', 'fg', 'item']):
            col_mapping[col] = 'finished_good'
        elif any(kw in col for kw in ['raw', 'rm', 'material', 'component']):
            col_mapping[col] = 'raw_material'
        elif any(kw in col for kw in ['qty', 'quantity', 'required', 'units']):
            col_mapping[col] = 'quantity_required'
            
    bom_df = bom_df.rename(columns=col_mapping)
    
    # Simple Mock Logic for Hackathon: 
    # If the exact columns aren't matched due to unstructured Excel, just return the BOM
    # so the dashboard can at least display the parsed data.
    if not all(col in bom_df.columns for col in ['finished_good', 'raw_material', 'quantity_required']):
        return bom_df
        
    # Example logic if perfectly mapped:
    # 1. Join predicted_orders with BOM on finished_good = sku
    # 2. Multiply predicted quantity by quantity_required
    # 3. Group by raw_material and sum requirements
    
    return bom_df # MVP simplified return for structural display