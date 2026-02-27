import pandas as pd
import numpy as np

def extract_true_dataframe(df):
    """Finds the true header row in an Excel sheet that might have metadata/letterheads on top."""
    # Put current columns back as a row, since pd.read_excel might have consumed the first metadata row as header
    if df.empty: return df
    
    current_cols = pd.DataFrame([df.columns.values])
    df.columns = range(df.shape[1])
    current_cols.columns = range(df.shape[1])
    df = pd.concat([current_cols, df], ignore_index=True)
    
    # Drop rows that are completely empty
    df = df.dropna(how='all').reset_index(drop=True)
    
    keywords = ['date', 'particulars', 'customer', 'item', 'qty', 'quantity', 'amount', 'total', 'voucher', 'sku', 'value', 'gross']
    
    best_row_idx = 0
    max_keywords_found = 0
    
    for i in range(min(20, len(df))):
        row_values = [str(x).lower() for x in df.iloc[i].values if pd.notna(x)]
        kw_count = sum(1 for rv in row_values for kw in keywords if kw in rv)
        if kw_count > max_keywords_found:
            max_keywords_found = kw_count
            best_row_idx = i
            
    if max_keywords_found == 0:
        non_null_counts = df.head(10).notna().sum(axis=1)
        best_row_idx = non_null_counts.idxmax()
        
    new_header = df.iloc[best_row_idx]
    df = df[best_row_idx+1:]
    df.columns = new_header
    
    # Clean up column names
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    
    df = df.dropna(how='all', axis=0).dropna(how='all', axis=1)
    return df

def clean_and_standardize_sales(df):
    df = extract_true_dataframe(df)
    
    col_mapping = {}
    for col in df.columns:
        if any(kw in col for kw in ['date', 'time', 'dt']):
            col_mapping[col] = 'date'
        elif any(kw in col for kw in ['customer', 'client', 'buyer', 'party', 'particulars']):
            col_mapping[col] = 'customer'
        elif any(kw in col for kw in ['item', 'product', 'sku', 'goods']):
            col_mapping[col] = 'sku'
        elif any(kw in col for kw in ['qty', 'quantity', 'volume', 'units']):
            col_mapping[col] = 'quantity'
        elif any(kw in col for kw in ['amount', 'price', 'total', 'revenue', 'value', 'gross']):
            col_mapping[col] = 'revenue'
            
    df = df.rename(columns=col_mapping)
    
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
    return df

def rank_anchor_customers(df):
    if 'customer' not in df.columns or 'revenue' not in df.columns:
        return pd.DataFrame() # Return empty if columns aren't found
        
    df['revenue'] = pd.to_numeric(df['revenue'], errors='coerce').fillna(0)
    
    # Group by customer and sum revenue
    customer_revenue = df.groupby('customer')['revenue'].sum().reset_index()
    customer_revenue = customer_revenue.sort_values(by='revenue', ascending=False)
    
    # Calculate percentage contribution
    total_rev = customer_revenue['revenue'].sum()
    if total_rev > 0:
        customer_revenue['contribution_pct'] = (customer_revenue['revenue'] / total_rev) * 100
    else:
        customer_revenue['contribution_pct'] = 0
    
    return customer_revenue

def predict_reorder_intervals(df, anchor_customers_df, top_n=5):
    if 'date' not in df.columns or 'customer' not in df.columns:
        return pd.DataFrame()
        
    top_customers = anchor_customers_df.head(top_n)['customer'].tolist()
    predictions = []
    
    # Drop rows without dates
    df = df.dropna(subset=['date'])
    
    for customer in top_customers:
        cust_sales = df[df['customer'] == customer].sort_values(by='date')
        
        # Only keep rows with positive revenue to count as actual orders
        if 'revenue' in cust_sales.columns:
            cust_sales = cust_sales[pd.to_numeric(cust_sales['revenue'], errors='coerce').fillna(0) > 0]
            
        # Group by date to avoid multiple entries on same day throwing off frequency
        cust_sales = cust_sales.drop_duplicates(subset=['date'])
        
        if len(cust_sales) > 1:
            # Calculate days between orders
            cust_sales['days_since_last'] = cust_sales['date'].diff().dt.days
            avg_interval = cust_sales['days_since_last'].mean()
            std_interval = cust_sales['days_since_last'].std()
            
            last_order_date = cust_sales['date'].max()
            
            # Predict next window
            next_expected = last_order_date + pd.Timedelta(days=avg_interval)
            
            # Simple confidence score based on standard deviation
            if pd.isna(std_interval) or avg_interval == 0:
                confidence = "Low"
            else:
                cv = std_interval / avg_interval
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
    df = extract_true_dataframe(df)
    
    col_mapping = {}
    for col in df.columns:
        if any(kw in col for kw in ['item', 'product', 'sku', 'goods', 'particulars']):
            col_mapping[col] = 'sku'
        elif any(kw in col for kw in ['in', 'receipt', 'purchase', 'received', 'qty', 'quantity', 'amount']):
            col_mapping[col] = 'stock_in'
        elif any(kw in col for kw in ['out', 'issue', 'sales', 'consumed']):
            col_mapping[col] = 'stock_out'
        elif any(kw in col for kw in ['balance', 'current', 'stock']):
            col_mapping[col] = 'current_stock'
            
    df = df.rename(columns=col_mapping)
    
    if 'sku' in df.columns:
        # Determine tracking column (prefer outgoing sales, fallback to incoming purchases)
        target_col = 'stock_out' if 'stock_out' in df.columns else 'stock_in' if 'stock_in' in df.columns else None
        
        if target_col:
            df[target_col] = pd.to_numeric(df[target_col], errors='coerce').fillna(0)
            
            velocity = df.groupby('sku')[target_col].sum().reset_index()
            velocity = velocity.sort_values(by=target_col, ascending=False)
            
            non_zero = velocity[velocity[target_col] > 0]
            if not non_zero.empty:
                quantiles = non_zero[target_col].quantile([0.33, 0.66])
                def categorize(val):
                    if pd.isna(val) or val == 0: return 'No Movement'
                    elif val <= quantiles[0.33]: return 'Low Volume'
                    elif val <= quantiles[0.66]: return 'Steady Volume'
                    else: return 'High Volume'
                    
                velocity['category'] = velocity[target_col].apply(categorize)
            else:
                velocity['category'] = 'No Movement'
            
            # Merge back if current stock is present
            if 'current_stock' in df.columns:
                df['current_stock'] = pd.to_numeric(df['current_stock'], errors='coerce')
                curr_stock = df.groupby('sku')['current_stock'].last().reset_index()
                velocity = pd.merge(velocity, curr_stock, on='sku', how='left')
                
            return velocity
        
    return df

def map_bom_to_raw_materials(bom_df, predicted_orders_df):
    if bom_df is None or bom_df.empty or predicted_orders_df is None or predicted_orders_df.empty:
        return pd.DataFrame()
        
    bom_df = extract_true_dataframe(bom_df)
    
    col_mapping = {}
    for col in bom_df.columns:
        if any(kw in col for kw in ['finished', 'product', 'fg', 'item']):
            col_mapping[col] = 'finished_good'
        elif any(kw in col for kw in ['raw', 'rm', 'material', 'component']):
            col_mapping[col] = 'raw_material'
        elif any(kw in col for kw in ['qty', 'quantity', 'required', 'units']):
            col_mapping[col] = 'quantity_required'
            
    bom_df = bom_df.rename(columns=col_mapping)
    
    if not all(col in bom_df.columns for col in ['finished_good', 'raw_material', 'quantity_required']):
        return bom_df
        
    return bom_df