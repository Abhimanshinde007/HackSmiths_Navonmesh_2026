import pandas as pd
import numpy as np

def extract_true_dataframe(df):
    """Finds the true header row in an Excel sheet that might have metadata/letterheads on top."""
    try:
        if df is None or df.empty: return pd.DataFrame()
        
        current_cols = pd.DataFrame([df.columns.values])
        df.columns = range(df.shape[1])
        current_cols.columns = range(df.shape[1])
        df = pd.concat([current_cols, df], ignore_index=True)
        
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
                
        if max_keywords_found == 0 and len(df) > 0:
            non_null_counts = df.head(10).notna().sum(axis=1)
            best_row_idx = non_null_counts.idxmax()
            
        new_header = df.iloc[best_row_idx]
        df = df[best_row_idx+1:]
        df.columns = new_header
        
        df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
        df = df.dropna(how='all', axis=0).dropna(how='all', axis=1)
        return df
    except Exception:
        # Failsafe fallback
        return pd.DataFrame()

def clean_and_standardize_sales(df):
    try:
        df = extract_true_dataframe(df)
        if df.empty: return df
        
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
    except Exception:
        return pd.DataFrame()

def rank_anchor_customers(df):
    try:
        if df is None or df.empty or 'customer' not in df.columns or 'revenue' not in df.columns:
            return pd.DataFrame()
            
        df['revenue'] = pd.to_numeric(df['revenue'], errors='coerce').fillna(0)
        
        customer_revenue = df.groupby('customer')['revenue'].sum().reset_index()
        customer_revenue = customer_revenue.sort_values(by='revenue', ascending=False)
        
        total_rev = customer_revenue['revenue'].sum()
        if total_rev > 0:
            customer_revenue['contribution_pct'] = (customer_revenue['revenue'] / total_rev) * 100
        else:
            customer_revenue['contribution_pct'] = 0
        
        return customer_revenue
    except Exception:
        return pd.DataFrame()

def predict_reorder_intervals(df, anchor_customers_df, top_n=5):
    try:
        if df is None or df.empty or 'date' not in df.columns or 'customer' not in df.columns:
            return pd.DataFrame()
            
        if anchor_customers_df is None or anchor_customers_df.empty:
            return pd.DataFrame()
            
        top_customers = anchor_customers_df.head(top_n)['customer'].tolist()
        predictions = []
        
        df = df.dropna(subset=['date'])
        
        for customer in top_customers:
            cust_sales = df[df['customer'] == customer].sort_values(by='date')
            
            if 'revenue' in cust_sales.columns:
                cust_sales = cust_sales[pd.to_numeric(cust_sales['revenue'], errors='coerce').fillna(0) > 0]
                
            cust_sales = cust_sales.drop_duplicates(subset=['date'])
            
            if len(cust_sales) > 1:
                cust_sales['days_since_last'] = cust_sales['date'].diff().dt.days
                avg_interval = cust_sales['days_since_last'].mean()
                std_interval = cust_sales['days_since_last'].std()
                
                last_order_date = cust_sales['date'].max()
                
                next_expected = last_order_date + pd.Timedelta(days=avg_interval)
                
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
    except Exception:
        return pd.DataFrame()

def process_stock_movement(df):
    try:
        df = extract_true_dataframe(df)
        if df.empty: return df
        
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
                
                if 'current_stock' in df.columns:
                    df['current_stock'] = pd.to_numeric(df['current_stock'], errors='coerce')
                    curr_stock = df.groupby('sku')['current_stock'].last().reset_index()
                    velocity = pd.merge(velocity, curr_stock, on='sku', how='left')
                    
                return velocity
            
        return df
    except Exception:
        return pd.DataFrame()

def map_bom_to_raw_materials(bom_df, predicted_orders_df):
    try:
        if bom_df is None or bom_df.empty or predicted_orders_df is None or predicted_orders_df.empty:
            return pd.DataFrame()
            
        bom_df = extract_true_dataframe(bom_df)
        if bom_df.empty: return bom_df
        
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
    except Exception:
        return pd.DataFrame()