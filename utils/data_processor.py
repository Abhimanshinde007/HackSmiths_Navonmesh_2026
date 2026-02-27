"""
Data Processing Engine - PDR v2
Predictive Inventory & Procurement Intelligence Platform

Modules:
  4.1 - Intelligent Sales Register Ingestion
  4.2 - Anchor Customer Identification
  4.3 - Reorder Prediction Engine
  4.4 - BOM Upload & Mapping
  4.5 - Raw Material Forecast
"""

import pandas as pd
import numpy as np

# ─────────────────────────────────────────────
# 4.1 INTELLIGENT SALES REGISTER INGESTION
# ─────────────────────────────────────────────

SALES_HEADER_KEYWORDS = [
    'date', 'invoice', 'voucher',
    'party', 'customer', 'particulars', 'name',
    'item', 'product', 'goods',
    'qty', 'quantity', 'units'
]

COLUMN_MAP = {
    # Date variants
    'date': 'date', 'invoice_date': 'date', 'voucher_date': 'date',
    'bill_date': 'date', 'transaction_date': 'date', 'dt': 'date',
    # Customer variants
    'party': 'customer', 'customer': 'customer', 'customer_name': 'customer',
    'particulars': 'customer', 'client': 'customer', 'buyer': 'customer',
    'party_name': 'customer',
    # Product variants
    'item': 'product', 'item_name': 'product', 'product': 'product',
    'goods': 'product', 'description': 'product', 'particulars_1': 'product',
    # Quantity variants — includes financial value columns for invoice-level registers
    'qty': 'quantity', 'quantity': 'quantity', 'units': 'quantity',
    'nos': 'quantity', 'pcs': 'quantity', 'volume': 'quantity',
    # Financial value fallbacks (used when no product-qty column exists)
    'gross_total': 'quantity', 'grosstotal': 'quantity',
    'value': 'quantity', 'net_amount': 'quantity', 'net_total': 'quantity',
    'amount': 'quantity', 'invoice_value': 'quantity', 'taxable_value': 'quantity',
}

TOTAL_KEYWORDS = ['total', 'grand total', 'sub total', 'subtotal', 'net total']


def _detect_header_row(df_raw):
    """Scan first 20 rows to find the true table header."""
    best_row = 0
    best_score = 0
    for i in range(min(20, len(df_raw))):
        row_values = [str(x).strip().lower() for x in df_raw.iloc[i].values if pd.notna(x)]
        score = sum(1 for cell in row_values for kw in SALES_HEADER_KEYWORDS if kw in cell)
        if score > best_score:
            best_score = score
            best_row = i
    return best_row


def _standardize_columns(df):
    """Lowercase, strip, map column names to standard schema, then deduplicate."""
    df.columns = [
        str(c).strip().lower()
                 .replace(" ", "_")
                 .replace(".", "_")
                 .replace("/", "_")
                 .replace("-", "_")
        for c in df.columns
    ]
    rename = {}
    for col in df.columns:
        if col in COLUMN_MAP:
            rename[col] = COLUMN_MAP[col]
        else:
            for key, std in COLUMN_MAP.items():
                if key in col:
                    rename[col] = std
                    break
    df = df.rename(columns=rename)

    # ── CRITICAL: resolve duplicate column names caused by multiple source columns
    # mapping to the same schema name (e.g. both 'value' and 'gross_total' → 'quantity').
    # For each duplicate, keep the first occurrence and drop the rest.
    seen = {}
    new_cols = []
    for col in df.columns:
        if col not in seen:
            seen[col] = 0
            new_cols.append(col)
        else:
            seen[col] += 1
            new_cols.append(f"{col}__drop_{seen[col]}")
    df.columns = new_cols
    drop_cols = [c for c in df.columns if '__drop_' in c]
    df = df.drop(columns=drop_cols)

    return df


def _is_totals_row(row):
    """Return True if this row looks like a totals/grand-total row."""
    for val in row.values:
        if any(kw in str(val).lower() for kw in TOTAL_KEYWORDS):
            return True
    return False


def load_sales_excel(file):
    """
    PDR 4.1 — Full ingestion pipeline.
    Returns clean df with columns: [date, customer, product, quantity]
    """
    try:
        # Read raw without any header so we can detect it
        raw = pd.read_excel(file, header=None, dtype=str)
        raw = raw.dropna(how='all').reset_index(drop=True)

        header_row = _detect_header_row(raw)
        df = pd.read_excel(file, header=header_row, dtype=str)

        df = _standardize_columns(df)

        # Remove totals rows
        df = df[~df.apply(_is_totals_row, axis=1)]

        # Ensure required columns exist - quantity may come from financial columns
        required = ['date', 'customer']
        missing = [c for c in required if c not in df.columns]
        if missing:
            return None, f"Could not find columns: {missing}. Detected columns: {list(df.columns)}"

        # If quantity still missing after mapping, try to use any numeric column as fallback
        if 'quantity' not in df.columns:
            numeric_cols = df.select_dtypes(include='number').columns.tolist()
            if numeric_cols:
                df = df.rename(columns={numeric_cols[0]: 'quantity'})
            else:
                return None, f"No numeric column found to use as order value. Columns: {list(df.columns)}"

        # Keep only schema columns that exist
        keep_cols = [c for c in ['date', 'customer', 'product', 'quantity'] if c in df.columns]
        df = df[keep_cols]

        # Type conversion
        df['date'] = pd.to_datetime(df['date'], errors='coerce', dayfirst=True)
        df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')

        # Drop nulls in required fields
        df = df.dropna(subset=['date', 'customer', 'quantity'])
        df = df[df['quantity'] > 0]

        # Clean customer names (strip whitespace, drop all-numeric rows)
        df['customer'] = df['customer'].astype(str).str.strip()
        df = df[df['customer'].str.len() > 1]

        df = df.reset_index(drop=True)
        return df, None

    except Exception as e:
        return None, str(e)


# ─────────────────────────────────────────────
# 4.2 ANCHOR CUSTOMER IDENTIFICATION
# ─────────────────────────────────────────────

def get_anchor_customers(df):
    """
    PDR 4.2 — Group, rank, return top 20% or top 5.
    Returns df: [customer, total_quantity, order_count, contribution_pct]
    """
    try:
        agg = df.groupby('customer').agg(
            total_quantity=('quantity', 'sum'),
            order_count=('quantity', 'count')
        ).reset_index()

        agg = agg.sort_values('total_quantity', ascending=False).reset_index(drop=True)

        total = agg['total_quantity'].sum()
        agg['contribution_pct'] = (agg['total_quantity'] / total * 100).round(2)

        # Top 20% or top 5
        top_n = max(1, min(5, int(np.ceil(len(agg) * 0.20))))
        return agg.head(top_n), None

    except Exception as e:
        return pd.DataFrame(), str(e)


# ─────────────────────────────────────────────
# 4.3 REORDER PREDICTION ENGINE
# ─────────────────────────────────────────────

def predict_reorder(df, anchor_customers):
    """
    PDR 4.3 — Per-customer statistical reorder prediction.
    Skips customers with < 3 orders.
    Returns df: [customer, last_order, avg_interval, predicted_date,
                 window_start, window_end, confidence_pct]
    """
    try:
        results = []
        for _, row in anchor_customers.iterrows():
            cust = row['customer']
            orders = df[df['customer'] == cust].sort_values('date')
            orders = orders.drop_duplicates(subset=['date'])

            if len(orders) < 3:
                continue

            intervals = orders['date'].diff().dt.days.dropna()
            mu = intervals.mean()
            sigma = intervals.std()

            if mu <= 0 or pd.isna(mu):
                continue

            last_order = orders['date'].max()
            predicted = last_order + pd.Timedelta(days=mu)
            half_sigma = (sigma * 0.5) if not pd.isna(sigma) else 0
            window_start = predicted - pd.Timedelta(days=half_sigma)
            window_end = predicted + pd.Timedelta(days=half_sigma)

            # Confidence = max(0, 100 − (σ/μ × 100))
            if pd.isna(sigma):
                confidence = 50.0
            else:
                confidence = max(0.0, round(100 - (sigma / mu * 100), 1))

            results.append({
                'Customer': cust,
                'Last Order': last_order.strftime('%d %b %Y'),
                'Avg Interval (Days)': round(mu, 1),
                'Predicted Next Order': predicted.strftime('%d %b %Y'),
                'Window Start': window_start.strftime('%d %b %Y'),
                'Window End': window_end.strftime('%d %b %Y'),
                'Confidence %': confidence
            })

        return pd.DataFrame(results), None

    except Exception as e:
        return pd.DataFrame(), str(e)


# ─────────────────────────────────────────────
# 4.4 BOM UPLOAD & MAPPING
# ─────────────────────────────────────────────

BOM_COLUMN_MAP = {
    'finished_good': ['finished', 'product', 'fg', 'item', 'goods'],
    'raw_material':  ['raw', 'rm', 'material', 'component', 'input'],
    'qty_per_unit':  ['qty', 'quantity', 'required', 'units', 'per_unit', 'ratio'],
}


def process_bom(file):
    """
    PDR 4.4 — Load and standardize BOM file.
    Returns df: [finished_good, raw_material, qty_per_unit]
    """
    try:
        raw = pd.read_excel(file, header=None, dtype=str)
        raw = raw.dropna(how='all').reset_index(drop=True)
        header_row = _detect_header_row(raw)
        df = pd.read_excel(file, header=header_row, dtype=str)

        df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

        rename = {}
        for std_name, keywords in BOM_COLUMN_MAP.items():
            for col in df.columns:
                if any(kw in col for kw in keywords) and col not in rename:
                    rename[col] = std_name
                    break

        df = df.rename(columns=rename)

        required = ['finished_good', 'raw_material', 'qty_per_unit']
        missing = [c for c in required if c not in df.columns]
        if missing:
            return None, f"BOM missing columns: {missing}. Found: {list(df.columns)}"

        df = df[required].dropna()
        df['qty_per_unit'] = pd.to_numeric(df['qty_per_unit'], errors='coerce')
        df = df.dropna(subset=['qty_per_unit'])
        return df.reset_index(drop=True), None

    except Exception as e:
        return None, str(e)


# ─────────────────────────────────────────────
# 4.5 RAW MATERIAL FORECAST
# ─────────────────────────────────────────────

def forecast_raw_materials(df, anchor_customers, predictions_df, bom_df):
    """
    PDR 4.5 — Convert predicted finished goods demand into raw material requirements.
    Returns df: [raw_material, projected_requirement, advisory]
    """
    try:
        if bom_df is None or bom_df.empty:
            return pd.DataFrame(), "No BOM data available."

        # Compute average past order quantity per anchor customer
        avg_qty = df.groupby('customer')['quantity'].mean().reset_index()
        avg_qty.columns = ['customer', 'avg_order_qty']

        # Overall historical avg consumption per product
        historical_avg = df.groupby('product')['quantity'].mean().reset_index() if 'product' in df.columns else pd.DataFrame()

        rows = []
        for _, cust_row in anchor_customers.iterrows():
            cust = cust_row['customer']
            avg_q = avg_qty[avg_qty['customer'] == cust]['avg_order_qty'].values
            if len(avg_q) == 0:
                continue
            predicted_qty = avg_q[0]

            # For each BOM entry, compute raw material requirement
            for _, bom_row in bom_df.iterrows():
                rm = bom_row['raw_material']
                qty_pu = bom_row['qty_per_unit']
                projected = round(predicted_qty * qty_pu, 2)
                rows.append({
                    'Customer': cust,
                    'Finished Good': bom_row['finished_good'],
                    'Raw Material': rm,
                    'Projected Requirement': projected,
                })

        if not rows:
            return pd.DataFrame(), None

        result = pd.DataFrame(rows)
        result = result.groupby('Raw Material')['Projected Requirement'].sum().reset_index()

        # Rule-based advisory
        result['Advisory'] = result['Projected Requirement'].apply(
            lambda x: '🔴 Prepare / Procure Soon' if x > result['Projected Requirement'].mean() else '🟡 Monitor'
        )

        return result.sort_values('Projected Requirement', ascending=False).reset_index(drop=True), None

    except Exception as e:
        return pd.DataFrame(), str(e)