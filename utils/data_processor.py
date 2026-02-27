"""
Inventory Intelligence Engine – Data Processor
Strict, rule-based, deterministic field mapping engine.
No AI. No APIs. No PDF. Excel-only.
"""

import re
import pandas as pd
import numpy as np


# ─────────────────────────────────────────────────────────────
# COLUMN DETECTION ENGINE
# ─────────────────────────────────────────────────────────────

def _norm(s):
    """Normalize a column name for matching."""
    return re.sub(r'[^a-z0-9 ]', ' ', str(s).lower()).strip()


def _detect_header_row(raw_df, min_kw, keywords):
    """Scan first 30 rows, find row with >= min_kw matches."""
    for i in range(min(30, len(raw_df))):
        row_vals = [_norm(v) for v in raw_df.iloc[i].tolist()]
        row_text = ' '.join(row_vals)
        hits = sum(1 for kw in keywords if kw in row_text)
        if hits >= min_kw:
            return i
    return None


# ─────────────────────────────────────────────────────────────
# STRICT COLUMN MAPPING DICTIONARIES
# ─────────────────────────────────────────────────────────────

# DATE: must contain "date", never contain "due" 
DATE_KEYWORDS     = {'invoice date', 'date', 'bill date', 'voucher date', 'invoice_date', 'bill_date', 'voucher_date'}
DATE_REJECT       = {'due'}

# CUSTOMER: must contain party/customer/buyer/bill
CUSTOMER_KEYWORDS = {'party name', 'customer name', 'buyer', 'bill to', 'm/s', 'customer', 'party', 'client', 'consignee'}
CUSTOMER_MUST_HAVE = {'party', 'customer', 'buyer', 'bill', 'client', 'consignee', 'm/s'}

# PRODUCT: must NOT contain rate/amount/price
PRODUCT_KEYWORDS  = {'item', 'product', 'description', 'stock item', 'goods', 'item name', 'particulars', 'material', 'stock_item'}
PRODUCT_REJECT    = {'rate', 'amount', 'price'}

# QUANTITY: allowed keywords, must NOT contain rate/amount/price/value
QUANTITY_KEYWORDS = {'qty', 'quantity', 'billed qty', 'actual qty', 'units', 'nos', 'billed_qty', 'actual_qty', 'pcs', 'pieces'}
QUANTITY_REJECT   = {'rate', 'amount', 'price', 'value'}

# RATE: allowed, must NOT contain qty/quantity
RATE_KEYWORDS     = {'rate', 'price', 'unit rate', 'unit_rate', 'rate/unit'}
RATE_REJECT       = {'qty', 'quantity'}

# SUPPLIER (purchase bills)
SUPPLIER_KEYWORDS = {'supplier', 'vendor', 'party name', 'party', 'seller', 'from', 'manufacturer'}
SUPPLIER_MUST_HAVE = {'supplier', 'vendor', 'party', 'seller', 'from', 'manufacturer'}

# INWARD / OUTWARD
INWARD_KEYWORDS   = {'inward', 'receipt', 'received', 'in', 'purchase', 'qty in', 'inward qty'}
OUTWARD_KEYWORDS  = {'outward', 'issue', 'issued', 'out', 'sales', 'qty out', 'dispatch', 'dispatched'}
BALANCE_KEYWORDS  = {'balance', 'closing', 'current stock', 'stock'}


def _map_column(col_name, allowed_keywords, reject_keywords=None):
    """Return True if col_name matches this field's rules."""
    c = _norm(col_name)
    reject_keywords = reject_keywords or set()
    # Reject if any reject keyword appears
    for rk in reject_keywords:
        if rk in c:
            return False
    # Accept if any allowed keyword appears
    for ak in allowed_keywords:
        if ak in c:
            return True
    return False


def _detect_columns(df, field_rules):
    """
    Map DataFrame columns to canonical field names.
    field_rules: dict of {canonical_name: (allowed_kws, reject_kws)}
    Returns dict {canonical: actual_col_name} and list of unmapped required fields.
    """
    mapped = {}
    for canon, (allowed, reject) in field_rules.items():
        for col in df.columns:
            if col in mapped.values():
                continue
            if _map_column(col, allowed, reject):
                mapped[canon] = col
                break
    return mapped


# ─────────────────────────────────────────────────────────────
# FILE READER
# ─────────────────────────────────────────────────────────────

def _read_excel_smart(file, context_keywords, min_kw=2):
    """
    Read an Excel file, detect the header row, and return a clean DataFrame.
    Returns (df, error_string).
    """
    try:
        raw = pd.read_excel(file, header=None, dtype=str).fillna('')
        hdr_row = _detect_header_row(raw, min_kw, context_keywords)
        if hdr_row is None:
            return None, "Unable to detect structured table header in this file. Ensure it contains column names like Date, Customer, Qty, etc."
        df = pd.read_excel(file, header=hdr_row, dtype=str)
        df.columns = [str(c).strip() for c in df.columns]
        df = df.replace('', pd.NA)
        df = df.dropna(how='all').reset_index(drop=True)
        return df, None
    except Exception as e:
        return None, f"File read error: {str(e)}"


def _clean_numeric(series):
    """Strip currency symbols, commas, etc. and convert to float."""
    return pd.to_numeric(
        series.astype(str).str.replace(r'[^\d.\-]', '', regex=True),
        errors='coerce'
    )


def _remove_totals(df, columns):
    """Remove rows that look like grand totals."""
    for col in columns:
        if col in df.columns:
            mask = df[col].astype(str).str.lower().str.contains(
                r'\btotal\b|\bgrand\b|\bsubtotal\b', regex=True, na=False
            )
            df = df[~mask]
    return df.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────
# 1. SALES BILLS INGESTION
# ─────────────────────────────────────────────────────────────

SALES_CONTEXT_KW = ['invoice', 'date', 'party', 'customer', 'item', 'qty', 'quantity',
                    'product', 'description', 'buyer', 'amount', 'rate']

SALES_FIELD_RULES = {
    'date':     (DATE_KEYWORDS,     DATE_REJECT),
    'customer': (CUSTOMER_KEYWORDS, set()),
    'product':  (PRODUCT_KEYWORDS,  PRODUCT_REJECT),
    'quantity': (QUANTITY_KEYWORDS, QUANTITY_REJECT),
    'rate':     (RATE_KEYWORDS,     RATE_REJECT),
}


def ingest_sales_excel(files):
    """
    Accept a list of file-like objects (or a single file).
    Returns (df, errors) — df has columns: date, customer, product, quantity, rate.
    """
    if not isinstance(files, list):
        files = [files]

    frames, errors = [], []
    for f in files:
        fname = getattr(f, 'name', str(f))
        df_raw, err = _read_excel_smart(f, SALES_CONTEXT_KW, min_kw=2)
        if err:
            errors.append(f"{fname}: {err}")
            continue

        mapped = _detect_columns(df_raw, SALES_FIELD_RULES)

        required = ['date', 'customer', 'quantity']
        missing = [r for r in required if r not in mapped]
        if missing:
            col_list = ', '.join(f'"{m}"' for m in missing)
            errors.append(f"{fname}: Required column(s) not detected: {col_list}. "
                          f"Check column names match: Date/Customer/Qty.")
            continue

        # Build clean frame
        rename = {v: k for k, v in mapped.items()}
        subset = df_raw[[v for v in mapped.values()]].rename(columns=rename)
        if 'product' not in subset.columns:
            subset['product'] = 'UNSPECIFIED'
        if 'rate' not in subset.columns:
            subset['rate'] = pd.NA

        subset = _remove_totals(subset, ['customer', 'product'])
        subset['date']     = pd.to_datetime(subset['date'], errors='coerce', dayfirst=True)
        subset['quantity'] = _clean_numeric(subset['quantity'])
        subset['rate']     = _clean_numeric(subset['rate']) if 'rate' in subset.columns else pd.NA

        subset = subset.dropna(subset=['date', 'customer', 'quantity'])
        subset = subset[subset['quantity'] > 0]
        subset['customer'] = subset['customer'].astype(str).str.strip().str.title()
        subset['product']  = subset['product'].astype(str).str.strip()
        subset             = subset[subset['customer'].str.len() > 1]
        subset['source']   = fname
        frames.append(subset)

    if not frames:
        return pd.DataFrame(), errors
    return pd.concat(frames, ignore_index=True), errors


# ─────────────────────────────────────────────────────────────
# 2. PURCHASE BILLS INGESTION
# ─────────────────────────────────────────────────────────────

PURCHASE_CONTEXT_KW = ['supplier', 'vendor', 'date', 'material', 'item', 'qty', 'quantity', 'rate', 'purchase']

PURCHASE_FIELD_RULES = {
    'date':     (DATE_KEYWORDS,     DATE_REJECT),
    'supplier': (SUPPLIER_KEYWORDS, set()),
    'material': (PRODUCT_KEYWORDS,  PRODUCT_REJECT),
    'quantity': (QUANTITY_KEYWORDS, QUANTITY_REJECT),
    'rate':     (RATE_KEYWORDS,     RATE_REJECT),
}


def ingest_purchase_excel(files):
    """
    Accept a list of file-like objects.
    Returns (df, errors) — df has columns: date, supplier, material, quantity, rate.
    """
    if not isinstance(files, list):
        files = [files]

    frames, errors = [], []
    for f in files:
        fname = getattr(f, 'name', str(f))
        df_raw, err = _read_excel_smart(f, PURCHASE_CONTEXT_KW, min_kw=2)
        if err:
            errors.append(f"{fname}: {err}")
            continue

        mapped = _detect_columns(df_raw, PURCHASE_FIELD_RULES)

        required = ['date', 'quantity']
        missing = [r for r in required if r not in mapped]
        if missing:
            errors.append(f"{fname}: Required column(s) not detected: {', '.join(missing)}.")
            continue

        rename = {v: k for k, v in mapped.items()}
        subset = df_raw[[v for v in mapped.values()]].rename(columns=rename)
        if 'supplier' not in subset.columns:
            subset['supplier'] = 'UNSPECIFIED'
        if 'material' not in subset.columns:
            subset['material'] = 'UNSPECIFIED'
        if 'rate' not in subset.columns:
            subset['rate'] = pd.NA

        subset = _remove_totals(subset, ['supplier', 'material'])
        subset['date']     = pd.to_datetime(subset['date'], errors='coerce', dayfirst=True)
        subset['quantity'] = _clean_numeric(subset['quantity'])
        subset['rate']     = _clean_numeric(subset['rate'])

        subset = subset.dropna(subset=['date', 'quantity'])
        subset = subset[subset['quantity'] > 0]
        subset['source'] = fname
        frames.append(subset)

    if not frames:
        return pd.DataFrame(), errors
    return pd.concat(frames, ignore_index=True), errors


# ─────────────────────────────────────────────────────────────
# 3. STOCK REGISTER INGESTION
# ─────────────────────────────────────────────────────────────

STOCK_CONTEXT_KW = ['material', 'item', 'inward', 'outward', 'stock', 'qty', 'balance', 'in', 'out']

STOCK_FIELD_RULES = {
    'material': (PRODUCT_KEYWORDS | {'material'}, PRODUCT_REJECT),
    'inward':   (INWARD_KEYWORDS,  set()),
    'outward':  (OUTWARD_KEYWORDS, set()),
    'balance':  (BALANCE_KEYWORDS, set()),
}


def ingest_stock_excel(file):
    """
    Accept a single stock register Excel file.
    Returns (df, error) — df has columns: material, inward, outward.
    """
    fname = getattr(file, 'name', str(file))
    df_raw, err = _read_excel_smart(file, STOCK_CONTEXT_KW, min_kw=2)
    if err:
        return pd.DataFrame(), err

    mapped = _detect_columns(df_raw, STOCK_FIELD_RULES)

    if 'material' not in mapped:
        return pd.DataFrame(), f"{fname}: Material/Item column not detected."

    rename = {v: k for k, v in mapped.items()}
    subset = df_raw[[v for v in mapped.values()]].rename(columns=rename)

    if 'inward' not in subset.columns:
        subset['inward'] = 0
    if 'outward' not in subset.columns:
        subset['outward'] = 0

    subset = _remove_totals(subset, ['material'])
    subset['inward']  = _clean_numeric(subset['inward']).fillna(0)
    subset['outward'] = _clean_numeric(subset['outward']).fillna(0)
    subset['material'] = subset['material'].astype(str).str.strip()
    subset = subset[subset['material'].str.len() > 1]
    return subset.reset_index(drop=True), None


# ─────────────────────────────────────────────────────────────
# 4. ANALYTICS ENGINE
# ─────────────────────────────────────────────────────────────

def get_anchor_customers(sales_df, top_n=5):
    """Top N customers by total quantity. Returns (df, error)."""
    try:
        if sales_df is None or sales_df.empty:
            return pd.DataFrame(), "No sales data."
        agg = (sales_df.groupby('customer')
               .agg(total_qty=('quantity', 'sum'),
                    invoices=('quantity', 'count'))
               .reset_index()
               .sort_values('total_qty', ascending=False)
               .head(top_n)
               .reset_index(drop=True))
        total = agg['total_qty'].sum()
        agg['share_pct'] = ((agg['total_qty'] / total) * 100).round(1) if total > 0 else 0.0
        agg.columns = ['Customer', 'Total Qty', 'Invoices', 'Share %']
        return agg, None
    except Exception as e:
        return pd.DataFrame(), str(e)


def predict_reorder(sales_df, anchor_df):
    """
    For each anchor customer, compute mean/sigma interval and predict next order.
    Returns (df, error).
    """
    try:
        if sales_df is None or sales_df.empty or anchor_df is None or anchor_df.empty:
            return pd.DataFrame(), "No data."
        results = []
        for _, row in anchor_df.iterrows():
            cust = row['Customer']
            orders = (sales_df[sales_df['customer'] == cust]
                      .sort_values('date')
                      .drop_duplicates('date'))
            if len(orders) < 2:
                continue
            intervals = orders['date'].diff().dt.days.dropna()
            mu    = intervals.mean()
            sigma = intervals.std() if len(intervals) > 1 else 0.0
            if mu <= 0 or pd.isna(mu):
                continue
            last_order = orders['date'].max()
            predicted  = last_order + pd.Timedelta(days=round(mu))
            half_sig   = (sigma * 0.5) if not pd.isna(sigma) else 0

            if mu == 0:
                confidence = 0.0
            else:
                confidence = max(0.0, round(100 - (sigma / mu * 100), 1)) if not pd.isna(sigma) else 50.0

            early = predicted - pd.Timedelta(days=round(half_sig))
            late  = predicted + pd.Timedelta(days=round(half_sig))

            results.append({
                'Customer':             cust,
                'Last Order':           last_order.strftime('%d %b %Y'),
                'Avg Interval (Days)':  round(mu, 1),
                'Predicted Next Order': predicted.strftime('%d %b %Y'),
                'Window':               f"{early.strftime('%d %b')} – {late.strftime('%d %b %Y')}",
                'Confidence %':         confidence,
            })
        if not results:
            return pd.DataFrame(), "Not enough order history (need ≥2 dates per customer)."
        return pd.DataFrame(results), None
    except Exception as e:
        return pd.DataFrame(), str(e)


def compute_stock(stock_df):
    """
    Aggregate material movements.
    Returns (df, error) with columns: Material, Total Inward, Total Outward, Current Stock.
    """
    try:
        if stock_df is None or stock_df.empty:
            return pd.DataFrame(), "No stock data."
        agg = (stock_df.groupby('material')
               .agg(total_inward=('inward', 'sum'),
                    total_outward=('outward', 'sum'))
               .reset_index())
        agg['current_stock'] = agg['total_inward'] - agg['total_outward']
        # Highlight low stock: below 20% of total inward
        agg['low_stock'] = agg['current_stock'] < (agg['total_inward'] * 0.2)
        agg.columns = ['Material', 'Total Inward', 'Total Outward', 'Current Stock', 'Low Stock']
        return agg.sort_values('Current Stock').reset_index(drop=True), None
    except Exception as e:
        return pd.DataFrame(), str(e)


def material_outlook(stock_df, predictions_df, sales_df):
    """
    Compare predicted demand vs current stock.
    Returns (df, error).
    """
    try:
        if stock_df is None or stock_df.empty:
            return pd.DataFrame(), "No stock data for outlook."

        stock_agg, _ = compute_stock(stock_df)
        if stock_agg is None or stock_agg.empty:
            return pd.DataFrame(), "No stock after aggregation."

        # Get avg monthly outward per material
        rows = []
        for _, srow in stock_agg.iterrows():
            mat   = srow['Material']
            c_stk = srow['Current Stock']

            # Estimate demand from sales data product column (fuzzy match on material name)
            demand_est = 0.0
            if sales_df is not None and not sales_df.empty and 'product' in sales_df.columns:
                mask = sales_df['product'].str.lower().str.contains(
                    re.escape(mat.lower()[:6]), na=False
                )
                demand_est = float(sales_df.loc[mask, 'quantity'].sum())

            if predictions_df is not None and not predictions_df.empty:
                # Simple: if any prediction is within 30 days, treat demand as urgent
                status = 'Monitor'
                if demand_est > c_stk and c_stk >= 0:
                    status = 'Buy Soon'
                elif demand_est > c_stk * 0.7:
                    status = 'Prepare'
            else:
                if c_stk <= 0:
                    status = 'Buy Soon'
                elif srow['Low Stock']:
                    status = 'Prepare'
                else:
                    status = 'Monitor'

            rows.append({
                'Material':       mat,
                'Current Stock':  round(c_stk, 1),
                'Est. Demand':    round(demand_est, 1),
                'Advisory':       status,
            })
        return pd.DataFrame(rows), None
    except Exception as e:
        return pd.DataFrame(), str(e)