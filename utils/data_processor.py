"""
Data Processing Engine — Full MVP with Tally PDF Support
Registers: Sales, Purchase, Inward/Outward
"""

import io
import re
import pandas as pd
import numpy as np

try:
    import pdfplumber
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False


# ─────────────────────────────────────────────────────────────
# SHARED CONSTANTS
# ─────────────────────────────────────────────────────────────

SALES_KEYWORDS   = [
    'voucher', 'date', 'party', 'particulars', 'stock', 'item', 'product',
    'qty', 'billed', 'quantity', 'gross', 'value', 'amount', 'total',
    'sgst', 'cgst', 'igst', 'gst', 'gstin', 'uin', 'tax',
    'sales', 'invoice', 'bill', 'account', 'ref', 'no',
]
PURCHASE_KEYWORDS = [
    'date', 'party', 'supplier', 'vendor', 'material', 'item', 'product',
    'qty', 'quantity', 'purchase', 'amount', 'value', 'total', 'gross',
    'sgst', 'cgst', 'igst', 'gst', 'gstin', 'uin', 'invoice', 'bill', 'voucher', 'ref',
]
INOUT_KEYWORDS   = [
    'date', 'material', 'item', 'inward', 'outward', 'receipt', 'issue',
    'balance', 'stock', 'qty', 'quantity', 'in', 'out', 'closing', 'opening',
]
TOTAL_KEYWORDS   = ['total', 'grand total', 'sub total', 'subtotal', 'net total', 'closing', 'opening']



# ─────────────────────────────────────────────────────────────
# SHARED HELPERS
# ─────────────────────────────────────────────────────────────

def _detect_header_row(df_raw, keywords, min_hits=3):
    """Scan first 25 rows to find the true table header row index."""
    best_row, best_score = 0, 0
    for i in range(min(25, len(df_raw))):
        row_vals = [str(x).strip().lower() for x in df_raw.iloc[i].values if pd.notna(x)]
        score = sum(1 for cell in row_vals for kw in keywords if kw in cell)
        if score > best_score:
            best_score = score
            best_row = i
    return best_row if best_score >= min_hits else 0


def _normalize_col(name):
    return (str(name).strip().lower()
            .replace(" ", "_").replace(".", "_")
            .replace("/", "_").replace("-", "_")
            .replace("(", "").replace(")", ""))


def _deduplicate_columns(df):
    """Resolve duplicate column names by keeping the first and tagging the rest for removal."""
    seen = {}
    new_cols = []
    for col in df.columns:
        if col not in seen:
            seen[col] = 0
            new_cols.append(col)
        else:
            seen[col] += 1
            new_cols.append(f"{col}__dup_{seen[col]}")
    df.columns = new_cols
    df = df.drop(columns=[c for c in df.columns if '__dup_' in c])
    return df


def _rename_by_map(df, col_map):
    """Fuzzy-rename columns using a mapping dict {keyword: standard_name}."""
    rename = {}
    for col in df.columns:
        if col in col_map:
            rename[col] = col_map[col]
        else:
            for key, std in col_map.items():
                if key in col and col not in rename:
                    rename[col] = std
                    break
    return df.rename(columns=rename)


def _remove_total_rows(df):
    mask = df.apply(
        lambda row: any(kw in str(v).lower() for v in row.values for kw in TOTAL_KEYWORDS),
        axis=1
    )
    return df[~mask]


# ─────────────────────────────────────────────────────────────
# PDF EXTRACTION HELPER
# ─────────────────────────────────────────────────────────────

def _ingest_pdf_to_dataframe(file_bytes, keywords):
    """
    Robust PDF → DataFrame extractor.
    Strategy 1: pdfplumber extract_tables() — works great on Tally PDF table exports.
    Strategy 2: text line parsing — fallback for plaintext-style PDFs.
    Returns (df, error).
    """
    if not PDF_AVAILABLE:
        return None, "pdfplumber not installed. Run: pip install pdfplumber"
    try:
        all_tables = []
        all_text_lines = []

        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            if len(pdf.pages) == 0:
                return None, "Scanned PDFs not supported. Please upload Excel or text-based Tally export."

            for page in pdf.pages:
                # Strategy 1: extract structured tables
                tables = page.extract_tables()
                for table in tables:
                    if table and len(table) > 1:
                        all_tables.append(table)

                # Also collect raw text as fallback
                t = page.extract_text()
                if t:
                    all_text_lines.extend(t.splitlines())

        # ── Strategy 1: Use structured table data ──────────────────
        if all_tables:
            combined_rows = []
            header = None

            def _is_letterhead_row(row):
                """True if this row looks like a merged letterhead/address cell."""
                for cell in row:
                    if '\n' in cell:           # multiline = merged address block
                        return True
                    if len(cell) > 80:          # very long single cell = not a table header
                        return True
                non_empty = [c for c in row if c.strip()]
                if len(non_empty) < 2:          # only 1 cell filled = likely address
                    return True
                return False

            for table in all_tables:
                # Clean nulls
                table = [[str(c).strip().replace('\n', ' ') if c else '' for c in row] for row in table]
                table = [row for row in table if any(c for c in row)]

                if not table:
                    continue

                if header is None:
                    # Scan every row in this table to find the real header
                    for ri, row in enumerate(table):
                        if _is_letterhead_row(row):
                            continue  # skip address/letterhead blocks
                        row_str = ' '.join(row).lower()
                        row_score = sum(1 for kw in keywords if kw in row_str)
                        non_empty_cells = len([c for c in row if c.strip()])
                        # Accept if keywords match OR if it has 4+ clean cells (column-like)
                        if row_score >= 1 or non_empty_cells >= 4:
                            header = row
                            combined_rows.extend(table[ri + 1:])
                            break
                else:
                    # Already have header — just append data rows, skip letterhead rows
                    for row in table:
                        if not _is_letterhead_row(row):
                            combined_rows.append(row)

            if header and combined_rows:
                n = len(header)
                trimmed = [r[:n] + [''] * max(0, n - len(r)) for r in combined_rows]
                df = pd.DataFrame(trimmed, columns=header)
                df = df.replace('', pd.NA).dropna(how='all').reset_index(drop=True)
                if len(df) > 0:
                    return df, None


        # ── Strategy 2: Text line fallback ─────────────────────────
        lines = [l.strip() for l in all_text_lines if l.strip()]
        if len(lines) < 5:
            return None, "Scanned PDFs not supported. Please upload Excel or text-based Tally export."

        # Find best header line (min score = 1 now, much more permissive)
        best_idx, best_score = 0, 0
        for i, line in enumerate(lines[:40]):
            score = sum(1 for kw in keywords if kw in line.lower())
            if score > best_score:
                best_score = score
                best_idx = i

        # Brute-force fallback: if still no match, find the first line with 4+ separate tokens
        # (likely the table header even if none of our keywords matched)
        if best_score < 1:
            for i, line in enumerate(lines[:40]):
                parts = re.split(r'\s{2,}', line)
                if len(parts) >= 4:
                    best_idx = i
                    best_score = 1
                    break

        if best_score < 1:
            return None, "Unable to parse PDF structure. Please upload Tally Excel export."

        # Split into columns by 2+ consecutive spaces (Tally alignment)
        header_line = lines[best_idx]
        cols = [c.strip() for c in re.split(r'\s{2,}', header_line) if c.strip()]
        if len(cols) < 2:
            cols = header_line.split()

        data_rows = []
        for line in lines[best_idx + 1:]:
            parts = [c.strip() for c in re.split(r'\s{2,}', line) if c.strip()]
            if not parts:
                continue
            while len(parts) < len(cols):
                parts.append('')
            data_rows.append(parts[:len(cols)])

        if not data_rows:
            return None, "No data rows found in PDF."

        df = pd.DataFrame(data_rows, columns=cols)
        df = df.replace('', pd.NA).dropna(how='all').reset_index(drop=True)
        return df, None

    except Exception as e:
        return None, f"PDF read error: {str(e)}"




# ─────────────────────────────────────────────────────────────
# 1. SALES REGISTER INGESTION
# ─────────────────────────────────────────────────────────────

SALES_COL_MAP = {
    'date': 'date', 'invoice_date': 'date', 'voucher_date': 'date', 'bill_date': 'date',
    'party': 'customer', 'customer': 'customer', 'customer_name': 'customer',
    'particulars': 'customer', 'client': 'customer', 'buyer': 'customer', 'party_name': 'customer',
    'item': 'product', 'item_name': 'product', 'product': 'product', 'stock_item': 'product',
    'goods': 'product', 'description': 'product',
    'billed_qty': 'quantity', 'qty': 'quantity', 'quantity': 'quantity',
    'units': 'quantity', 'nos': 'quantity', 'pcs': 'quantity',
    # Financial fallbacks
    'gross_total': 'quantity', 'value': 'quantity', 'amount': 'quantity',
    'net_amount': 'quantity', 'invoice_value': 'quantity',
}


FINANCIAL_QTY_COLS = {'gross_total', 'grosstotal', 'value', 'net_amount', 'net_total',
                      'amount', 'invoice_value', 'taxable_value'}


def _clean_sales_df(df, original_cols_before_rename):
    """Shared sales cleaning logic. Returns (df, error, qty_label)."""
    df = _remove_total_rows(df)

    required = ['date', 'customer']
    missing = [c for c in required if c not in df.columns]
    if missing:
        return None, f"Could not find required columns: {missing}. Detected: {list(df.columns)}", None

    # Detect whether quantity is physical units or financial value
    qty_label = 'Quantity'
    financial_used = any(c in FINANCIAL_QTY_COLS for c in original_cols_before_rename)
    if financial_used and 'quantity' not in [c for c in original_cols_before_rename
                                              if c not in FINANCIAL_QTY_COLS]:
        qty_label = 'Invoice Value (₹)'

    # Fallback: if quantity still missing, pick first remaining column
    if 'quantity' not in df.columns:
        candidates = [c for c in df.columns if c not in ('date', 'customer', 'product')]
        if candidates:
            df = df.rename(columns={candidates[0]: 'quantity'})
            qty_label = 'Invoice Value (₹)'
        else:
            return None, "No numeric column found for order value.", None

    keep = [c for c in ['date', 'customer', 'product', 'quantity'] if c in df.columns]
    df = df[keep]

    df['date'] = pd.to_datetime(df['date'], errors='coerce', dayfirst=True)
    df['quantity'] = pd.to_numeric(df['quantity'].astype(str).str.replace(r'[^\d.\-]', '', regex=True), errors='coerce')

    df = df.dropna(subset=['date', 'customer', 'quantity'])
    df = df[df['quantity'] > 0]
    df['customer'] = df['customer'].astype(str).str.strip()
    df = df[df['customer'].str.len() > 1]
    return df.reset_index(drop=True), None, qty_label



def ingest_sales_excel(file):
    """Load a Sales Register from Excel. Returns (df, error, qty_label)."""
    try:
        raw = pd.read_excel(file, header=None, dtype=str)
        raw = raw.dropna(how='all').reset_index(drop=True)
        hdr = _detect_header_row(raw, SALES_KEYWORDS, min_hits=3)
        df = pd.read_excel(file, header=hdr, dtype=str)
        orig_cols = [_normalize_col(c) for c in df.columns]
        df.columns = orig_cols
        df = _rename_by_map(df, SALES_COL_MAP)
        df = _deduplicate_columns(df)
        return _clean_sales_df(df, orig_cols)
    except Exception as e:
        return None, str(e), None



def ingest_sales_pdf(file_bytes):
    """Load a Sales Register from a text-based Tally PDF. Returns (df, error, qty_label)."""
    try:
        df_raw, err = _ingest_pdf_to_dataframe(file_bytes, SALES_KEYWORDS)
        if err:
            return None, err, None
        orig_cols = [_normalize_col(c) for c in df_raw.columns]
        df_raw.columns = orig_cols
        df_raw = _rename_by_map(df_raw, SALES_COL_MAP)
        df_raw = _deduplicate_columns(df_raw)
        return _clean_sales_df(df_raw, orig_cols)
    except Exception as e:
        return None, str(e), None




# ─────────────────────────────────────────────────────────────
# 2. ANCHOR CUSTOMER IDENTIFICATION & PREDICTION
# ─────────────────────────────────────────────────────────────

def get_anchor_customers(df, top_n=5):
    """Return top N customers by total quantity. Returns (df, error)."""
    try:
        agg = df.groupby('customer').agg(
            total_quantity=('quantity', 'sum'),
            order_count=('quantity', 'count')
        ).reset_index().sort_values('total_quantity', ascending=False).reset_index(drop=True)
        total = agg['total_quantity'].sum()
        agg['contribution_pct'] = (agg['total_quantity'] / total * 100).round(2) if total > 0 else 0.0
        return agg.head(top_n), None
    except Exception as e:
        return pd.DataFrame(), str(e)


def predict_reorder(df, anchor_customers):
    """Predict reorder window per anchor customer (min 3 orders). Returns (df, error)."""
    try:
        results = []
        for _, row in anchor_customers.iterrows():
            cust = row['customer']
            orders = df[df['customer'] == cust].sort_values('date').drop_duplicates('date')
            if len(orders) < 2:
                continue
            intervals = orders['date'].diff().dt.days.dropna()
            mu = intervals.mean()
            sigma = intervals.std()
            if mu <= 0 or pd.isna(mu):
                continue
            last_order = orders['date'].max()
            predicted = last_order + pd.Timedelta(days=mu)
            half_sig = (sigma * 0.5) if not pd.isna(sigma) else 0
            confidence = max(0.0, round(100 - (sigma / mu * 100), 1)) if not pd.isna(sigma) else 50.0
            results.append({
                'Customer': cust,
                'Last Order': last_order.strftime('%d %b %Y'),
                'Avg Interval (Days)': round(mu, 1),
                'Predicted Next Order': predicted.strftime('%d %b %Y'),
                'Window': f"{(predicted - pd.Timedelta(days=half_sig)).strftime('%d %b')} – {(predicted + pd.Timedelta(days=half_sig)).strftime('%d %b %Y')}",
                'Confidence %': confidence,
            })
        return pd.DataFrame(results), None
    except Exception as e:
        return pd.DataFrame(), str(e)


# ─────────────────────────────────────────────────────────────
# 3. INWARD / OUTWARD REGISTER
# ─────────────────────────────────────────────────────────────

INOUT_COL_MAP = {
    'date': 'date', 'invoice_date': 'date',
    'material': 'material', 'item': 'material', 'item_name': 'material',
    'product': 'material', 'stock_item': 'material', 'goods': 'material',
    'inward': 'inward', 'receipt': 'inward', 'received': 'inward', 'in': 'inward',
    'purchase': 'inward', 'qty_in': 'inward',
    'outward': 'outward', 'issue': 'outward', 'issued': 'outward', 'out': 'outward',
    'sales': 'outward', 'qty_out': 'outward', 'dispatch': 'outward',
    'balance': 'balance', 'current_stock': 'balance', 'closing': 'balance',
}


def _clean_inout_df(df):
    df = _remove_total_rows(df)
    if 'material' not in df.columns:
        candidates = [c for c in df.columns if c not in ('date', 'inward', 'outward', 'balance')]
        if candidates:
            df = df.rename(columns={candidates[0]: 'material'})
    if 'inward' not in df.columns:
        df['inward'] = 0
    if 'outward' not in df.columns:
        df['outward'] = 0
    for col in ['inward', 'outward']:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(r'[^\d.\-]', '', regex=True), errors='coerce').fillna(0)
    df['material'] = df['material'].astype(str).str.strip()
    df = df[df['material'].str.len() > 1]
    return df.reset_index(drop=True), None


def ingest_inout_excel(file):
    """Load Inward/Outward register from Excel."""
    try:
        raw = pd.read_excel(file, header=None, dtype=str).dropna(how='all').reset_index(drop=True)
        hdr = _detect_header_row(raw, INOUT_KEYWORDS, min_hits=2)
        df = pd.read_excel(file, header=hdr, dtype=str)
        df.columns = [_normalize_col(c) for c in df.columns]
        df = _rename_by_map(df, INOUT_COL_MAP)
        df = _deduplicate_columns(df)
        return _clean_inout_df(df)
    except Exception as e:
        return None, str(e)


def ingest_inout_pdf(file_bytes):
    """Load Inward/Outward register from PDF."""
    try:
        df_raw, err = _ingest_pdf_to_dataframe(file_bytes, INOUT_KEYWORDS)
        if err:
            return None, err
        df_raw.columns = [_normalize_col(c) for c in df_raw.columns]
        df_raw = _rename_by_map(df_raw, INOUT_COL_MAP)
        df_raw = _deduplicate_columns(df_raw)
        return _clean_inout_df(df_raw)
    except Exception as e:
        return None, str(e)



def compute_stock(inout_df):
    """Compute current stock per material."""
    try:
        agg = inout_df.groupby('material').agg(
            total_inward=('inward', 'sum'),
            total_outward=('outward', 'sum')
        ).reset_index()
        agg['current_stock'] = agg['total_inward'] - agg['total_outward']
        agg = agg.sort_values('current_stock', ascending=True).reset_index(drop=True)
        return agg, None
    except Exception as e:
        return pd.DataFrame(), str(e)


# ─────────────────────────────────────────────────────────────
# 4. PURCHASE REGISTER
# ─────────────────────────────────────────────────────────────

PURCHASE_COL_MAP = {
    'date': 'date', 'invoice_date': 'date', 'purchase_date': 'date',
    'supplier': 'supplier', 'party': 'supplier', 'vendor': 'supplier',
    'material': 'material', 'item': 'material', 'item_name': 'material',
    'product': 'material', 'stock_item': 'material', 'goods': 'material',
    'qty': 'quantity', 'quantity': 'quantity', 'units': 'quantity',
    'nos': 'quantity', 'pcs': 'quantity', 'amount': 'quantity',
    'value': 'quantity', 'gross_total': 'quantity',
}


def _clean_purchase_df(df):
    df = _remove_total_rows(df)
    if 'material' not in df.columns:
        candidates = [c for c in df.columns if c not in ('date', 'supplier', 'quantity')]
        if candidates:
            df = df.rename(columns={candidates[0]: 'material'})
    if 'quantity' not in df.columns:
        candidates = [c for c in df.columns if c not in ('date', 'supplier', 'material')]
        if candidates:
            df = df.rename(columns={candidates[0]: 'quantity'})
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce', dayfirst=True)
    if 'quantity' in df.columns:
        df['quantity'] = pd.to_numeric(df['quantity'].astype(str).str.replace(r'[^\d.\-]', '', regex=True), errors='coerce').fillna(0)
    keep = [c for c in ['date', 'supplier', 'material', 'quantity'] if c in df.columns]
    df = df[keep]
    df['material'] = df['material'].astype(str).str.strip()
    df = df[df['material'].str.len() > 1]
    return df.reset_index(drop=True), None


def ingest_purchase_excel(file):
    """Load Purchase register from Excel."""
    try:
        raw = pd.read_excel(file, header=None, dtype=str).dropna(how='all').reset_index(drop=True)
        hdr = _detect_header_row(raw, PURCHASE_KEYWORDS, min_hits=2)
        df = pd.read_excel(file, header=hdr, dtype=str)
        df.columns = [_normalize_col(c) for c in df.columns]
        df = _rename_by_map(df, PURCHASE_COL_MAP)
        df = _deduplicate_columns(df)
        return _clean_purchase_df(df)
    except Exception as e:
        return None, str(e)


def ingest_purchase_pdf(file_bytes):
    """Load Purchase register from PDF."""
    try:
        df_raw, err = _ingest_pdf_to_dataframe(file_bytes, PURCHASE_KEYWORDS)
        if err:
            return None, err
        df_raw.columns = [_normalize_col(c) for c in df_raw.columns]
        df_raw = _rename_by_map(df_raw, PURCHASE_COL_MAP)
        df_raw = _deduplicate_columns(df_raw)
        return _clean_purchase_df(df_raw)
    except Exception as e:
        return None, str(e)



def compute_purchase_summary(purchase_df):
    """Summarise total purchase and recent 60-day purchase per material."""
    try:
        total_agg = purchase_df.groupby('material')['quantity'].sum().reset_index()
        total_agg.columns = ['Material', 'Total Purchased']

        if 'date' in purchase_df.columns:
            cutoff = purchase_df['date'].max() - pd.Timedelta(days=60)
            recent = purchase_df[purchase_df['date'] >= cutoff]
            recent_agg = recent.groupby('material')['quantity'].sum().reset_index()
            recent_agg.columns = ['Material', 'Last 60 Days']
            result = pd.merge(total_agg, recent_agg, on='Material', how='left').fillna(0)
        else:
            result = total_agg

        return result.sort_values('Total Purchased', ascending=False).reset_index(drop=True), None
    except Exception as e:
        return pd.DataFrame(), str(e)


# ─────────────────────────────────────────────────────────────
# 5. MATERIAL OUTLOOK
# ─────────────────────────────────────────────────────────────

def material_outlook(stock_df, predictions_df, sales_df):
    """
    Compare predicted demand vs current stock.
    Uses anchor customer avg order qty as predicted demand.
    Rule-based advisory: if predicted demand > current stock → Prepare/Buy Soon.
    """
    try:
        if stock_df is None or stock_df.empty:
            return pd.DataFrame(), "No stock data available."

        outlook = stock_df[['material', 'current_stock']].copy()

        # Estimate predicted demand: avg daily quantity from sales × avg reorder interval
        predicted_demand_total = 0
        if predictions_df is not None and not predictions_df.empty:
            predicted_demand_total = predictions_df['Avg Interval (Days)'].mean()

        if sales_df is not None and not sales_df.empty and 'product' in sales_df.columns:
            product_demand = sales_df.groupby('product')['quantity'].mean().reset_index()
            product_demand.columns = ['material', 'avg_order_qty']
            outlook = pd.merge(outlook, product_demand, on='material', how='left').fillna(0)
            outlook['projected_demand'] = outlook['avg_order_qty'] * (predicted_demand_total if predicted_demand_total > 0 else 1)
        else:
            outlook['projected_demand'] = 0

        def advisory(row):
            if row['projected_demand'] > row['current_stock']:
                return '🔴 Prepare / Buy Soon'
            elif row['current_stock'] <= 0:
                return '🔴 Stock Out'
            else:
                return '🟡 Monitor'

        outlook['Advisory'] = outlook.apply(advisory, axis=1)
        outlook = outlook.sort_values('current_stock', ascending=True).reset_index(drop=True)
        outlook.columns = [c.replace('_', ' ').title() for c in outlook.columns]
        return outlook, None
    except Exception as e:
        return pd.DataFrame(), str(e)