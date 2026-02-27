"""
Inventory Intelligence Engine – Data Processor
Strict, rule-based, deterministic field mapping engine.
No AI. No APIs. No PDF. Excel-only.
"""

import re
import pandas as pd
import numpy as np


# ─────────────────────────────────────────────────────────────
# TALLY VISUAL EXCEL INVOICE PARSER
# These are PDF-converted XLS files with NO tabular headers.
# Structure: Row 0 = "GST INVOICE", Buyer section, then item table.
# ─────────────────────────────────────────────────────────────

SKIP_PRODUCT_KW = {
    'sgst', 'cgst', 'igst', 'vat', 'tax', 'total', 'grand total',
    'amount in words', 'amount chargeable', 'taxable', 'declaration',
    'freight', 'packing', 'rounding', 'discount', 'advance', 'tds'
}

UNITS = {'nos', 'pcs', 'kg', 'mtr', 'mts', 'set', 'box', 'ltr', 'unit', 'piece', 'pieces'}


def _is_visual_tally_xls(raw_df):
    """Return True if the file looks like a visual Tally GST invoice."""
    first_rows_text = ' '.join(
        str(v).lower() for v in raw_df.iloc[:5].values.flatten() if str(v).strip()
    )
    return 'gst invoice' in first_rows_text or 'tax invoice' in first_rows_text


def _cell(v):
    s = str(v).strip() if v is not None else ''
    if s.lower() in ('nan', 'none', ''):
        return ''
    return s


def _clean_num(s):
    try:
        return float(re.sub(r'[^\d.\-]', '', str(s)))
    except Exception:
        return None


def _parse_one_tally_invoice(rows):
    """
    Parse one Tally GST invoice block.
    Each 'row' is a list of cell values (strings).
    Cells may contain embedded newlines from merged PDF cells.

    KEY INSIGHT from diagnostic:
    - Row 1: mega-cell containing seller info + Buyer section + customer name
      + Invoice No. cell + Dated cell (all separated by \n within same cells)
    - Row 10: header row ['Sl No.', 'Description of Goods', 'HSN/SAC', 'Quantity', 'Rate', 'per', 'Amount']
    - Row 11: product row, amount cell may have SGST/CGST amounts stacked with \n
    """
    date_val      = None
    invoice_no    = None
    customer      = None
    result_rows   = []
    prod_hdr_row  = None
    col_map       = {}  # {field: column_index}

    PROD_HDR_KW = {'description', 'goods', 'hsn', 'sac', 'qty', 'quantity', 'amount', 'rate'}
    SKIP = {
        'sgst', 'cgst', 'igst', 'vat', 'output tax', 'total', 'grand total',
        'amount in words', 'amount chargeable', 'taxable', 'declaration',
        'freight', 'packing', 'rounding', 'discount', 'advance', 'tds'
    }

    for r_idx, row in enumerate(rows):
        # Flatten each cell by splitting on newlines - this is critical
        # Each cell may contain multi-line content from merged PDF cells
        flat_lines = []
        for cell in row:
            for line in str(cell).split('\n'):
                l = line.strip()
                if l and l.lower() not in ('nan', 'none'):
                    flat_lines.append(l)

        full_text = ' '.join(flat_lines)
        full_low  = full_text.lower()

        # ── Extract Invoice Number ─────────────────────────
        if not invoice_no:
            for cell in row:
                m = re.search(
                    r'Invoice\s*No\.?\s*[\n\s:]*([A-Z0-9/\-]+)',
                    str(cell), re.IGNORECASE
                )
                if m:
                    candidate = m.group(1).strip()
                    if len(candidate) > 2 and candidate.upper() != 'DATED':
                        invoice_no = candidate
                        break

        # ── Extract Date ───────────────────────────────────
        if not date_val:
            for cell in row:
                dm = re.search(
                    r'(?:Dated|Date)[:\s\n]*([\d]{1,2}[-][A-Za-z]{3}[-][\d]{2,4})',
                    str(cell), re.IGNORECASE
                )
                if dm:
                    date_val = dm.group(1).strip()
                    break
            if not date_val:
                dm = re.search(r'\b(\d{1,2}[-][A-Za-z]{3}[-]\d{2,4})\b', full_text)
                if dm:
                    date_val = dm.group(1)

        # ── Extract Customer (Buyer section) ───────────────
        if not customer:
            for cell in row:
                cell_lines = [l.strip() for l in str(cell).split('\n') if l.strip()]
                buyer_idx = None
                for li, line in enumerate(cell_lines):
                    if re.search(r'buyer\s*(\(bill\s*to\))?', line, re.IGNORECASE):
                        buyer_idx = li
                        break
                if buyer_idx is not None:
                    # Next non-empty line after 'Buyer (Bill to)' is the customer
                    for next_line in cell_lines[buyer_idx + 1:]:
                        nl = next_line.strip()
                        nll = nl.lower()
                        if len(nl) < 5: continue
                        if re.match(r'^[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]', nl): continue  # GSTIN
                        if any(kw in nll for kw in ('gstin', 'state', 'contact',
                                                     'e-mail', 'email', 'invoice',
                                                     'dispatch', 'dated')): continue
                        if re.match(r'^\d+[/,]', nl): continue  # address
                        customer = nl  # keep original casing
                        break

        # ── Find Product Table Header ──────────────────────
        if prod_hdr_row is None:
            hits = sum(1 for kw in PROD_HDR_KW if kw in full_low)
            if hits >= 3:
                prod_hdr_row = r_idx
                # Map column index → field name
                for ci, cell in enumerate(row):
                    cl = str(cell).lower().strip()
                    if 'description' in cl or 'goods' in cl:
                        col_map['desc'] = ci
                    elif 'hsn' in cl or 'sac' in cl:
                        col_map['hsn'] = ci
                    elif 'qty' in cl or 'quantity' in cl:
                        col_map['qty'] = ci
                    elif 'rate' in cl:
                        col_map['rate'] = ci
                    elif 'per' == cl.strip():
                        col_map['per'] = ci
                    elif 'amount' in cl or 'value' in cl:
                        col_map['amt'] = ci
                continue

        # ── Extract Product Rows ───────────────────────────
        if prod_hdr_row is not None and r_idx > prod_hdr_row:
            # Build first-line-only text for filtering
            # CRITICAL: Do NOT use full multi-line cell content for skip check
            # because product cells have SGST/CGST embedded as sub-lines
            first_line_text = ' '.join(
                str(cell).split('\n')[0].strip()
                for cell in row
                if str(cell).strip() and str(cell).lower() not in ('nan','none')
            ).lower()

            # Stop at Total row
            if re.match(r'^\s*total\b', first_line_text):
                break

            # Skip rows whose FIRST line is a tax/meta keyword (not the product description)
            if any(kw == first_line_text.strip() for kw in SKIP):
                continue

            # Get row cells as a fixed-length padded list
            padded = list(row) + [''] * 12

            # Description from mapped column or first text cell
            desc = ''
            if 'desc' in col_map:
                raw_desc = str(padded[col_map['desc']]).strip()
                # Take only first line (ignore SGST/CGST lines embedded in cell)
                desc = raw_desc.split('\n')[0].strip()
            if not desc:
                for ci, cell in enumerate(padded[:8]):
                    c = str(cell).split('\n')[0].strip()
                    if ci == 0 and re.match(r'^\d+$', c): continue
                    if re.match(r'^[\d,\.]+$', c): continue
                    if re.match(r'^(NOS|PCS|KG|MTR|MTS|SET|BOX|LTR|UNIT)\.?$', c, re.IGNORECASE): continue
                    if len(c) > 3:
                        desc = c
                        break

            if not desc or len(desc) < 3:
                continue
            if any(kw in desc.lower() for kw in SKIP):
                continue

            # Quantity
            qty = None
            unit = ''
            qty_cell = str(padded[col_map.get('qty', 3)]) if len(padded) > col_map.get('qty', 3) else ''
            qty_m = re.search(
                r'(\d+(?:\.\d+)?)\s*(NOS|PCS|KG|MTR|MTS|SET|BOX|LTR|UNIT|PIECE|PIECES)?\.?',
                qty_cell.split('\n')[0], re.IGNORECASE
            )
            if qty_m:
                qty = float(qty_m.group(1))
                if qty_m.group(2):
                    unit = qty_m.group(2).upper()

            # Rate
            rate = None
            rate_cell = str(padded[col_map.get('rate', 4)]) if len(padded) > col_map.get('rate', 4) else ''
            rate_m = re.search(r'([\d,]+\.\d{2})', rate_cell.split('\n')[0])
            if rate_m:
                rate = float(rate_m.group(1).replace(',', ''))

            # Amount: use mapped column, take FIRST decimal value (ignore SGST/CGST embedded below)
            amount = None
            amt_cell = str(padded[col_map.get('amt', 6)]) if len(padded) > col_map.get('amt', 6) else ''
            amt_first_line = amt_cell.split('\n')[0]  # first line only = product amount, not tax
            amt_m = re.search(r'([\d,]+\.\d{2})', amt_first_line)
            if amt_m:
                amount = float(amt_m.group(1).replace(',', ''))

            if not amount or amount <= 0:
                continue

            # Per unit
            if not unit:
                per_cell = str(padded[col_map.get('per', 5)]).strip().split('\n')[0]
                if re.match(r'^(NOS|PCS|KG|MTR|MTS|SET|BOX|LTR|UNIT)\.?$', per_cell, re.IGNORECASE):
                    unit = per_cell.upper().rstrip('.')

            result_rows.append({
                'customer':   customer or 'UNKNOWN',
                'date':       date_val,
                'invoice_no': invoice_no or 'UNKNOWN',
                'product':    desc,
                'quantity':   qty,
                'unit':       unit,
                'rate':       rate,
                'amount':     amount,
            })

    return result_rows


def parse_tally_visual_excel(file):
    """
    Parse a Tally GST invoice file that was converted from PDF to XLS.
    Handles multiple invoices per file (merged invoices).
    Returns (df, error).
    """
    date_val   = None
    invoice_no = None
    customer   = None
    buyer_found = False
    prod_hdr_row = None
    result_rows = []

    PROD_HDR_KW = {'description', 'goods', 'hsn', 'sac', 'qty', 'quantity', 'amount', 'rate', 'sl'}

    for r_idx, row in enumerate(rows):
        row_text = ' '.join(row).strip()
        row_low  = row_text.lower()

        # ── Invoice metadata ───────────────────────────────
        if 'invoice no' in row_low or 'voucher no' in row_low:
            # Look for the invoice number pattern in same combined cell text
            m = re.search(r'(?:invoice no\.?|voucher no\.?)[^\w]*([A-Z0-9/\-]+)', row_text, re.IGNORECASE)
            if m and not invoice_no:
                invoice_no = m.group(1).strip()
            # Date is often in the same row text after "Dated"
            dm = re.search(r'[Dd]ated[^\d]*(\d{1,2}[-/][A-Za-z0-9]{2,3}[-/]\d{2,4})', row_text)
            if dm and not date_val:
                date_val = dm.group(1).strip()

        # Standalone date row detection
        if not date_val:
            dm = re.search(r'\b(\d{1,2}[-][A-Za-z]{3}[-]\d{2,4})\b', row_text)
            if dm:
                date_val = dm.group(1)

        # ── Buyer section ──────────────────────────────────
        if 'buyer' in row_low and ('bill to' in row_low or 'bill' in row_low):
            buyer_found = True
            continue

        if buyer_found and not customer:
            # First non-empty, non-address, non-keyword line after "Buyer"
            for cell in row:
                c = cell.strip()
                cl = c.lower()
                if not c or len(c) < 5:
                    continue
                if re.match(r'^[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]', c):
                    continue  # GSTIN
                if any(kw in cl for kw in ('gstin', 'state', 'contact', 'code',
                                            'e-mail', 'email', 'invoice', 'dated')):
                    continue
                if re.match(r'^\d', c) and ('midc' in cl or 'pune' in cl or
                                              'satara' in cl or 'road' in cl or 'near' in cl):
                    continue  # address line
                customer = c.title()
                buyer_found = False
                break

        # ── Product table header ───────────────────────────
        hits = sum(1 for kw in PROD_HDR_KW if kw in row_low)
        if hits >= 3 and prod_hdr_row is None:
            prod_hdr_row = r_idx
            # Find column positions for qty, rate, amount
            # We'll use positional parsing instead - qty is always 4th main col
            continue

        # ── Product rows ───────────────────────────────────
        if prod_hdr_row is not None and r_idx > prod_hdr_row:
            # Stop at "Total" row
            if re.match(r'^\s*total\b', row_low):
                break

            # Gather all non-empty cells
            cells = [c for c in row if c.strip()]
            if not cells:
                continue

            row_low_flat = ' '.join(cells).lower()
            if any(kw in row_low_flat for kw in SKIP_PRODUCT_KW):
                continue

            # Find amount: last decimal number in row
            all_nums = re.findall(r'[\d,]+\.\d{2}', ' '.join(cells))
            if not all_nums:
                continue
            amount = _clean_num(all_nums[-1])
            if not amount or amount <= 0:
                continue

            # Find rate: second-to-last decimal number
            rate = _clean_num(all_nums[-2]) if len(all_nums) >= 2 else None

            # Find quantity: integer or decimal followed by unit
            qty = None
            unit = ''
            qty_m = re.search(
                r'(\d+(?:\.\d+)?)\s*(NOS|PCS|KG|MTR|MTS|SET|BOX|LTR|UNIT|PIECE|PIECES)\.?',
                ' '.join(cells), re.IGNORECASE
            )
            if qty_m:
                qty  = float(qty_m.group(1))
                unit = qty_m.group(2).upper()
            else:
                # No unit found, try standalone integer
                int_m = re.search(r'\b(\d+)\b', ' '.join(cells[1:4]))
                if int_m:
                    qty = float(int_m.group(1))

            # Description: first non-numeric cell (skip row number at position 0)
            desc = ''
            for ci, cell in enumerate(cells):
                c = cell.strip()
                if ci == 0 and re.match(r'^\d+$', c):
                    continue  # skip serial number
                if re.match(r'^[\d,\.]+$', c):
                    continue
                if re.match(r'^(NOS|PCS|KG|MTR|MTS|SET|BOX|LTR|UNIT)\.?$', c, re.IGNORECASE):
                    continue
                if len(c) > 3:
                    desc = c
                    break

            if not desc or any(kw in desc.lower() for kw in SKIP_PRODUCT_KW):
                continue

            result_rows.append({
                'customer':   customer or 'UNKNOWN',
                'date':       date_val,
                'invoice_no': invoice_no or 'UNKNOWN',
                'product':    desc,
                'quantity':   qty,
                'unit':       unit,
                'rate':       rate,
                'amount':     amount,
            })

    return result_rows


def parse_tally_visual_excel(file_bytes, fname='file'):
    """
    Parse a Tally GST invoice file (PDF-to-XLS converted).
    Accepts raw bytes so that Streamlit UploadedFile seek issues don't occur.
    Returns (df, error).
    """
    import io as _io
    raw = None

    # Try openpyxl first (for .xlsx), then xlrd (for .xls), then HTML
    for engine in [None, 'xlrd']:
        try:
            kwargs = {'header': None, 'dtype': str}
            if engine:
                kwargs['engine'] = engine
            raw = pd.read_excel(_io.BytesIO(file_bytes), **kwargs).fillna('')
            break
        except Exception:
            continue

    if raw is None:
        try:
            tables = pd.read_html(_io.BytesIO(file_bytes))
            raw = pd.concat(tables, ignore_index=True).fillna('').astype(str)
        except Exception as e:
            return pd.DataFrame(), f"{fname}: Cannot open - {e}"

    # Convert each row to list of cell values
    all_rows_as_lists = []
    for _, row in raw.iterrows():
        cells = [_cell(v) for v in row.values]
        all_rows_as_lists.append(cells)

    # Find invoice block boundaries
    start_indices = []
    for i, row in enumerate(all_rows_as_lists):
        joined = ' '.join(row).lower()
        if 'gst invoice' in joined or 'tax invoice' in joined:
            start_indices.append(i)

    if not start_indices:
        start_indices = [0]

    end_indices = start_indices[1:] + [len(all_rows_as_lists)]

    all_results = []
    for start, end in zip(start_indices, end_indices):
        invoice_rows = all_rows_as_lists[start:end]
        items = _parse_one_tally_invoice(invoice_rows)
        all_results.extend(items)

    if not all_results:
        return pd.DataFrame(), (
            f"{fname}: No product rows found. "
            "Ensure the file is a Tally GST Invoice converted from PDF."
        )

    df = pd.DataFrame(all_results)
    df['date']     = pd.to_datetime(df['date'], errors='coerce', dayfirst=True)
    df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')
    df['amount']   = pd.to_numeric(df['amount'],   errors='coerce')
    df['rate']     = pd.to_numeric(df['rate'],     errors='coerce')
    df = df.dropna(subset=['amount']).reset_index(drop=True)
    return df, None






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
    Reads each file's bytes ONCE to avoid Streamlit UploadedFile seek issues.
    Returns (df, errors) - df has columns: date, customer, product, quantity, rate.
    """
    import io as _io
    if not isinstance(files, list):
        files = [files]

    frames, errors = [], []
    for f in files:
        fname = getattr(f, 'name', str(f))

        # Read ALL bytes upfront - avoids any seek/position issues with Streamlit uploads
        try:
            file_bytes = f.read() if hasattr(f, 'read') else open(f, 'rb').read()
        except Exception as e:
            errors.append(f"{fname}: Cannot read file bytes - {e}")
            continue

        # Peek to detect if this is a visual Tally XLS
        try:
            raw_peek = pd.read_excel(_io.BytesIO(file_bytes), header=None, dtype=str, nrows=5).fillna('')
        except Exception:
            try:
                raw_peek = pd.read_html(_io.BytesIO(file_bytes))[0].fillna('').astype(str)
            except Exception as e:
                errors.append(f"{fname}: Cannot parse file - {e}")
                continue

        if _is_visual_tally_xls(raw_peek):
            # Visual Tally invoice (PDF-converted) - pass raw bytes to dedicated parser
            df_out, err = parse_tally_visual_excel(file_bytes, fname=fname)
            if err:
                errors.append(err)
            elif df_out is not None and not df_out.empty:
                df_out['source'] = fname
                frames.append(df_out)
            continue

        # Structured tabular Excel - use strict column mapping
        df_raw, err = _read_excel_smart(_io.BytesIO(file_bytes), SALES_CONTEXT_KW, min_kw=2)
        if err:
            errors.append(f"{fname}: {err}")
            continue

        mapped = _detect_columns(df_raw, SALES_FIELD_RULES)
        required = ['date', 'customer', 'quantity']
        missing = [r for r in required if r not in mapped]
        if missing:
            col_list = ', '.join(f'"{m}"' for m in missing)
            errors.append(f"{fname}: Required column(s) not detected: {col_list}.")
            continue

        rename = {v: k for k, v in mapped.items()}
        subset = df_raw[[v for v in mapped.values()]].rename(columns=rename)
        if 'product' not in subset.columns: subset['product'] = 'UNSPECIFIED'
        if 'rate' not in subset.columns: subset['rate'] = pd.NA

        subset = _remove_totals(subset, ['customer', 'product'])
        subset['date']     = pd.to_datetime(subset['date'], errors='coerce', dayfirst=True)
        subset['quantity'] = _clean_numeric(subset['quantity'])
        subset['rate']     = _clean_numeric(subset['rate'])
        subset = subset.dropna(subset=['date', 'customer', 'quantity'])
        subset = subset[subset['quantity'] > 0]
        subset['customer'] = subset['customer'].astype(str).str.strip().str.title()
        subset['product']  = subset['product'].astype(str).str.strip()
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
    Returns (df, errors) - df has columns: date, supplier, material, quantity, rate.
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
    Returns (df, error) - df has columns: material, inward, outward.
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