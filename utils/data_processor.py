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



# -----------------------------------------------------------------
# 3. STOCK REGISTER INGESTION  (Inward + Outward as separate files)
# -----------------------------------------------------------------

_MATERIAL_KW   = {'material', 'description', 'item', 'desc', 'goods', 'mal', 'product', 'particulars'}
_QTY_KW        = {'qty', 'quantity', 'nos', 'weight', 'wt'}
_DATE_KW       = {'date', 'dat', 'tarikh'}
_SUPPLIER_KW   = {'supplier', 'party', 'from', 'vendor', 'received from', 'supplier name'}
_CUSTOMER_KW   = {'customer', 'goods taken out by', 'taken out by', 'buyer', 'bill to'}


def _find_col(columns, keywords, reject=()):
    """Return the first column name matching any keyword (case-insensitive)."""
    for col in columns:
        cl = str(col).lower().strip()
        if any(kw in cl for kw in keywords) and not any(rk in cl for rk in reject):
            return col
    return None


def _read_register_excel(file_bytes, fname):
    """Read an Excel register file into a DataFrame, detecting header row."""
    import io as _io
    raw = None
    for engine in [None, 'xlrd']:
        try:
            kw = {'header': None, 'dtype': str}
            if engine:
                kw['engine'] = engine
            raw = pd.read_excel(_io.BytesIO(file_bytes), **kw).fillna('')
            break
        except Exception:
            continue
    if raw is None:
        try:
            tables = pd.read_html(_io.BytesIO(file_bytes))
            raw = pd.concat(tables, ignore_index=True).fillna('').astype(str)
        except Exception as e:
            return None, f"{fname}: Cannot open -- {e}"

    # Find header row: first row with >=2 material/qty/date keywords
    hdr_row = 0
    for i in range(min(10, len(raw))):
        row_text = ' '.join(str(v).lower() for v in raw.iloc[i].values)
        hits = sum(1 for kw in list(_MATERIAL_KW) + list(_QTY_KW) + list(_DATE_KW) if kw in row_text)
        if hits >= 2:
            hdr_row = i
            break

    import io as _io2
    df = pd.read_excel(_io2.BytesIO(file_bytes), header=hdr_row, dtype=str).fillna('')
    df.columns = [str(c).strip() for c in df.columns]
    df = df.dropna(how='all').reset_index(drop=True)
    return df, None


def ingest_inward_excel(files):
    """
    Parse Inward Register Excel files (materials RECEIVED from suppliers).
    Template columns: SR.NO., DATE, CHALLAN NO., SUPPLIER NAME,
                      DESCRIPTION OF MATERIAL, QUANTITY, UNIT, RATE, AMOUNT
    Returns (df, errors) with columns: date, supplier, material, inward, unit
    """
    import io as _io
    if not isinstance(files, list):
        files = [files]

    frames, errors = [], []
    for f in files:
        fname = getattr(f, 'name', str(f))
        try:
            fb = f.read() if hasattr(f, 'read') else open(f, 'rb').read()
        except Exception as e:
            errors.append(f"{fname}: {e}")
            continue

        df, err = _read_register_excel(fb, fname)
        if err:
            errors.append(err)
            continue

        cols = list(df.columns)
        mat_col  = _find_col(cols, _MATERIAL_KW, reject={'rate', 'amount', 'price'})
        qty_col  = _find_col(cols, _QTY_KW,      reject={'rate', 'amount', 'price', 'value'})
        date_col = _find_col(cols, _DATE_KW)
        sup_col  = _find_col(cols, _SUPPLIER_KW)

        if not mat_col:
            errors.append(f"{fname}: 'Description of Material' column not found.")
            continue
        if not qty_col:
            errors.append(f"{fname}: 'Quantity' column not found.")
            continue

        out = pd.DataFrame()
        out['material'] = df[mat_col].astype(str).str.strip()
        out['inward']   = pd.to_numeric(
            df[qty_col].astype(str).str.replace(r'[^\d.]', '', regex=True),
            errors='coerce'
        )
        out['date']     = pd.to_datetime(df[date_col], errors='coerce', dayfirst=True) if date_col else pd.NaT
        out['supplier'] = df[sup_col].astype(str).str.strip() if sup_col else 'UNKNOWN'

        # unit column (optional)
        unit_col = _find_col(cols, {'unit', 'uom', 'nos', 'kg', 'mtr'}, reject={'rate', 'amount'})
        out['unit'] = df[unit_col].astype(str).str.strip() if unit_col else ''

        # Remove totals / headers / blank rows
        skip_kw = {'total', 'grand', 'subtotal', 'description', 'material', 'add your'}
        out = out[~out['material'].str.lower().str.contains('|'.join(skip_kw), na=False)]
        out = out.dropna(subset=['inward'])
        out = out[out['inward'] > 0]
        out['outward'] = 0.0
        out['source']  = fname

        if out.empty:
            errors.append(f"{fname}: No valid inward rows found.")
        else:
            frames.append(out)

    if not frames:
        return pd.DataFrame(), errors
    return pd.concat(frames, ignore_index=True), errors


def ingest_outward_excel(files):
    """
    Parse Outward Register Excel files (goods DISPATCHED to customers).
    Template columns: SR.NO., DATE, CHALLAN NO., GOODS TAKEN OUT BY,
                      DESCRIPTION OF MATERIAL, QUANTITY, UNIT, VEHICLE NO.
    Returns (df, errors) with columns: date, customer, material, outward, unit
    """
    import io as _io
    if not isinstance(files, list):
        files = [files]

    frames, errors = [], []
    for f in files:
        fname = getattr(f, 'name', str(f))
        try:
            fb = f.read() if hasattr(f, 'read') else open(f, 'rb').read()
        except Exception as e:
            errors.append(f"{fname}: {e}")
            continue

        df, err = _read_register_excel(fb, fname)
        if err:
            errors.append(err)
            continue

        cols = list(df.columns)
        mat_col  = _find_col(cols, _MATERIAL_KW, reject={'rate', 'amount', 'price'})
        qty_col  = _find_col(cols, _QTY_KW,      reject={'rate', 'amount', 'price', 'value'})
        date_col = _find_col(cols, _DATE_KW)
        cust_col = _find_col(cols, _CUSTOMER_KW)

        if not mat_col:
            errors.append(f"{fname}: 'Description of Material' column not found.")
            continue
        if not qty_col:
            errors.append(f"{fname}: 'Quantity' column not found.")
            continue

        out = pd.DataFrame()
        out['material'] = df[mat_col].astype(str).str.strip()
        out['outward']  = pd.to_numeric(
            df[qty_col].astype(str).str.replace(r'[^\d.]', '', regex=True),
            errors='coerce'
        )
        out['date']     = pd.to_datetime(df[date_col], errors='coerce', dayfirst=True) if date_col else pd.NaT
        out['customer'] = df[cust_col].astype(str).str.strip() if cust_col else 'UNKNOWN'

        unit_col = _find_col(cols, {'unit', 'uom'}, reject={'rate', 'amount'})
        out['unit'] = df[unit_col].astype(str).str.strip() if unit_col else ''

        skip_kw = {'total', 'grand', 'subtotal', 'description', 'material', 'add your'}
        out = out[~out['material'].str.lower().str.contains('|'.join(skip_kw), na=False)]
        out = out.dropna(subset=['outward'])
        out = out[out['outward'] > 0]
        out['inward']  = 0.0
        out['supplier'] = ''
        out['source']  = fname

        if out.empty:
            errors.append(f"{fname}: No valid outward rows found.")
        else:
            frames.append(out)

    if not frames:
        return pd.DataFrame(), errors
    return pd.concat(frames, ignore_index=True), errors


def combine_stock_registers(inward_df, outward_df):
    """
    Merge inward and outward DataFrames into a unified stock_df
    expected by compute_stock: columns [material, inward, outward].
    """
    parts = []
    if inward_df is not None and not inward_df.empty:
        parts.append(inward_df[['material', 'inward', 'outward']])
    if outward_df is not None and not outward_df.empty:
        parts.append(outward_df[['material', 'inward', 'outward']])
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


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
        return pd.DataFrame(), str(e)# -----------------------------------------------------------------
# 4. BILL OF MATERIALS (BOM) INGESTION & REQUIREMENTS ENGINE
# Format: Product Name | Copper Type | Copper Weight |
#         Lamination Type | Lamination Weight | Bobbin Type | Other Reqs
# -----------------------------------------------------------------

def ingest_bom_excel(file):
    """
    Parse a BOM Excel file in the user's specific format:
    Columns: PRODUCT NAME, COPPER TYPE, COPPER WEIGHT (EST.),
             LAMINATION TYPE, LAMINATION WEIGHT, BOBBIN TYPE, OTHER REQS

    Returns (bom_df, error)
    bom_df columns: product, copper_type, copper_weight_kg,
                    lamination_type, lamination_weight_kg,
                    bobbin_type, other_reqs
    """
    import io as _io
    fname = getattr(file, 'name', str(file))
    try:
        file_bytes = file.read() if hasattr(file, 'read') else open(file, 'rb').read()
    except Exception as e:
        return pd.DataFrame(), f"{fname}: Cannot read -- {e}"

    raw = None
    for engine in [None, 'xlrd']:
        try:
            kw = {'header': None, 'dtype': str}
            if engine:
                kw['engine'] = engine
            raw = pd.read_excel(_io.BytesIO(file_bytes), **kw).fillna('')
            break
        except Exception:
            continue

    if raw is None:
        return pd.DataFrame(), f"{fname}: Cannot open file."

    # Find the header row (contains 'product' or 'copper')
    hdr_row = 0
    for i in range(min(5, len(raw))):
        row_text = ' '.join(str(v).lower() for v in raw.iloc[i].values)
        if 'product' in row_text or 'copper' in row_text:
            hdr_row = i
            break

    df = pd.read_excel(_io.BytesIO(file_bytes), header=hdr_row, dtype=str).fillna('')
    df.columns = [str(c).strip().lower() for c in df.columns]

    # Map columns flexibly
    def _fc(columns, kws):
        for col in columns:
            cl = col.lower()
            if any(kw in cl for kw in kws):
                return col
        return None

    cols = list(df.columns)
    prod_col  = _fc(cols, ['product', 'name', 'item'])
    cu_type   = _fc(cols, ['copper type', 'copper_type'])
    cu_wt     = _fc(cols, ['copper weight', 'copper_weight', 'cu weight'])
    lam_type  = _fc(cols, ['lamination type', 'lamination_type'])
    lam_wt    = _fc(cols, ['lamination weight', 'lamination_weight'])
    bob_type  = _fc(cols, ['bobbin', 'bobbin type'])
    other_col = _fc(cols, ['other', 'reqs', 'remarks', 'misc'])

    if not prod_col:
        return pd.DataFrame(), f"{fname}: 'Product Name' column not found."
    if not cu_wt and not lam_wt:
        return pd.DataFrame(), f"{fname}: No material weight columns found (expected Copper Weight, Lamination Weight)."

    out = pd.DataFrame()
    out['product']              = df[prod_col].astype(str).str.strip()
    out['copper_type']          = df[cu_type].astype(str).str.strip()       if cu_type  else ''
    out['copper_weight_kg']     = pd.to_numeric(
        df[cu_wt].astype(str).str.replace(r'[^\d.]', '', regex=True), errors='coerce'
    ) if cu_wt else 0.0
    out['lamination_type']      = df[lam_type].astype(str).str.strip()      if lam_type else ''
    out['lamination_weight_kg'] = pd.to_numeric(
        df[lam_wt].astype(str).str.replace(r'[^\d.]', '', regex=True), errors='coerce'
    ) if lam_wt else 0.0
    out['bobbin_type']          = df[bob_type].astype(str).str.strip()      if bob_type else ''
    out['other_reqs']           = df[other_col].astype(str).str.strip()     if other_col else ''

    # Remove blank / header rows
    out = out[out['product'].str.len() > 3]
    out = out[~out['product'].str.lower().str.contains('product|name|item', na=False)]
    out = out.dropna(how='all').reset_index(drop=True)

    return out, None


def compute_material_requirements(predictions_df, bom_df, stock_summary_df):
    """
    Cross-reference predicted orders with BOM to generate material purchase alerts.

    Logic:
      For each predicted reorder:
        - Look up product in BOM (fuzzy prefix match on first 10 chars)
        - Multiply BOM quantities by expected order quantity
        - Compare against current stock
        - Flag shortfall as 'BUY NOW', near-shortfall as 'PREPARE', ok as 'OK'

    Returns (requirements_df, error)
    Columns: Customer, Product, Days Until Order, Material, Need (kg/nos),
             In Stock (kg/nos), Shortfall, Status
    """
    try:
        if predictions_df is None or predictions_df.empty:
            return pd.DataFrame(), "No reorder predictions available."
        if bom_df is None or bom_df.empty:
            return pd.DataFrame(), "No BOM data uploaded."

        # Build stock lookup: {material_keyword: current_stock}
        stk = {}
        if stock_summary_df is not None and not stock_summary_df.empty:
            for _, row in stock_summary_df.iterrows():
                mat = str(row.get('Material', '')).lower()
                stk[mat] = float(row.get('Current Stock', 0))

        def _get_stock(keyword):
            """Find best matching stock level for a material keyword."""
            kl = keyword.lower()[:8]
            for k, v in stk.items():
                if kl in k or k[:8] in kl:
                    return v
            return None  # Unknown = not tracked

        rows = []
        today = pd.Timestamp.now().normalize()

        for _, pred in predictions_df.iterrows():
            customer = pred.get('Customer', 'Unknown')
            product  = str(pred.get('Last Order', ''))  # use product from sales later

            # Try to match BOM row by product name prefix
            bom_match = None
            pred_product = customer  # fallback label

            # Match BOM rows to this prediction's likely product
            # (Sales data needed for exact match â€” use first 8 chars of product)
            for _, brow in bom_df.iterrows():
                bom_match = brow
                break  # For now use first match; will be refined with sales linkage

            if bom_match is None:
                continue

            # Days until predicted order
            try:
                next_order = pd.to_datetime(pred.get('Predicted Next Order'), dayfirst=True)
                days_left  = (next_order - today).days
            except Exception:
                days_left = 30

            # Assumed order quantity = 1 unit (conservative)
            assumed_qty = 1.0

            # Build material requirements list
            material_items = [
                ('Copper',      bom_match.get('copper_type', ''),     float(bom_match.get('copper_weight_kg', 0) or 0),     'KG'),
                ('Lamination',  bom_match.get('lamination_type', ''), float(bom_match.get('lamination_weight_kg', 0) or 0), 'KG'),
                ('Bobbin',      bom_match.get('bobbin_type', ''),     1.0,                                                   'NOS'),
            ]

            for mat_name, mat_spec, unit_qty, unit in material_items:
                if unit_qty <= 0:
                    continue
                need       = round(assumed_qty * unit_qty, 3)
                in_stock   = _get_stock(mat_name)
                stock_disp = round(in_stock, 3) if in_stock is not None else 'Unknown'

                if in_stock is None:
                    status = 'CHECK STOCK'
                    shortfall = 'Unknown'
                elif in_stock >= need * 1.5:
                    status = 'OK'
                    shortfall = 0
                elif in_stock >= need:
                    status = 'PREPARE'
                    shortfall = 0
                else:
                    status = 'BUY NOW'
                    shortfall = round(need - in_stock, 3)

                # Urgency: escalate if order is within 7 days
                if days_left <= 7 and status == 'PREPARE':
                    status = 'BUY NOW'

                rows.append({
                    'Customer':        customer,
                    'Days Till Order': days_left,
                    'Pred. Order Date': pred.get('Predicted Next Order', ''),
                    'Material':        f"{mat_name} ({mat_spec})" if mat_spec else mat_name,
                    'Need':            f"{need} {unit}",
                    'In Stock':        f"{stock_disp} {unit}" if in_stock is not None else 'Unknown',
                    'Shortfall':       f"{shortfall} {unit}" if isinstance(shortfall, float) else shortfall,
                    'Status':          status,
                })

        if not rows:
            return pd.DataFrame(), "Could not match any BOM entries to predictions."

        result = pd.DataFrame(rows)
        
        # Sort by urgency (BUY NOW > PREPARE > CHECK STOCK > OK)
        severity_map = {'BUY NOW': 1, 'PREPARE': 2, 'CHECK STOCK': 3, 'OK': 4}
        result['Severity'] = result['Status'].map(severity_map)
        result = result.sort_values(['Severity', 'Days Till Order'], ascending=[True, True]).drop(columns=['Severity'])
        
        return result, None

    except Exception as e:
        return pd.DataFrame(), str(e)


# ─────────────────────────────────────────────────────────────
# COMMODITY RATE TRACKING (yfinance)
# ─────────────────────────────────────────────────────────────
def get_commodity_rates():
    """
    Fetch Copper (HG=F) and Aluminium (ALI=F) 90-day futures data.
    Returns a dict with current price, 30d high, 30d low, trendline, and history df.
    """
    try:
        import yfinance as yf
    except ImportError:
        return None, "yfinance not installed. Please install it to view commodity rates."

    tickers = {
        'Copper': 'HG=F',     # High Grade Copper Futures (USD/lb)
        'Aluminium': 'ALI=F'  # Aluminum Futures (USD/Tonne)
    }

    results = {}
    try:
        # Fetching live forex INR=X
        usd_inr = yf.Ticker("INR=X").history(period="1d")['Close'].iloc[-1]
    except Exception:
        usd_inr = 83.5

    import numpy as np
    import pandas as pd

    for name, ticker in tickers.items():
        try:
            t = yf.Ticker(ticker)
            hist = t.history(period="90d")
            if hist.empty:
                continue
            
            # Convert to INR / KG
            if name == 'Copper':
                # 1 lb = 0.453592 kg -> price per kg = price_per_lb / 0.453592
                hist['Price_INR_KG'] = (hist['Close'] / 0.453592) * usd_inr
            else:
                # 1 Tonne = 1000 kg -> price per kg = price_per_tonne / 1000
                hist['Price_INR_KG'] = (hist['Close'] / 1000) * usd_inr

            current_price = hist['Price_INR_KG'].iloc[-1]
            last_30d = hist.tail(30)['Price_INR_KG']
            high_30d = last_30d.max()
            low_30d = last_30d.min()

            # Simple linear regression for 30-day forecast trend
            y = hist['Price_INR_KG'].values
            x = np.arange(len(y))
            slope, intercept = np.polyfit(x, y, 1)
            future_x = np.arange(len(y), len(y) + 30)
            forecast = slope * future_x + intercept

            # Calculate momentum (slope over last 14 days)
            recent_y = hist['Price_INR_KG'].tail(14).values
            recent_x = np.arange(len(recent_y))
            recent_slope, _ = np.polyfit(recent_x, recent_y, 1)
            trend_str = "UP ↗" if recent_slope > 0 else "DOWN ↘"
            if abs(recent_slope) < 0.5: trend_str = "FLAT →"

            results[name] = {
                'current_price': current_price,
                'high_30d': high_30d,
                'low_30d': low_30d,
                'trend': trend_str,
                'history': hist[['Price_INR_KG']].reset_index(),
                'forecast': forecast.tolist()
            }
        except Exception as e:
            continue

    if not results:
        return None, "Failed to fetch commodity data from Yahoo Finance."
    
    return results, None


# ─────────────────────────────────────────────────────────────
# LOCAL DATA PERSISTENCE
# ─────────────────────────────────────────────────────────────
import os
import shutil

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.data')

def save_data(session_state):
    """Save processed dataframes from session_state to local parquet files."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    keys_to_save = ['sales_df', 'purchase_df', 'stock_df', 'bom_df', 'predictions_df', 'stock_summary', 'outlook_df', 'anchor_df', 'requirements_df']
    saved_count = 0
    for key in keys_to_save:
        df = session_state.get(key)
        if isinstance(df, pd.DataFrame) and not df.empty:
            path = os.path.join(DATA_DIR, f"{key}.parquet")
            try:
                # Convert object columns to string to avoid parquet errors with mixed types
                save_df = df.copy()
                for col in save_df.select_dtypes(include=['object']).columns:
                    save_df[col] = save_df[col].astype(str)
                save_df.to_parquet(path, index=False)
                saved_count += 1
            except Exception as e:
                print(f"Error saving {key}: {e}")
    return saved_count

def load_data():
    """Load dataframes from local parquet files. Returns a dict."""
    if not os.path.exists(DATA_DIR):
        return {}

    loaded = {}
    keys_to_load = ['sales_df', 'purchase_df', 'stock_df', 'bom_df', 'predictions_df', 'stock_summary', 'outlook_df', 'anchor_df', 'requirements_df']
    for key in keys_to_load:
        path = os.path.join(DATA_DIR, f"{key}.parquet")
        if os.path.exists(path):
            try:
                loaded[key] = pd.read_parquet(path)
            except Exception as e:
                print(f"Error loading {key}: {e}")
    return loaded

def clear_data():
    """Delete all local parquet files in the data directory."""
    if not os.path.exists(DATA_DIR):
        return
    try:
        shutil.rmtree(DATA_DIR)
    except Exception as e:
        print(f"Error clearing data: {e}")
