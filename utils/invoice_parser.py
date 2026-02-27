"""
GST Invoice PDF Parser
Parses individual Tally-generated GST Invoice PDFs into structured rows.
Each PDF = one invoice. Returns a combined Sales Register DataFrame.
"""

import io
import re
import pandas as pd

try:
    import pdfplumber
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False


def _extract_pdf_tables_and_text(file_bytes):
    """Extract all tables and full text from a PDF page."""
    tables, text_lines = [], []
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                tbls = page.extract_tables()
                if tbls:
                    tables.extend(tbls)
                t = page.extract_text()
                if t:
                    text_lines.extend(t.splitlines())
    except Exception:
        pass
    return tables, text_lines


def _clean_cell(val):
    """Normalize a table cell value."""
    if val is None:
        return ''
    return str(val).strip().replace('\n', ' ')


def _find_in_text(lines, *labels):
    """Search text lines for a value that appears after a given label."""
    text = '\n'.join(lines)
    for label in labels:
        # Try pattern: label followed by colon and value on same or next line
        pattern = re.compile(re.escape(label) + r'\s*[:\-]?\s*(.+)', re.IGNORECASE)
        m = pattern.search(text)
        if m:
            val = m.group(1).strip().split('\n')[0].strip()
            if val:
                return val
    return None


def parse_gst_invoice_pdf(file_bytes):
    """
    Parse a single Tally GST Invoice PDF.
    Returns (list_of_row_dicts, error_string).
    Each row has: date, invoice_no, customer, product, quantity, unit, amount.
    """
    if not PDF_AVAILABLE:
        return [], "pdfplumber not installed."

    try:
        tables, text_lines = _extract_pdf_tables_and_text(file_bytes)
        text = '\n'.join(text_lines)

        if not text.strip():
            return [], "Scanned PDF - no text extractable. Use text-based Tally export."

        # ── Extract invoice-level fields ────────────────────────────
        # Strategy: first try raw text regex (most reliable for merged cells)
        # then fall back to label search

        # Date — look for date patterns in raw text
        date_val = None
        # Priority 1: dd-Mon-yy or dd-Mon-yyyy pattern (e.g. 8-Jan-26)
        m = re.search(r'\b(\d{1,2}[-](?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[\-\/]\d{2,4})\b', text, re.IGNORECASE)
        if m:
            date_val = m.group(1)
        else:
            # dd/mm/yyyy or dd-mm-yyyy
            m = re.search(r'\b(\d{1,2}[/\-]\d{1,2}[/\-]\d{4})\b', text)
            date_val = m.group(1) if m else None

        # Invoice number — look for Tally pattern like AE/25-26/1316 or INV/001
        invoice_no = 'UNKNOWN'
        # Try 3-part pattern first: AE/25-26/1316
        m = re.search(r'\b([A-Z]{1,6}[/\-]\d{2,4}[/\-]\d{2,4}[/\-]\d{1,6})\b', text)
        if not m:
            # 2-part: AE/1316 or INV/001
            m = re.search(r'\b([A-Z]{1,6}[/\-]\d{1,6})\b', text)
        if m:
            invoice_no = m.group(1)


        # Customer name — "Buyer (Bill to)" section
        customer = None
        # Look for text after "Buyer (Bill to)" or "Bill To"
        for label in ['Buyer (Bill to)', 'Bill to', 'Buyer', 'Customer', 'Party Name']:
            m = re.search(re.escape(label) + r'[)]*\s*\n+\s*([A-Z][A-Z0-9\s\.\&\,\-]+)', text, re.IGNORECASE)
            if m:
                candidate = m.group(1).strip().split('\n')[0].strip()
                # Must be reasonably long and look like a company name
                if len(candidate) > 5 and not any(kw in candidate.lower() for kw in ['certif', 'iso', 'state', 'gstin', 'contact']):
                    customer = candidate
                    break

        # Fallback: find ALL-CAPS lines that look like company names after skipping the seller header
        if not customer:
            seller_done = False
            for line in text_lines:
                line_clean = line.strip()
                if not seller_done:
                    if 'buyer' in line_clean.lower() or 'bill to' in line_clean.lower():
                        seller_done = True
                    continue
                if (re.match(r'^[A-Z][A-Z\s\.\&\,\-]{5,}$', line_clean) and
                        not any(kw in line_clean.lower() for kw in ['gstin', 'state', 'contact', 'sr.', 'no.'])):
                    customer = line_clean
                    break

        # ── Extract product rows from the items table ───────────────
        # Find the items table — it has columns: Sl No, Description, HSN, Qty, Rate, per, Amount
        items_rows = []

        for table in tables:
            if not table or len(table) < 2:
                continue
            # Clean table
            clean_table = [[_clean_cell(c) for c in row] for row in table]
            clean_table = [r for r in clean_table if any(c for c in r)]

            # Find header row with "description" and "quantity"/"qty"
            hdr_idx = None
            for i, row in enumerate(clean_table):
                row_str = ' '.join(row).lower()
                if ('description' in row_str or 'goods' in row_str) and \
                   ('qty' in row_str or 'quantity' in row_str or 'amount' in row_str):
                    hdr_idx = i
                    break

            if hdr_idx is None:
                continue

            header = clean_table[hdr_idx]

            # Map header positions
            col_idx = {}
            for ci, h in enumerate(header):
                hl = h.lower()
                if 'description' in hl or 'goods' in hl:
                    col_idx['product'] = ci
                elif 'qty' in hl or 'quantity' in hl:
                    col_idx['quantity'] = ci
                elif 'per' == hl.strip():
                    col_idx['unit'] = ci
                elif 'amount' in hl:
                    col_idx['amount'] = ci
                elif 'rate' in hl:
                    col_idx['rate'] = ci
                elif 'hsn' in hl or 'sac' in hl:
                    col_idx['hsn'] = ci

            if 'product' not in col_idx:
                continue

            # Parse data rows
            for row in clean_table[hdr_idx + 1:]:
                product = row[col_idx['product']] if col_idx.get('product') is not None and col_idx['product'] < len(row) else ''
                # Skip totals, SGST, CGST rows
                if not product or any(kw in product.lower() for kw in
                                       ['total', 'sgst', 'cgst', 'igst', 'tax', 'amount in words', 'grand']):
                    continue
                # Skip rows where product looks like a sub-note (italic spec line)
                if product.startswith('SEC-') or product.startswith('sec-'):
                    continue

                qty_raw = row[col_idx['quantity']] if col_idx.get('quantity') is not None and col_idx['quantity'] < len(row) else ''
                qty_str = re.sub(r'[^\d.]', '', qty_raw)  # strip "NOS." etc.
                qty = float(qty_str) if qty_str else None

                unit_raw = ''
                if col_idx.get('unit') is not None and col_idx['unit'] < len(row):
                    unit_raw = row[col_idx['unit']]

                # Fallback unit from qty string (e.g. "30 NOS.")
                if not unit_raw:
                    m = re.search(r'[A-Za-z]+', qty_raw)
                    unit_raw = m.group(0) if m else ''

                amt_raw = row[col_idx['amount']] if col_idx.get('amount') is not None and col_idx['amount'] < len(row) else ''
                amt_str = re.sub(r'[^\d.]', '', amt_raw)
                amount = float(amt_str) if amt_str else None

                if product and (qty or amount):
                    items_rows.append({
                        'date': date_val,
                        'invoice_no': invoice_no,
                        'customer': customer or 'UNKNOWN',
                        'product': product,
                        'quantity': qty,
                        'unit': unit_raw,
                        'amount': amount,
                    })

        if not items_rows:
            return [], f"Could not extract product rows. Date={date_val}, Customer={customer}, Tables found={len(tables)}"

        return items_rows, None

    except Exception as e:
        return [], f"Invoice parse error: {str(e)}"


def ingest_multiple_invoices(file_list):
    """
    Parse a list of (filename, bytes) tuples as individual GST invoices.
    Returns (combined_df, list_of_errors).
    combined_df columns: date, invoice_no, customer, product, quantity, unit, amount
    """
    all_rows = []
    errors = []

    for fname, fbytes in file_list:
        rows, err = parse_gst_invoice_pdf(fbytes)
        if err:
            errors.append(f"{fname}: {err}")
        else:
            all_rows.extend(rows)

    if not all_rows:
        return pd.DataFrame(), errors

    df = pd.DataFrame(all_rows)
    df['date'] = pd.to_datetime(df['date'], errors='coerce', dayfirst=True)
    df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')
    df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
    df = df.dropna(subset=['customer']).reset_index(drop=True)
    return df, errors
