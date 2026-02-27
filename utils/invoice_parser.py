"""
GST Invoice PDF Parser — Full Text Scan Approach
Reads all raw text from each Tally GST Invoice PDF page by page,
then extracts fields by scanning patterns across the whole text.
"""

import io
import re
import pandas as pd

try:
    import pdfplumber
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def _read_pdf_text(file_bytes):
    """Extract all text from a PDF. Returns (full_text, lines_list, error)."""
    if not PDF_AVAILABLE:
        return None, [], "pdfplumber not installed."
    try:
        lines_all = []
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            if not pdf.pages:
                return None, [], "Empty PDF."
            for page in pdf.pages:
                t = page.extract_text(x_tolerance=3, y_tolerance=3)
                if t:
                    lines_all.extend(t.splitlines())
        if not lines_all:
            return None, [], "Scanned PDF – no extractable text. Use text-based Tally export."
        full_text = '\n'.join(lines_all)
        return full_text, lines_all, None
    except Exception as e:
        return None, [], f"PDF read error: {str(e)}"


def _strip(s):
    return str(s).strip() if s else ''


# ─────────────────────────────────────────────────────────────
# EXTRACTION FUNCTIONS  (scan the full text)
# ─────────────────────────────────────────────────────────────

def _extract_date(full_text):
    """Find invoice date — try dd-Mon-yy first, then dd/mm/yyyy."""
    # dd-Mon-yy or dd-Mon-yyyy  (e.g. 8-Jan-26, 15-Mar-2025)
    m = re.search(
        r'\b(\d{1,2}[-](?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[-]\d{2,4})\b',
        full_text, re.IGNORECASE
    )
    if m:
        return m.group(1)
    # dd/mm/yyyy
    m = re.search(r'\b(\d{1,2}[/]\d{1,2}[/]\d{4})\b', full_text)
    if m:
        return m.group(1)
    return None


def _extract_invoice_no(full_text):
    """Find Tally invoice number like AE/25-26/1316."""
    # 3-segment: PREFIX/YYYY-YY/NUM
    m = re.search(r'\b([A-Z]{1,6}[/\-]\d{2,4}[/\-]\d{2,4}[/\-]\d{1,6})\b', full_text)
    if m:
        return m.group(1)
    # 2-segment: PREFIX/NUM
    m = re.search(r'\b([A-Z]{1,6}[/\-]\d{1,6})\b', full_text)
    if m:
        return m.group(1)
    return 'UNKNOWN'


def _extract_customer(lines):
    """
    Scan lines top-to-bottom.
    Find 'Buyer' / 'Bill to' label, then return the NEXT non-empty,
    non-metadata line as the customer name.
    """
    SKIP_KEYWORDS = [
        'gstin', 'state', 'contact', 'sr.', 'no.', 'code', 'e-mail',
        'email', 'phone', 'mob', 'dispatched', 'destination',
        'delivery', 'term', 'reference', 'udyam', 'iso', 'certified',
        'invoice', 'dated', 'voucher', 'bill no', 'mode', 'payment',
        'buyer', 'bill to', 'ship to',
    ]

    buyer_section = False
    for i, line in enumerate(lines):
        l = line.strip()
        ll = l.lower()

        if re.search(r'buyer\s*[\(\[]?\s*bill\s+to', ll) or ll.startswith('buyer') or 'bill to' in ll:
            buyer_section = True
            continue

        if buyer_section:
            if not l:
                continue
            if any(kw in ll for kw in SKIP_KEYWORDS):
                continue
            if len(l) < 5:
                continue
            # Strip trailing date patterns like "8-Jan-26" or dates that got merged on same line
            l = re.sub(r'\s+\d{1,2}[-/]\w{3,9}[-/]\d{2,4}\s*$', '', l).strip()
            l = re.sub(r'\s+\d{1,2}[/-]\d{1,2}[/-]\d{4}\s*$', '', l).strip()
            if l and len(l) > 5:
                return l


    # Fallback: look for lines with company indicators
    for line in lines:
        l = line.strip()
        if (re.search(r'\b(PVT|LTD|LLC|INC|CORP|CO\.|PRIVATE|LIMITED|INDUSTRIES|ENTERPRISES|TRADING|SOLUTIONS|SYSTEMS|PERIPHERALS)\b', l, re.IGNORECASE)
                and len(l) > 8):
            return l
    return None


def _extract_items_from_text(lines):
    """
    Scan all lines looking for the items table header
    ('Description of Goods', 'Sl', 'HSN', 'Qty', 'Amount' etc.)
    then parse subsequent lines as product rows.
    """
    ITEMS_HEADER_KEYWORDS = ['description', 'goods', 'hsn', 'sac', 'qty', 'quantity', 'amount', 'rate']
    SKIP_PRODUCT_KEYWORDS = ['sgst', 'cgst', 'igst', 'total', 'tax', 'cess', 'amount in words',
                              'declaration', 'bank', 'authoris', 'subject', 'computer', 'e. & o.e']

    # Find items header line
    header_idx = None
    for i, line in enumerate(lines):
        ll = line.lower()
        hits = sum(1 for kw in ITEMS_HEADER_KEYWORDS if kw in ll)
        if hits >= 3:  # Must match at least 3 column keywords
            header_idx = i
            break

    if header_idx is None:
        return []

    items = []
    # Scan lines after the header
    i = header_idx + 1
    while i < len(lines):
        line = lines[i].strip()
        i += 1

        if not line:
            continue

        ll = line.lower()

        # Stop if we hit the totals/footer section
        if any(kw in ll for kw in SKIP_PRODUCT_KEYWORDS):
            continue

        # A product line starts with a serial number or has a numeric amount at the end
        # Pattern: optional-sl-no  product-description  optional-HSN  qty  rate  amount
        # Try to detect lines that have a monetary amount at the end: digits with comma, e.g. 9,300.00
        has_amount = bool(re.search(r'\d{1,3}(?:,\d{3})*\.\d{2}\s*$', line))
        starts_with_num = bool(re.match(r'^\d+\s+\S', line))

        if has_amount or starts_with_num:
            # Extract numbers from line
            numbers = re.findall(r'[\d,]+\.?\d*', line)
            clean_nums = []
            for n in numbers:
                try:
                    clean_nums.append(float(n.replace(',', '')))
                except ValueError:
                    pass

            # Remove serial number if line starts with it
            desc_line = re.sub(r'^\d+\s+', '', line)

            # Extract description (text before HSN and numbers)
            # Description is the longest text segment at the start
            desc_match = re.match(r'^([A-Za-z0-9\s\-\/\.\(\)]+?)(?:\s+\d{4,}\s|\s+\d+\s+(?:NOS|PCS|KG|MTR|SET|BOX|LTR|MTS|UNIT))', desc_line, re.IGNORECASE)
            if desc_match:
                description = desc_match.group(1).strip()
            else:
                # Take all text before the first 4+ digit number (likely HSN code)
                description = re.sub(r'\s+\d{4,}.*$', '', desc_line).strip()

            if not description or len(description) < 3:
                continue
            if any(kw in description.lower() for kw in SKIP_PRODUCT_KEYWORDS):
                continue

            # Quantity: look for "N NOS." or "N PCS" pattern
            qty = None
            qty_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:NOS|PCS|KG|MTR|SET|BOX|LTR|MTS|UNIT|Nos|Pcs)\.?', line, re.IGNORECASE)
            if qty_match:
                try:
                    qty = float(qty_match.group(1))
                except ValueError:
                    pass
            elif clean_nums:
                # Use the number that's likely qty (usually smallest, non-amount)
                for n in clean_nums:
                    if n < 10000 and n == int(n):
                        qty = n
                        break

            # Unit
            unit_match = re.search(r'\b(NOS|PCS|KG|MTR|SET|BOX|LTR|MTS|UNIT)\b\.?', line, re.IGNORECASE)
            unit = unit_match.group(1).upper() if unit_match else ''

            # Amount: last number in line is typically the amount
            amount = clean_nums[-1] if clean_nums else None

            if description and (qty is not None or amount is not None):
                items.append({
                    'description': description,
                    'quantity': qty,
                    'unit': unit,
                    'amount': amount,
                })

        # Check if next line is a continuation of description (no numbers = sub-description or spec)
        # Skip it silently — the description is already captured

    return items


# ─────────────────────────────────────────────────────────────
# MAIN PARSER
# ─────────────────────────────────────────────────────────────

def parse_gst_invoice_pdf(file_bytes):
    """
    Parse a single Tally GST Invoice PDF by full-text scan.
    Returns (list_of_row_dicts, error_string).
    Each row: date, invoice_no, customer, product, quantity, unit, amount
    """
    full_text, lines, err = _read_pdf_text(file_bytes)
    if err:
        return [], err

    date_val   = _extract_date(full_text)
    invoice_no = _extract_invoice_no(full_text)
    customer   = _extract_customer(lines)
    items      = _extract_items_from_text(lines)

    if not items:
        return [], (
            f"Could not extract product rows. "
            f"Date={date_val}, Customer={customer}, "
            f"Lines scanned={len(lines)}"
        )

    rows = []
    for item in items:
        rows.append({
            'date':       date_val,
            'invoice_no': invoice_no,
            'customer':   customer or 'UNKNOWN',
            'product':    item['description'],
            'quantity':   item['quantity'],
            'unit':       item['unit'],
            'amount':     item['amount'],
        })

    return rows, None


# ─────────────────────────────────────────────────────────────
# MULTI-INVOICE INGESTION
# ─────────────────────────────────────────────────────────────

def ingest_multiple_invoices(file_list):
    """
    Parse a list of (filename, bytes) tuples as individual GST invoices.
    Returns (combined_df, list_of_errors).
    """
    all_rows, errors = [], []

    for fname, fbytes in file_list:
        rows, err = parse_gst_invoice_pdf(fbytes)
        if err:
            errors.append(f"{fname}: {err}")
        else:
            all_rows.extend(rows)

    if not all_rows:
        return pd.DataFrame(), errors

    df = pd.DataFrame(all_rows)
    df['date']     = pd.to_datetime(df['date'], errors='coerce', dayfirst=True)
    df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')
    df['amount']   = pd.to_numeric(df['amount'],   errors='coerce')
    df = df.dropna(subset=['customer']).reset_index(drop=True)
    return df, errors
