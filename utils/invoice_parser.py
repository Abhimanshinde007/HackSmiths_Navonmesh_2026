"""
Gemini AI-Powered GST Invoice Parser
Uses Google Gemini Flash to read raw PDF text and extract structured invoice data.
Falls back to regex parser if no API key is available.
"""

import io
import re
import json
import pandas as pd

try:
    import pdfplumber
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


# ─────────────────────────────────────────────────────────────
# PDF TEXT EXTRACTION (shared)
# ─────────────────────────────────────────────────────────────

def _read_pdf_text(file_bytes):
    """Extract all raw text from a PDF. Returns (text, error)."""
    if not PDF_AVAILABLE:
        return None, "pdfplumber not installed."
    try:
        lines = []
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            if not pdf.pages:
                return None, "Empty PDF."
            for page in pdf.pages:
                t = page.extract_text(x_tolerance=3, y_tolerance=3)
                if t:
                    lines.extend(t.splitlines())
        if not lines:
            return None, "Scanned PDF — no extractable text. Use text-based Tally export."
        return '\n'.join(lines), None
    except Exception as e:
        return None, f"PDF read error: {str(e)}"


# ─────────────────────────────────────────────────────────────
# GEMINI AI EXTRACTION
# ─────────────────────────────────────────────────────────────

GEMINI_PROMPT = """You are a data extraction assistant for Indian GST invoices.

Given the raw text extracted from a Tally-generated GST Invoice PDF, extract the following fields and return ONLY valid JSON with no explanation or markdown.

Required JSON structure:
{
  "date": "DD-Mon-YY or DD/MM/YYYY string, or null",
  "invoice_no": "invoice number string like AE/25-26/1316 or 322",
  "customer": "exact buyer company name (from Buyer / Bill to section only — NOT the seller, NOT vehicle number, NOT address)",
  "products": [
    {
      "description": "product/item description",
      "quantity": numeric_value_or_null,
      "unit": "NOS/PCS/KG/etc or empty string",
      "amount": numeric_invoice_value_or_null
    }
  ]
}

Rules:
- customer must be the BUYER company name only (the party who is purchasing)
- Ignore SGST, CGST, IGST, Total rows in products
- Ignore sub-descriptions, material notes, italic lines below the main product line
- quantity should be the physical unit count (like 30, 1000), not rupee amounts
- amount should be the line item amount in rupees (like 9300.00, 12500.00)
- If a field is not found, use null
- Return only the JSON object, nothing else

Invoice text:
"""


def _parse_with_gemini(text, api_key):
    """Send invoice text to Gemini Flash and return structured dict."""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(
            GEMINI_PROMPT + text,
            generation_config=genai.types.GenerationConfig(
                temperature=0,
                max_output_tokens=1024,
            )
        )
        raw = response.text.strip()
        # Strip markdown code fences if present
        raw = re.sub(r'^```(?:json)?\s*', '', raw, flags=re.MULTILINE)
        raw = re.sub(r'\s*```$', '', raw, flags=re.MULTILINE)
        return json.loads(raw), None
    except json.JSONDecodeError as e:
        return None, f"Gemini returned non-JSON: {str(e)}"
    except Exception as e:
        return None, f"Gemini API error: {str(e)}"


# ─────────────────────────────────────────────────────────────
# MAIN INVOICE PARSER
# ─────────────────────────────────────────────────────────────

def parse_gst_invoice_pdf(file_bytes, api_key=None):
    """
    Parse a single Tally GST invoice PDF.
    Uses Gemini AI if api_key is provided, otherwise uses regex fallback.
    Returns (list_of_row_dicts, error_string).
    """
    text, err = _read_pdf_text(file_bytes)
    if err:
        return [], err
    if not text or len(text.strip()) < 50:
        return [], "Scanned PDF — no extractable text."

    # ── Gemini path ────────────────────────────────────────────
    if api_key and GEMINI_AVAILABLE:
        data, err = _parse_with_gemini(text, api_key)
        if err:
            return [], f"AI parse failed: {err}"

        customer = data.get('customer') or 'UNKNOWN'
        date_val = data.get('date')
        invoice_no = data.get('invoice_no') or 'UNKNOWN'
        products = data.get('products') or []

        rows = []
        for p in products:
            desc = str(p.get('description') or '').strip()
            if not desc or len(desc) < 3:
                continue
            rows.append({
                'date':       date_val,
                'invoice_no': invoice_no,
                'customer':   customer,
                'product':    desc,
                'quantity':   p.get('quantity'),
                'unit':       p.get('unit') or '',
                'amount':     p.get('amount'),
            })

        if not rows:
            return [], f"Gemini extracted 0 products. Customer={customer}, Invoice={invoice_no}"
        return rows, None

    # ── Regex fallback (no API key) ────────────────────────────
    return _regex_parse(text)


def _regex_parse(text):
    """Simple regex fallback when no Gemini key is set."""
    lines = text.splitlines()

    # Date
    m = re.search(r'\b(\d{1,2}[-](?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[-]\d{2,4})\b', text, re.IGNORECASE)
    date_val = m.group(1) if m else None
    if not date_val:
        m = re.search(r'\b(\d{1,2}[/]\d{1,2}[/]\d{4})\b', text)
        date_val = m.group(1) if m else None

    # Invoice no
    m = re.search(r'\b([A-Z]{1,6}[/\-]\d{2,4}[/\-]\d{2,4}[/\-]\d{1,6})\b', text)
    invoice_no = m.group(1) if m else 'UNKNOWN'

    # Customer — look for company indicators in Buyer section
    customer = None
    buyer_section = False
    SKIP = ['gstin', 'state', 'contact', 'udyam', 'iso', 'certified', 'invoice',
            'dated', 'voucher', 'mode', 'payment', 'buyer', 'bill to', 'dispatch',
            'destination', 'vehicle', 'lading', 'delivery', 'term', 'place', 'supply', 'road']
    for line in lines:
        l, ll = line.strip(), line.strip().lower()
        if re.search(r'buyer\s*[\(\[]?\s*bill\s+to', ll) or ll.startswith('buyer') or 'bill to' in ll:
            buyer_section = True
            continue
        if buyer_section:
            if not l or len(l) < 5: continue
            if any(kw in ll for kw in SKIP): continue
            if re.match(r'^[A-Z]{2}\d{2}[A-Z]{2}\d{4}$', l): continue  # skip vehicle numbers
            l = re.sub(r'\s+\d{1,2}[-/]\w{3,9}[-/]\d{2,4}\s*$', '', l).strip()
            if l and len(l) > 5:
                customer = l
                break

    if not customer:
        for line in lines:
            if re.search(r'\b(PVT|LTD|PRIVATE|LIMITED|INDUSTRIES|ENTERPRISES|ENGINEERING)\b', line, re.IGNORECASE) and len(line.strip()) > 8:
                customer = line.strip()
                break

    # Items — look for lines with amount at end
    HEADER_KW = ['description', 'goods', 'hsn', 'sac', 'qty', 'quantity', 'amount', 'rate']
    SKIP_KW = ['sgst', 'cgst', 'igst', 'output', 'total', 'tax', 'amount in words',
               'chargeable', 'declaration', 'bank', 'authoris', 'e. & o.e', 'taxable value',
               'central tax', 'state tax', 'material', 'sec-']

    hdr_idx = None
    for i, line in enumerate(lines):
        hits = sum(1 for kw in HEADER_KW if kw in line.lower())
        if hits >= 3:
            hdr_idx = i
            break

    items = []
    if hdr_idx is not None:
        for line in lines[hdr_idx + 1:]:
            l = line.strip()
            if not l: continue
            ll = l.lower()
            if any(kw in ll for kw in SKIP_KW): continue
            has_amount = bool(re.search(r'\d{1,3}(?:,\d{3})*\.\d{2}\s*$', l))
            if not has_amount: continue
            # Skip obvious non-product lines (only numbers/tax rows)
            if re.match(r'^[\d\s%,\.]+$', l): continue

            nums = [float(n.replace(',', '')) for n in re.findall(r'[\d,]+\.\d{2}', l) if n.replace(',','').replace('.','').isdigit() or True]
            desc = re.sub(r'^\d+\s+', '', l)
            desc = re.sub(r'\s+\d{4,}.*$', '', desc).strip()

            qty_m = re.search(r'(\d+(?:\.\d+)?)\s*(?:NOS|PCS|KG|MTR|SET|BOX|LTR|MTS|UNIT)\.?', l, re.IGNORECASE)
            qty = float(qty_m.group(1)) if qty_m else None
            unit_m = re.search(r'\b(NOS|PCS|KG|MTR|SET|BOX|LTR|MTS|UNIT)\b', l, re.IGNORECASE)
            unit = unit_m.group(1).upper() if unit_m else ''
            amount = nums[-1] if nums else None

            if desc and len(desc) > 3 and amount:
                items.append({'description': desc, 'quantity': qty, 'unit': unit, 'amount': amount})

    if not items:
        return [], f"No product rows found. Date={date_val}, Customer={customer}. Please set GEMINI_API_KEY for AI-powered parsing."

    rows = [{'date': date_val, 'invoice_no': invoice_no, 'customer': customer or 'UNKNOWN',
             'product': it['description'], 'quantity': it['quantity'],
             'unit': it['unit'], 'amount': it['amount']} for it in items]
    return rows, None


# ─────────────────────────────────────────────────────────────
# MULTI-INVOICE INGESTION
# ─────────────────────────────────────────────────────────────

def ingest_multiple_invoices(file_list, api_key=None):
    """
    Parse a list of (filename, bytes) tuples.
    Returns (combined_df, list_of_errors).
    """
    all_rows, errors = [], []
    for fname, fbytes in file_list:
        rows, err = parse_gst_invoice_pdf(fbytes, api_key=api_key)
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
