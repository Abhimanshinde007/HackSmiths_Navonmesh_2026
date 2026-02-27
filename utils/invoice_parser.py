"""
Gemini AI-Powered GST Invoice Parser
Reads PDF text page by page, sends each page to Gemini Flash to extract
structured invoice data. Falls back to regex if no API key.
"""

import io
import re
import json
import time
import pandas as pd

try:
    import pdfplumber
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

try:
    from google import genai
    from google.genai import types as genai_types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False



# ─────────────────────────────────────────────────────────────
# PDF READING
# ─────────────────────────────────────────────────────────────

def _read_pdf_pages(file_bytes):
    """Extract text page-by-page. Returns ([(page_num, text), ...], error)."""
    if not PDF_AVAILABLE:
        return [], "pdfplumber not installed."
    try:
        pages = []
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            if not pdf.pages:
                return [], "Empty PDF."
            for i, page in enumerate(pdf.pages):
                t = page.extract_text(x_tolerance=3, y_tolerance=3)
                if t and t.strip():
                    pages.append((i + 1, t.strip()))
        if not pages:
            return [], "Scanned PDF — no extractable text. Use text-based Tally export."
        return pages, None
    except Exception as e:
        return [], f"PDF read error: {str(e)}"


# ─────────────────────────────────────────────────────────────
# GEMINI AI
# ─────────────────────────────────────────────────────────────

GEMINI_PROMPT = """You are a data extraction assistant for Indian GST invoices.

Given the raw text extracted from a PDF containing ONE OR MORE Tally GST Invoices, extract the details for EACH invoice.
Return ONLY a valid JSON array of objects, with no markdown formatting and no explanation.

Required JSON structure:
[
  {
    "date": "DD-Mon-YY or DD/MM/YYYY string, or null",
    "invoice_no": "invoice number string, or null",
    "customer": "exact buyer company name from Buyer/Bill to section ONLY — NOT the seller, NOT vehicle number",
    "products": [
      {
        "description": "product or item description",
        "quantity": numeric_or_null,
        "unit": "NOS/PCS/KG/MTR/SET or empty string",
        "amount": numeric_line_item_amount_in_rupees_or_null
      }
    ]
  }
]

Critical rules:
- Extract EVERY invoice you find in the text as a separate object in the array.
- customer = the BUYER party only (comes after Buyer/Bill to label).
- EXCLUDE: SGST, CGST, IGST, Output Tax, Total rows from products.
- EXCLUDE: vehicle numbers (e.g. MH12CT1998), dispatch info, delivery terms from customer name.
- EXCLUDE: sub-description italic lines below main product.
- quantity = physical unit count (e.g. 30, 1000), NOT rupee amounts.
- amount = line item value in rupees (e.g. 9300.00, 12500.00).
- If no products found for an invoice, set products to [].
- Return ONLY the JSON array.

Text to parse:
"""


# Models available on this advanced API key
GEMINI_MODELS = [
    'gemini-2.5-flash',
    'gemini-2.0-flash-lite',
    'gemini-2.0-flash'
]

def _gemini_parse_all(text, api_key):
    """Parse entire text via Gemini in ONE call. Returns (list_of_dicts, error)."""
    try:
        client = genai.Client(api_key=api_key)
    except Exception as e:
        return None, f"Failed to init Gemini client: {e}"
        
    last_err = "No models succeeded"
    for model_name in GEMINI_MODELS:
        try:
            # We enforce JSON output at the API level
            response = client.models.generate_content(
                model=model_name,
                contents=GEMINI_PROMPT + text,
                config=genai_types.GenerateContentConfig(
                    temperature=0,
                    max_output_tokens=8192, 
                    response_mime_type="application/json",
                )
            )
            raw = response.text.strip()
            
            # Since we forced application/json, it should be pure json
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                # Attempt robust cleanup for common LLM JSON errors in large outputs
                raw_clean = re.sub(r'```(?:json)?\s*', '', raw)
                raw_clean = re.sub(r'\s*```', '', raw_clean)
                # Fix trailing commas
                raw_clean = re.sub(r',\s*([\]}])', r'\1', raw_clean)
                # Fix unescaped quotes inside strings (basic heuristic)
                raw_clean = re.sub(r'([a-zA-Z])"([a-zA-Z])', r'\1\"\2', raw_clean)
                
                try:
                    parsed = json.loads(raw_clean)
                except json.JSONDecodeError as nested_e:
                    err_pos = getattr(nested_e, 'pos', 0)
                    snippet = raw_clean[max(0, err_pos-40):min(len(raw_clean), err_pos+40)]
                    last_err = f"JSON format error from {model_name}: {nested_e}. Snippet: '...{snippet}...' "
                    continue

            if isinstance(parsed, dict):
                parsed = [parsed]
            return parsed, None

        except Exception as e:
            err_str = str(e)
            if '429' in err_str or 'quota' in err_str.lower():
                last_err = f"{model_name}: Quota exceeded"
                continue
            if '404' in err_str or 'not found' in err_str.lower():
                last_err = f"{model_name}: Model not found"
                continue
            last_err = f"API error ({model_name}): {err_str}"
            
    return None, last_err



# ─────────────────────────────────────────────────────────────
# MAIN PARSER
# ─────────────────────────────────────────────────────────────


SKIP_PRODUCT_KEYWORDS = [
    'sgst', 'cgst', 'igst', 'output cgst', 'output sgst',
    'total', 'grand total', 'amount in words', 'taxable value',
    'central tax', 'state tax', 'output tax'
]


def parse_gst_invoice_pdf(file_bytes, api_key=None):
    """
    Parse a Tally GST Invoice PDF (single or merged multi-invoice).
    Sends ALL text to Gemini in ONE request to avoid rate limits.
    Returns (list_of_row_dicts, error_string).
    """
    pages, err = _read_pdf_pages(file_bytes)
    if err:
        return [], err
    if not pages:
        return [], "Scanned PDF — no extractable text."

    full_text = '\n'.join(t for _, t in pages)

    # ── Gemini path: one call for everything ──────────────────────
    if api_key and GEMINI_AVAILABLE:
        invoices, err = _gemini_parse_all(full_text, api_key)
        
        if err:
            return [], err
            
        if not invoices:
            return [], "AI returned no data."

        all_rows = []
        for inv in invoices:
            customer   = str(inv.get('customer') or 'UNKNOWN').strip()
            date_val   = inv.get('date')
            invoice_no = str(inv.get('invoice_no') or 'UNKNOWN').strip()
            products   = inv.get('products') or []

            for p in products:
                desc = str(p.get('description') or '').strip()
                if not desc or len(desc) < 3:
                    continue
                if any(kw in desc.lower() for kw in SKIP_PRODUCT_KEYWORDS):
                    continue
                all_rows.append({
                    'date':       date_val,
                    'invoice_no': invoice_no,
                    'customer':   customer,
                    'product':    desc,
                    'quantity':   p.get('quantity'),
                    'unit':       str(p.get('unit') or ''),
                    'amount':     p.get('amount'),
                })

        if not all_rows:
            return [], "AI parsed invoices but found 0 valid products."

        return all_rows, None

    # ── Regex fallback (no API key) ─────────────────────────────
    return _regex_parse(full_text)



# ─────────────────────────────────────────────────────────────
# REGEX FALLBACK
# ─────────────────────────────────────────────────────────────

def _regex_parse(text):
    """Best-effort regex parser for when no Gemini key is set."""
    lines = text.splitlines()

    # Date
    m = re.search(r'\b(\d{1,2}[-](?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[-]\d{2,4})\b', text, re.IGNORECASE)
    date_val = m.group(1) if m else None
    if not date_val:
        m = re.search(r'\b(\d{1,2}[/]\d{1,2}[/]\d{4})\b', text)
        date_val = m.group(1) if m else None

    # Invoice number
    m = re.search(r'\b([A-Z]{1,6}[/\-]\d{2,4}[/\-]\d{2,4}[/\-]\d{1,6})\b', text)
    if not m:
        m = re.search(r'\b([A-Z]{1,6}[/\-]\d{1,6})\b', text)
    invoice_no = m.group(1) if m else 'UNKNOWN'

    # Customer
    customer = None
    SKIP = ['gstin', 'state', 'contact', 'udyam', 'iso', 'certified', 'invoice',
            'dated', 'voucher', 'mode', 'payment', 'buyer', 'bill to', 'dispatch',
            'destination', 'vehicle', 'lading', 'delivery', 'term', 'place', 'supply', 'road']
    buyer_section = False
    for line in lines:
        l, ll = line.strip(), line.strip().lower()
        if re.search(r'buyer\s*[\(\[]?\s*bill\s+to', ll) or ll.startswith('buyer') or 'bill to' in ll:
            buyer_section = True
            continue
        if buyer_section:
            if not l or len(l) < 5:
                continue
            if any(kw in ll for kw in SKIP):
                continue
            if re.match(r'^[A-Z]{2}\d{2}[A-Z]{2}\d{4}$', l):
                continue  # skip vehicle numbers
            l = re.sub(r'\s+\d{1,2}[-/]\w{3,9}[-/]\d{2,4}\s*$', '', l).strip()
            if l and len(l) > 5:
                customer = l
                break

    if not customer:
        for line in lines:
            if re.search(r'\b(PVT|LTD|PRIVATE|LIMITED|INDUSTRIES|ENTERPRISES|ENGINEERING)\b', line, re.IGNORECASE) and len(line.strip()) > 8:
                customer = line.strip()
                break

    # Items
    HEADER_KW = ['description', 'goods', 'hsn', 'sac', 'qty', 'quantity', 'amount', 'rate']
    hdr_idx = None
    for i, line in enumerate(lines):
        if sum(1 for kw in HEADER_KW if kw in line.lower()) >= 3:
            hdr_idx = i
            break

    items = []
    if hdr_idx is not None:
        for line in lines[hdr_idx + 1:]:
            l = line.strip()
            if not l:
                continue
            if any(kw in l.lower() for kw in SKIP_PRODUCT_KEYWORDS):
                continue
            if not re.search(r'\d{1,3}(?:,\d{3})*\.\d{2}\s*$', l):
                continue
            if re.match(r'^[\d\s%,\.]+$', l):
                continue

            nums = [float(n.replace(',', '')) for n in re.findall(r'[\d,]+\.\d{2}', l)]
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
        return [], "No product rows found. Set GEMINI_API_KEY in .streamlit/secrets.toml for AI-powered parsing."

    return [{'date': date_val, 'invoice_no': invoice_no, 'customer': customer or 'UNKNOWN',
             'product': it['description'], 'quantity': it['quantity'],
             'unit': it['unit'], 'amount': it['amount']} for it in items], None


# ─────────────────────────────────────────────────────────────
# MULTI-INVOICE INGESTION
# ─────────────────────────────────────────────────────────────

def ingest_multiple_invoices(file_list, api_key=None):
    """
    Parse a list of (filename, bytes) tuples as GST invoices.
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
