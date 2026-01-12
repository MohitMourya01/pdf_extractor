import base64
import csv
import io
import json
import os
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from dotenv import load_dotenv
from pdf2image import convert_from_bytes
import pdfplumber
from mistralai import Mistral


# =======================
# Load environment
# =======================
load_dotenv()

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", "mistral-large-2512")
POPPLER_PATH = os.getenv("POPPLER_PATH", "C:\\poppler-25.12.0\\Library\\bin")  # Optional for Windows pdf2image

if not MISTRAL_API_KEY:
    raise RuntimeError("Mistral API Key missing in .env")

client = Mistral(api_key=MISTRAL_API_KEY)


# =======================
# JSON save directory
# =======================
SAVE_DIR = Path("extracted_json")
SAVE_DIR.mkdir(exist_ok=True)

# =======================
# Images save directory
# =======================
SAVE_IMG_DIR = Path("extracted_images")
SAVE_IMG_DIR.mkdir(exist_ok=True)


# =======================
# PDF → images (300 DPI)
# =======================
def pdf_to_images(file_bytes: bytes, dpi: int = 450) -> list[io.BytesIO]:
    """Convert PDF bytes to a list of in-memory PNG images at the given DPI with preprocessing."""
    if os.name == "nt":
        if not POPPLER_PATH:
            raise RuntimeError("POPPLER_PATH not set. Set it to the folder that contains pdftoppm.exe.")
        poppler_dir = Path(POPPLER_PATH)
        if not poppler_dir.exists():
            raise RuntimeError(f"POPPLER_PATH does not exist: {POPPLER_PATH}")
        pdftoppm = poppler_dir / "pdftoppm.exe"
        if not pdftoppm.exists():
            raise RuntimeError(f"pdftoppm.exe not found in POPPLER_PATH: {POPPLER_PATH}")

    pil_images = convert_from_bytes(file_bytes, dpi=dpi, poppler_path=POPPLER_PATH)
    images: list[io.BytesIO] = []
    
    for page_num, img in enumerate(pil_images, start=1):
        # Apply image preprocessing for better OCR
        from PIL import ImageEnhance
        
        # Enhance contrast (helps with faded text)
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.5)
        
        # Enhance sharpness (helps with blurry/overlapping text)
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(2.0)
        
        # Save enhanced image to disk
        img_filename = f"extract_{uuid.uuid4().hex[:8]}_page_{page_num}.png"
        img_path = SAVE_IMG_DIR / img_filename
        img.save(img_path, format="PNG")
        
        # Create buffer for processing
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        images.append(buf)
    return images


# =======================
# PDF → text per page (fallback when poppler unavailable)
# =======================
def pdf_to_text_pages(file_bytes: bytes) -> list[str]:
    """Extract text per page as fallback (uses pdfplumber)."""
    pages: list[str] = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            text = (page.extract_text() or "").strip()
            pages.append(text)
    return pages


# =======================
# Helpers
# =======================
def _encode_image_b64(img_buf: io.BytesIO) -> str:
    """Encode an image buffer to base64 string."""
    return base64.b64encode(img_buf.getvalue()).decode("utf-8")


def _parse_json_response(raw: str) -> dict:
    """Try to coerce an LLM response into valid JSON with robust error handling."""
    cleaned = raw.strip()

    # Remove markdown code fences if present
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`").replace("json", "", 1).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        # Try to fix common JSON errors
        fixed = cleaned
        
        # Remove trailing commas before closing braces/brackets
        import re
        fixed = re.sub(r',(\s*[}\]])', r'\1', fixed)
        
        # Try to close unclosed strings
        # If the string ends with a key-value pair where the value string is unclosed
        if fixed.count('"') % 2 == 1:  # Odd number of quotes, so unclosed string
            fixed += '"'
        
        # Try closing unclosed brackets
        if fixed.count("{") > fixed.count("}"):
            fixed += "}" * (fixed.count("{") - fixed.count("}"))
        
        if fixed.count("[") > fixed.count("]"):
            fixed += "]" * (fixed.count("[") - fixed.count("]"))
        
        try:
            return json.loads(fixed)
        except json.JSONDecodeError:
            # Last resort: try to extract JSON from the response
            # Look for first { and last }
            start = fixed.find('{')
            end = fixed.rfind('}')
            if start != -1 and end != -1 and end > start:
                json_str = fixed[start:end+1]
                # Remove trailing commas again
                json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
                # Close unclosed strings in the extracted part
                if json_str.count('"') % 2 == 1:
                    json_str += '"'
                try:
                    return json.loads(json_str)
                except:
                    pass
            
            # If all else fails, return error info
            print(f"JSON Parse Error: {e}")
            print(f"Raw response (first 500 chars): {raw[:500]}")
            return {
                "error": "Failed to parse JSON",
                "error_details": str(e),
                "raw_response_preview": raw[:200]
            }




# =======================
# LLM: Vision per-page JSON
# =======================
def call_mistral_page_json(image_b64: str, page_number: int, total_pages: int) -> dict:
    prompt = f"""
You are an expert OCR system. Extract data from page {page_number} of {total_pages} WITHOUT assuming any fixed schema.

DISCOVER STRUCTURE DYNAMICALLY:
- Identify document type automatically (invoice, bill, form, report, prescription, etc.)
- Detect all sections, headers, tables, and data blocks
- Use ACTUAL field names/labels found in the document
- Do NOT assume predefined field names or structure

HANDLE PROBLEMATIC TEXT:

1. OVERLAPPING TEXT:
   - Extract the clearest/most readable version
   - If both equally visible: "text1 / text2"
   - Mark with "overlapping_detected": true in metadata

2. COLUMN OVERFLOW:
   - Extract complete text even if it crosses boundaries
   - Use spatial positioning and context for column assignment
   - For tables: use row alignment to match data

3. MISSING DATA:
   - Completely absent → null
   - Unreadable/illegible → null
   - Empty cells → null (not empty string "")
   - NEVER guess or fabricate

4. TABLE EXTRACTION:
   - Auto-detect column headers from the image
   - Extract all rows with their values
   - Use null for empty/missing cells
   - Preserve row order and structure

5. DATA VALIDATION:
   - Check date logic (manufacturing < expiry)
   - Validate number formats
   - If suspicious, set to null and add to "validation_warnings"

OUTPUT STRUCTURE:
- Flexible JSON based on actual document content
- Use descriptive keys from document labels
- Include metadata:
  * "document_type": auto-detected type
  * "page_number": {page_number}
  * "extraction_confidence": 0.0-1.0

Return ONLY valid JSON. No markdown, no explanations.
"""

    response = client.chat.complete(
        model=MISTRAL_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are an expert OCR system that discovers document structure dynamically. Handle overlapping text, column overflow, and missing data. Return ONLY valid JSON with null for missing fields. Never assume fixed schemas.",
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": f"data:image/png;base64,{image_b64}"},
                ],
            },
        ],
        temperature=0,
    )

    raw = response.choices[0].message.content
    return _parse_json_response(raw)


def call_mistral_page_json_text(page_text: str, page_number: int, total_pages: int) -> dict:
    prompt = f"""
You are an expert text extraction system. Extract data from page {page_number} of {total_pages} WITHOUT assuming any fixed schema.

DISCOVER STRUCTURE DYNAMICALLY:
- Identify document type from text patterns
- Detect sections, headers, tables from text structure
- Use ACTUAL field names/labels found in the text
- Do NOT assume predefined field names

HANDLE TEXT ISSUES:

1. OVERLAPPING/GARBLED TEXT:
   - Extract most coherent interpretation
   - If completely garbled → null
   - Look for patterns and context

2. COLUMN MISALIGNMENT:
   - Use whitespace patterns to infer columns
   - Match data based on typical document structure
   - For tables: align by patterns (amounts right-aligned, etc.)

3. MISSING DATA:
   - Missing fields → null (not empty string)
   - Do not guess or extrapolate
   - Partial data: extract what's available

4. VALIDATION:
   - Check date logic
   - Validate number formats
   - If suspicious → null + add to "validation_warnings"

OUTPUT STRUCTURE:
- Flexible JSON based on actual text content
- Use descriptive keys from text labels
- Include metadata:
  * "document_type": auto-detected
  * "page_number": {page_number}
  * "extraction_confidence": 0.0-1.0

Page text:
{page_text}

Return ONLY valid JSON. No markdown, no explanations.
"""

    response = client.chat.complete(
        model=MISTRAL_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are an expert text extraction system that discovers document structure dynamically. Handle malformed text, misalignment, and missing data. Return ONLY valid JSON with null for missing fields. Never assume fixed schemas.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )

    raw = response.choices[0].message.content
    return _parse_json_response(raw)


# =======================
# Smart Item Deduplication
# =======================
def get_item_fingerprint(item: dict) -> str:
    """
    Create a fingerprint for an item based on key identifying fields.
    This is more robust than JSON string comparison.
    """
    # Key fields that typically identify unique items
    key_fields = ['name', 'item_name', 'description', 'product', 'particular', 
                  'item_code', 'sku', 'product_code', 'hsn', 'hsn_code']
    
    # Quantity and price fields for additional matching
    quantity_fields = ['quantity', 'qty', 'amount', 'units']
    price_fields = ['price', 'rate', 'unit_price', 'amount', 'total']
    
    fingerprint_parts = []
    
    # Extract key identifying fields
    for field in key_fields:
        if field in item:
            value = str(item[field]).strip().lower()
            if value:
                fingerprint_parts.append(f"{field}:{value}")
    
    # Add quantity if available
    for field in quantity_fields:
        if field in item and item[field] is not None:
            fingerprint_parts.append(f"qty:{item[field]}")
            break
    
    # Add price if available
    for field in price_fields:
        if field in item and item[field] is not None:
            fingerprint_parts.append(f"price:{item[field]}")
            break
    
    # If no fingerprint parts found, fall back to full JSON
    if not fingerprint_parts:
        return json.dumps(item, sort_keys=True)
    
    return "|".join(fingerprint_parts)


def deduplicate_items(items: list[dict], strategy: str = 'smart') -> tuple[list[dict], dict]:
    """
    Deduplicate items with configurable strategy.
    
    Args:
        items: List of item dictionaries
        strategy: 'strict' (exact match), 'smart' (fingerprint), or 'preserve' (no dedup)
    
    Returns:
        Tuple of (deduplicated_items, stats)
    """
    if strategy == 'preserve':
        return items, {'original_count': len(items), 'deduplicated_count': len(items), 'duplicates_removed': 0}
    
    seen = set()
    deduplicated = []
    duplicates_count = 0
    
    for item in items:
        if not isinstance(item, dict):
            # Non-dict items, use simple comparison
            if item not in seen:
                seen.add(item)
                deduplicated.append(item)
            else:
                duplicates_count += 1
            continue
        
        if strategy == 'strict':
            # Exact JSON match
            item_key = json.dumps(item, sort_keys=True)
        else:  # 'smart'
            # Fingerprint-based matching
            item_key = get_item_fingerprint(item)
        
        if item_key not in seen:
            seen.add(item_key)
            deduplicated.append(item)
        else:
            duplicates_count += 1
    
    stats = {
        'original_count': len(items),
        'deduplicated_count': len(deduplicated),
        'duplicates_removed': duplicates_count
    }
    
    return deduplicated, stats


# =======================
# Clean Result Accumulation
# =======================
def accumulate_results(per_page_json: list[dict], dedup_strategy: str = 'strict') -> dict:
    """
    Cleanly accumulate results from all pages into a structured JSON.
    
    Args:
        per_page_json: List of JSON objects, one per page
        dedup_strategy: 'strict' (exact match), 'smart' (fingerprint), or 'preserve' (no dedup)
    
    Returns:
        Clean JSON structure with headers, items, and metadata
    """
    if not per_page_json:
        return {
            "headers": {},
            "items": [],
            "metadata": {
                "total_pages": 0,
                "extraction_confidence": 0.0
            }
        }
    
    # Initialize result structure
    result = {
        "headers": {},
        "items": [],
        "metadata": {
            "total_pages": len(per_page_json),
            "extraction_methods": [],
            "page_summary": []
        }
    }
    
    # Field patterns for categorization
    header_field_patterns = [
        'invoice_number', 'invoice_no', 'bill_number', 'bill_no', 'document_number',
        'date', 'invoice_date', 'bill_date', 'issue_date', 'document_date',
        'po_number', 'po_no', 'purchase_order',
        'vendor', 'seller', 'supplier', 'from', 'vendor_name', 'seller_name',
        'customer', 'buyer', 'client', 'to', 'customer_name', 'buyer_name',
        'vendor_address', 'seller_address', 'customer_address', 'buyer_address',
        'gstin', 'gst_number', 'tax_id', 'vendor_gstin', 'customer_gstin',
        'total', 'grand_total', 'total_amount', 'amount_due', 'final_amount',
        'subtotal', 'sub_total',
        'tax', 'gst', 'sgst', 'cgst', 'igst', 'total_tax', 'tax_amount',
        'discount', 'total_discount',
        'bank_name', 'bank', 'account_number', 'account_no', 'acc_no',
        'ifsc', 'ifsc_code', 'swift', 'swift_code',
        'branch', 'branch_name', 'bank_branch'
    ]
    
    items_field_patterns = ['items', 'line_items', 'products', 'particulars', 'details']
    metadata_field_patterns = ['document_type', 'extraction_confidence', 'extraction_notes', 
                               'validation_warnings', 'overlapping_detected', 'page_number']
    
    all_items = []
    items_key = 'items'  # Standard key name
    max_confidence = 0.0
    document_type = None
    
    # Process each page
    for page_idx, page in enumerate(per_page_json):
        page_items_count = 0
        page_has_items = False
        
        for key, value in page.items():
            # Skip internal tracking fields
            if key.startswith('_') and key not in ['_extraction_method', '_page_number']:
                continue
            
            key_lower = key.lower()
            
            # Check if it's an items field
            is_items_field = False
            if isinstance(value, list) and value:
                for items_pattern in items_field_patterns:
                    if items_pattern in key_lower:
                        is_items_field = True
                        items_key = key  # Remember the original key name
                        break
            
            if is_items_field:
                # Accumulate items from all pages
                page_items_count = len(value)
                page_has_items = True
                all_items.extend(value)
            # Check if it's metadata
            elif any(mf in key_lower for mf in metadata_field_patterns):
                if key == 'extraction_confidence' and isinstance(value, (int, float)):
                    max_confidence = max(max_confidence, float(value))
                elif key == 'document_type' and value and not document_type:
                    document_type = value
            # Check if it's a header field
            elif any(hf in key_lower for hf in header_field_patterns):
                # Take from first page, or update if current page has better data
                if key not in result["headers"] or result["headers"][key] is None:
                    result["headers"][key] = value
            # Nested dictionaries - add to headers
            elif isinstance(value, dict):
                if key not in result["headers"]:
                    result["headers"][key] = value
            # Default: add to headers if not already present
            else:
                if key not in result["headers"]:
                    result["headers"][key] = value
        
        # Track page processing info
        page_info = {
            'page_number': page.get('_page_number', page.get('page_number', page_idx + 1)),
            'extraction_method': page.get('_extraction_method', 'unknown'),
            'items_count': page_items_count,
            'has_items': page_has_items
        }
        result["metadata"]["page_summary"].append(page_info)
        
        # Track extraction methods
        extraction_method = page.get('_extraction_method', 'unknown')
        if extraction_method not in result["metadata"]["extraction_methods"]:
            result["metadata"]["extraction_methods"].append(extraction_method)
    
    # Deduplicate items
    if all_items:
        print(f"Total items collected from all pages: {len(all_items)}")
        deduplicated_items, stats = deduplicate_items(all_items, strategy=dedup_strategy)
        print(f"After {dedup_strategy} deduplication: {stats['deduplicated_count']} unique items (removed {stats['duplicates_removed']} duplicates)")
        
        result["items"] = deduplicated_items
        result["metadata"]["deduplication_stats"] = stats
    else:
        print("No items found in any page")
    
    # Set metadata
    result["metadata"]["extraction_confidence"] = max_confidence
    if document_type:
        result["metadata"]["document_type"] = document_type
    
    return result



# =======================
# Post-processing validation
# =======================
def validate_and_clean_extraction(data: dict) -> dict:
    """Clean and validate extracted data after OCR"""
    cleaned = data.copy()
    issues = []
    warnings = []
    
    # Clean overlapping text artifacts
    def clean_text_field(value):
        if isinstance(value, str):
            # Remove excessive spaces
            value = ' '.join(value.split())
            
            # Detect potential overlaps (marked with /)
            if '/' in value and len(value.split('/')) > 1:
                parts = value.split('/')
                # If parts are very similar, might be overlap artifact
                if len(parts) == 2 and parts[0].strip() and parts[1].strip():
                    issues.append(f"Potential text overlap detected: {value}")
        return value
    
    # Recursively clean all string fields
    def clean_recursive(obj):
        if isinstance(obj, dict):
            return {k: clean_recursive(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [clean_recursive(item) for item in obj]
        elif isinstance(obj, str):
            return clean_text_field(obj)
        return obj
    
    cleaned = clean_recursive(cleaned)
    
    # Validate dates in items if present
    if 'items' in cleaned and isinstance(cleaned['items'], list):
        for idx, item in enumerate(cleaned['items']):
            if isinstance(item, dict):
                # Check for date validation
                if 'expiry_date' in item and 'manufacturing_date' in item:
                    exp = item.get('expiry_date')
                    mfg = item.get('manufacturing_date')
                    if exp and mfg and exp < mfg:
                        warnings.append(f"Item {idx+1}: Expiry date before manufacturing date")
                        item['expiry_date'] = None
                        item['manufacturing_date'] = None
    

    return cleaned


# =======================
# Dual-pass extraction with confidence merging
# =======================
def merge_with_confidence(vision_result: dict, text_result: dict) -> dict:
    """
    Best practice merge:
    - Headers → text first
    - Tables → vision structure
    - Values → per-field confidence
    """
    merged = {}

    def resolve_field(v: dict | None, t: dict | None):
        if v and t:
            return t["value"] if t["confidence"] >= v["confidence"] else v["value"]
        return t["value"] if t else v["value"] if v else None

    for key in set(vision_result) | set(text_result):
        if key.startswith("_") or key in [
            "page_number",
            "extraction_confidence",
            "extraction_notes",
            "validation_warnings",
        ]:
            continue

        v_val = vision_result.get(key)
        t_val = text_result.get(key)

        # -------- TABLES (VISION OWNS STRUCTURE) --------
        if key in ["items", "line_items", "products"] and isinstance(v_val, list):
            merged_items = []

            for v_row in v_val:
                merged_row = {}
                for field, v_cell in v_row.items():
                    t_cell = None
                    if isinstance(t_val, list):
                        for t_row in t_val:
                            if field in t_row:
                                t_cell = t_row[field]
                                break

                    merged_row[field] = resolve_field(v_cell, t_cell)

                merged_items.append(merged_row)

            if merged_items:
                merged[key] = merged_items
            continue

        # -------- HEADERS (TEXT FIRST) --------
        if isinstance(t_val, dict) and "value" in t_val:
            merged[key] = resolve_field(v_val, t_val)
        elif t_val is not None:
            merged[key] = t_val
        elif v_val is not None:
            merged[key] = v_val

    # -------- METADATA --------
    merged["extraction_confidence"] = max(
        vision_result.get("extraction_confidence", 0.5),
        text_result.get("extraction_confidence", 0.5),
    )

    merged["_merge_strategy"] = (
        "text_headers + vision_tables + field_level_confidence"
    )

    return merged



# =======================
# Structured Response Formatter
# =======================
def structure_response(data: dict) -> dict:
    """
    Structure the extracted data into well-defined sections:
    - header: Document header information
    - items: Line items/products
    - bank_info: Banking details
    - metadata: Extraction metadata
    """
    structured = {
        "header": {},
        "items": [],
        "bank_info": {},
        "metadata": {}
    }
    
    # Common header field patterns
    header_fields = [
        'invoice_number', 'invoice_no', 'bill_number', 'bill_no', 'document_number',
        'date', 'invoice_date', 'bill_date', 'issue_date',
        'po_number', 'po_no', 'purchase_order',
        'vendor', 'seller', 'supplier', 'from', 'vendor_name', 'seller_name',
        'customer', 'buyer', 'client', 'to', 'customer_name', 'buyer_name',
        'vendor_address', 'seller_address', 'customer_address', 'buyer_address',
        'gstin', 'gst_number', 'tax_id', 'vendor_gstin', 'customer_gstin',
        'total', 'grand_total', 'total_amount', 'amount_due',
        'subtotal', 'sub_total',
        'tax', 'gst', 'sgst', 'cgst', 'igst', 'total_tax',
        'discount', 'total_discount'
    ]
    
    # Bank info field patterns
    bank_fields = [
        'bank_name', 'bank', 'account_number', 'account_no', 'acc_no',
        'ifsc', 'ifsc_code', 'swift', 'swift_code',
        'branch', 'branch_name', 'bank_branch'
    ]
    
    # Metadata fields (including deduplication stats)
    metadata_fields = [
        'document_type', 'page_number', 'total_pages',
        'extraction_confidence', 'extraction_notes', 'validation_warnings',
        'overlapping_detected', '_deduplication_stats', '_page_processing_summary'
    ]
    
    # Items field patterns
    items_fields = ['items', 'line_items', 'products', 'particulars', 'details']
    
    # Categorize fields
    for key, value in data.items():
        key_lower = key.lower()
        
        # Check if it's items - match both exact key and key_lower
        is_items_field = False
        if isinstance(value, list):
            # Check if key matches any items field pattern
            for items_field in items_fields:
                if items_field in key_lower or key_lower == items_field:
                    is_items_field = True
                    break
        
        if is_items_field:
            # Use 'items' as the standard key name
            if 'items' not in structured or not structured['items']:
                structured['items'] = value
            else:
                # Merge if there are multiple item arrays
                structured['items'].extend(value)
        # Check if it's bank info
        elif any(bank_field in key_lower for bank_field in bank_fields):
            structured['bank_info'][key] = value
        # Check if it's metadata
        elif key in metadata_fields:
            structured['metadata'][key] = value
        # Check if it's header info
        elif any(header_field in key_lower for header_field in header_fields):
            structured['header'][key] = value
        # If it's a nested dict that might contain bank/vendor info
        elif isinstance(value, dict):
            # Check if it contains bank info
            if any(bank_field in str(value).lower() for bank_field in bank_fields):
                structured['bank_info'][key] = value
            else:
                structured['header'][key] = value
        else:
            # Default to header
            structured['header'][key] = value
    
    # Clean up empty sections
    if not structured['bank_info']:
        del structured['bank_info']
    
    return structured


# =======================
# CSV Converter
# =======================
def convert_to_csv(data: dict) -> str:
    """Convert structured JSON data to CSV format"""
    output = io.StringIO()
    
    # Handle new structure format (headers, items, metadata)
    items = data.get('items', [])
    
    # If no items in new structure, try old structure format for backward compatibility
    if not items:
        # Try to find items in old format
        for value in data.values():
            if isinstance(value, list) and value and isinstance(value[0], dict):
                items = value
                break
    
    if not items:
        # No items found, return header info as CSV
        headers = data.get('headers', data)
        if isinstance(headers, dict):
            writer = csv.DictWriter(output, fieldnames=headers.keys())
            writer.writeheader()
            writer.writerow(headers)
        else:
            writer = csv.DictWriter(output, fieldnames=data.keys())
            writer.writeheader()
            writer.writerow(data)
    else:
        # Write items as CSV
        if items and isinstance(items[0], dict):
            fieldnames = list(items[0].keys())
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(items)
        else:
            # Simple list
            writer = csv.writer(output)
            for item in items:
                writer.writerow([item])
    
    return output.getvalue()




# =======================
# Helper functions for multi-threading
# =======================
def process_page(img_buf: io.BytesIO, page_number: int, total_pages: int) -> dict:
    """Process a single page image with Mistral vision."""
    img_b64 = _encode_image_b64(img_buf)
    return call_mistral_page_json(img_b64, page_number, total_pages)


def process_page_text(page_text: str, page_number: int, total_pages: int) -> dict | None:
    """Process a single page text with Mistral."""
    if not page_text:
        return None
    return call_mistral_page_json_text(page_text, page_number, total_pages)


# =======================
# Save JSON
# =======================
def save_json(data: dict) -> str:
    filename = f"extract_{uuid.uuid4().hex[:8]}.json"
    filepath = SAVE_DIR / filename
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    return str(filepath)


# =======================
# FastAPI setup
# =======================
app = FastAPI(
    title="Universal PDF → JSON Extractor",
    description="Upload ANY PDF and LLM extracts structured JSON.",
    version="1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def call_mistral_page_raw(image_b64: str, page_number: int, total_pages: int) -> str:
    prompt = f"""
You are an expert OCR system. Extract data from page {page_number} of {total_pages} WITHOUT assuming any fixed schema.

DISCOVER STRUCTURE DYNAMICALLY:
- Identify document type automatically (invoice, bill, form, report, prescription, etc.)
- Detect all sections, headers, tables, and data blocks
- Use ACTUAL field names/labels found in the document
- Do NOT assume predefined field names or structure

HANDLE PROBLEMATIC TEXT:

1. OVERLAPPING TEXT:
   - Extract the clearest/most readable version
   - If both equally visible: "text1 / text2"
   - Mark with "overlapping_detected": true in metadata

2. COLUMN OVERFLOW:
   - Extract complete text even if it crosses boundaries
   - Use spatial positioning and context for column assignment
   - For tables: use row alignment to match data

3. MISSING DATA:
   - Completely absent → null
   - Unreadable/illegible → null
   - Empty cells → null (not empty string "")
   - NEVER guess or fabricate

4. TABLE EXTRACTION:
   - Auto-detect column headers from the image
   - Extract all rows with their values
   - Use null for empty/missing cells
   - Preserve row order and structure

5. DATA VALIDATION:
   - Check date logic (manufacturing < expiry)
   - Validate number formats
   - If suspicious, set to null and add to "validation_warnings"

OUTPUT STRUCTURE:
- Flexible JSON based on actual document content
- Use descriptive keys from document labels
- Include metadata:
  * "document_type": auto-detected type
  * "page_number": {page_number}
  * "extraction_confidence": 0.0-1.0

Return ONLY valid JSON. No markdown, no explanations.
"""

    response = client.chat.complete(
        model=MISTRAL_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are an expert OCR system that discovers document structure dynamically. Handle overlapping text, column overflow, and missing data. Return ONLY valid JSON with null for missing fields. Never assume fixed schemas.",
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": f"data:image/png;base64,{image_b64}"},
                ],
            },
        ],
        temperature=0,
    )

    return response.choices[0].message.content


def call_mistral_page_raw_text(page_text: str, page_number: int, total_pages: int) -> str:
    prompt = f"""
You are an expert text extraction system. Extract data from page {page_number} of {total_pages} WITHOUT assuming any fixed schema.

DISCOVER STRUCTURE DYNAMICALLY:
- Identify document type from text patterns
- Detect sections, headers, tables from text structure
- Use ACTUAL field names/labels found in the text
- Do NOT assume predefined field names

HANDLE TEXT ISSUES:

1. OVERLAPPING/GARBLED TEXT:
   - Extract most coherent interpretation
   - If completely garbled → null
   - Look for patterns and context

2. COLUMN MISALIGNMENT:
   - Use whitespace patterns to infer columns
   - Match data based on typical document structure
   - For tables: align by patterns (amounts right-aligned, etc.)

3. MISSING DATA:
   - Missing fields → null (not empty string)
   - Do not guess or extrapolate
   - Partial data: extract what's available

4. VALIDATION:
   - Check date logic
   - Validate number formats
   - If suspicious → null + add to "validation_warnings"

OUTPUT STRUCTURE:
- Flexible JSON based on actual text content
- Use descriptive keys from text labels
- Include metadata:
  * "document_type": auto-detected
  * "page_number": {page_number}
  * "extraction_confidence": 0.0-1.0

Page text:
{page_text}

Return ONLY valid JSON. No markdown, no explanations.
"""

    response = client.chat.complete(
        model=MISTRAL_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are an expert text extraction system that discovers document structure dynamically. Handle malformed text, misalignment, and missing data. Return ONLY valid JSON with null for missing fields. Never assume fixed schemas.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )

    return response.choices[0].message.content


# =======================
# API Endpoint
# =======================
@app.post("/upload-pdf")
async def upload_pdf(request: Request, file: UploadFile = File(...)):
    # Validate file
    if file.content_type not in ["application/pdf", "application/octet-stream"]:
        raise HTTPException(status_code=400, detail="Only PDF files allowed")

    file_bytes = await file.read()

    per_page_json: list[dict] = []
    use_dual_pass = True   # Enable dual-pass extraction for better accuracy

    # Dual-pass extraction: Vision + Text
    if use_dual_pass:
        try:
            # Pass 1: Vision-based extraction
            
            images = pdf_to_images(file_bytes, dpi=450)
            if not images:
                raise RuntimeError("Could not render PDF pages")

            total_pages = len(images)
            print(f"Processing {total_pages} pages with dual-pass extraction...")
            
            # Get vision results (sequential processing)
            vision_results = []
            for page_num in range(1, total_pages + 1):
                result = process_page(images[page_num - 1], page_num, total_pages)
                vision_results.append(result)
                
            
            # Pass 2: Text-based extraction (sequential processing)
            pages_text = pdf_to_text_pages(file_bytes)
            text_results = []
            for page_num in range(1, total_pages + 1):
                result = process_page_text(pages_text[page_num - 1], page_num, total_pages)
                if result is not None:
                    text_results.append(result)
            
            # Merge vision and text results with confidence scoring
            for i in range(total_pages):
                vision_data = vision_results[i] if i < len(vision_results) else {}
                text_data = text_results[i] if i < len(text_results) else {}
                
                extraction_method = "unknown"
                
                # Check for errors in extraction
                if vision_data.get('error') and text_data.get('error'):
                    print(f"Warning: Both vision and text extraction failed for page {i+1}")
                    print(f"Vision error: {vision_data.get('error_details', 'Unknown')}")
                    print(f"Text error: {text_data.get('error_details', 'Unknown')}")
                    # Use whichever has partial data
                    merged_page = vision_data if len(vision_data) > 1 else text_data
                    extraction_method = "error_fallback"
                elif vision_data.get('error'):
                    print(f"Warning: Vision extraction failed for page {i+1}, using text extraction")
                    merged_page = text_data
                    extraction_method = "text_only"
                elif text_data.get('error'):
                    print(f"Warning: Text extraction failed for page {i+1}, using vision extraction")
                    merged_page = vision_data
                    extraction_method = "vision_only"
                elif vision_data and text_data:
                    merged_page = merge_with_confidence(vision_data, text_data)
                    extraction_method = "dual_pass"
                elif vision_data:
                    merged_page = vision_data
                    extraction_method = "vision_only"
                else:
                    merged_page = text_data
                    extraction_method = "text_only"
                
                # Add extraction method tracking
                merged_page['_extraction_method'] = extraction_method
                merged_page['_page_number'] = i + 1
                
                # Apply post-processing validation
                merged_page = validate_and_clean_extraction(merged_page)
                per_page_json.append(merged_page)
                
                print(f"Page {i+1}: Processed using {extraction_method}")
                
        except Exception as dual_err:
            print(f"Dual-pass failed: {dual_err}, falling back to single-pass vision")
            # Fallback to single-pass vision
            try:
                
                images = pdf_to_images(file_bytes, dpi=450)
                if not images:
                    raise RuntimeError("Could not render PDF pages")

                total_pages = len(images)
                per_page_json = []
                for page_num in range(1, total_pages + 1):
                    result = process_page(images[page_num - 1], page_num, total_pages)
                    result['_extraction_method'] = 'vision_fallback'
                    result['_page_number'] = page_num
                    per_page_json.append(result)
                    print(f"Page {page_num}: Processed using vision_fallback")
                    
                # Apply post-processing to each page
                per_page_json = [validate_and_clean_extraction(page) for page in per_page_json]
            except Exception as vision_err:
                # Final fallback: text extraction only
                pages_text = pdf_to_text_pages(file_bytes)
                if not any(pages_text):
                    raise HTTPException(status_code=500, detail=f"PDF processing failed: {vision_err}")

                total_pages = len(pages_text)
                results = []
                for page_num in range(1, total_pages + 1):
                    result = process_page_text(pages_text[page_num - 1], page_num, total_pages)
                    if result is not None:
                        result['_extraction_method'] = 'text_fallback'
                        result['_page_number'] = page_num
                        results.append(result)
                        print(f"Page {page_num}: Processed using text_fallback")
                per_page_json = [validate_and_clean_extraction(r) for r in results]
    else:
        # Original single-pass extraction (kept for compatibility)
        try:
        
            images = pdf_to_images(file_bytes, dpi=450)
            if not images:
                raise RuntimeError("Could not render PDF pages")

            total_pages = len(images)
            per_page_json = []
            for page_num in range(1, total_pages + 1):
                result = process_page(images[page_num - 1], page_num, total_pages)
                result['_extraction_method'] = 'vision_single_pass'
                result['_page_number'] = page_num
                per_page_json.append(result)
                print(f"Page {page_num}: Processed using vision_single_pass")
                per_page_json = [validate_and_clean_extraction(page) for page in per_page_json]
        except Exception as vision_err:
            # Fallback: text extraction per page
            pages_text = pdf_to_text_pages(file_bytes)
            if not any(pages_text):
                raise HTTPException(status_code=500, detail=f"PDF processing failed: {vision_err}")

            total_pages = len(pages_text)
            results = []
            for page_num in range(1, total_pages + 1):
                result = process_page_text(pages_text[page_num - 1], page_num, total_pages)
                if result is not None:
                    result['_extraction_method'] = 'text_single_pass'
                    result['_page_number'] = page_num
                    results.append(result)
                    print(f"Page {page_num}: Processed using text_single_pass")
            per_page_json = [validate_and_clean_extraction(r) for r in results]

    if not per_page_json:
        raise HTTPException(status_code=400, detail="No pages extracted")

    # Get deduplication strategy from query params (default: strict - only removes exact duplicates)
    dedup_strategy = request.query_params.get('dedup_strategy', 'strict')
    if dedup_strategy not in ['strict', 'smart', 'preserve']:
        dedup_strategy = 'strict'
    
    # Accumulate results cleanly from all pages
    accumulated_data = accumulate_results(per_page_json, dedup_strategy=dedup_strategy)
    
    # Apply final post-processing validation to accumulated data
    if accumulated_data.get("headers"):
        accumulated_data["headers"] = validate_and_clean_extraction(accumulated_data["headers"])
    if accumulated_data.get("items"):
        # Clean each item
        cleaned_items = []
        for item in accumulated_data["items"]:
            if isinstance(item, dict):
                cleaned_items.append(validate_and_clean_extraction(item))
            else:
                cleaned_items.append(item)
        accumulated_data["items"] = cleaned_items

    # Save JSON
    file_path = save_json(accumulated_data)
    
    # Check Accept header for response format
    accept_header = request.headers.get("accept", "application/json").lower()
    
    if "text/csv" in accept_header or "application/csv" in accept_header:
        # Return CSV format
        csv_data = convert_to_csv(accumulated_data)
        return Response(
            content=csv_data,
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename=extracted_{int(time.time())}.csv"
            }
        )
    else:
        # Return JSON format (default) - compact without extra whitespace
        response_data = {
            "message": "PDF processed successfully",
            "saved_to": file_path,
            "data": accumulated_data,
            "total_pages": len(per_page_json),
        }
        
        # Return compact JSON (no indentation, minimal whitespace)
        return Response(
            content=json.dumps(response_data, separators=(',', ':'), ensure_ascii=False),
            media_type="application/json",
            status_code=200
        )


@app.post("/upload-pdf-raw")
async def upload_pdf_raw(file: UploadFile = File(...)):
    # Validate file
    if file.content_type not in ["application/pdf", "application/octet-stream"]:
        raise HTTPException(status_code=400, detail="Only PDF files allowed")

    file_bytes = await file.read()

    raw_responses = []
    use_dual_pass = True   # Enable dual-pass extraction for better accuracy

    # Dual-pass extraction: Vision + Text
    if use_dual_pass:
        try:
            # Pass 1: Vision-based extraction
            
            images = pdf_to_images(file_bytes, dpi=450)
            if not images:
                raise RuntimeError("Could not render PDF pages")

            total_pages = len(images)
            print(f"Processing {total_pages} pages with dual-pass extraction...")
            
            # Get vision results (sequential processing)
            vision_raw = []
            for page_num in range(1, total_pages + 1):
                raw = call_mistral_page_raw(_encode_image_b64(images[page_num - 1]), page_num, total_pages)
                vision_raw.append(f"Page {page_num} (Vision):\n{raw}\n\n")
                
            
            # Pass 2: Text-based extraction (sequential processing)
            pages_text = pdf_to_text_pages(file_bytes)
            text_raw = []
            for page_num in range(1, total_pages + 1):
                if pages_text[page_num - 1]:
                    raw = call_mistral_page_raw_text(pages_text[page_num - 1], page_num, total_pages)
                    text_raw.append(f"Page {page_num} (Text):\n{raw}\n\n")
            
            # Combine vision and text raw responses
            for i in range(total_pages):
                raw_responses.append(vision_raw[i])
                if i < len(text_raw):
                    raw_responses.append(text_raw[i])
                
        except Exception as dual_err:
            print(f"Dual-pass failed: {dual_err}, falling back to single-pass vision")
            # Fallback to single-pass vision
            try:
                
                images = pdf_to_images(file_bytes, dpi=450)
                if not images:
                    raise RuntimeError("Could not render PDF pages")

                total_pages = len(images)
                for page_num in range(1, total_pages + 1):
                    raw = call_mistral_page_raw(_encode_image_b64(images[page_num - 1]), page_num, total_pages)
                    raw_responses.append(f"Page {page_num} (Vision Fallback):\n{raw}\n\n")
                    
            except Exception as vision_err:
                # Final fallback: text extraction only
                pages_text = pdf_to_text_pages(file_bytes)
                if not any(pages_text):
                    raise HTTPException(status_code=500, detail=f"PDF processing failed: {vision_err}")

                total_pages = len(pages_text)
                for page_num in range(1, total_pages + 1):
                    if pages_text[page_num - 1]:
                        raw = call_mistral_page_raw_text(pages_text[page_num - 1], page_num, total_pages)
                        raw_responses.append(f"Page {page_num} (Text Fallback):\n{raw}\n\n")
    else:
        # Original single-pass extraction (kept for compatibility)
        try:
        
            images = pdf_to_images(file_bytes, dpi=450)
            if not images:
                raise RuntimeError("Could not render PDF pages")

            total_pages = len(images)
            for page_num in range(1, total_pages + 1):
                raw = call_mistral_page_raw(_encode_image_b64(images[page_num - 1]), page_num, total_pages)
                raw_responses.append(f"Page {page_num} (Vision Single Pass):\n{raw}\n\n")
        except Exception as vision_err:
            # Fallback: text extraction per page
            pages_text = pdf_to_text_pages(file_bytes)
            if not any(pages_text):
                raise HTTPException(status_code=500, detail=f"PDF processing failed: {vision_err}")

            total_pages = len(pages_text)
            for page_num in range(1, total_pages + 1):
                if pages_text[page_num - 1]:
                    raw = call_mistral_page_raw_text(pages_text[page_num - 1], page_num, total_pages)
                    raw_responses.append(f"Page {page_num} (Text Single Pass):\n{raw}\n\n")

    if not raw_responses:
        raise HTTPException(status_code=400, detail="No pages processed")

    # Combine all raw responses into a single text
    full_text = "".join(raw_responses)

    # Return as text file
    return Response(
        content=full_text,
        media_type="text/plain",
        headers={
            "Content-Disposition": f"attachment; filename=raw_responses_{int(time.time())}.txt"
        }
    )


@app.get("/")
def home():
    return {"message": "Universal PDF → JSON API Running 🚀"}

