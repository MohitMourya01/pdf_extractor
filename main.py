import base64
import io
import json
import os
import uuid
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from pdf2image import convert_from_bytes
import pdfplumber
from openai import OpenAI


# =======================
# Load environment
# =======================
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
POPPLER_PATH = os.getenv("POPPLER_PATH", "C:\\poppler-25.12.0\\Library\\bin")  # Optional for Windows pdf2image

if not OPENAI_API_KEY:
    raise RuntimeError("OpenAI API Key missing in .env")

client = OpenAI(api_key=OPENAI_API_KEY)


# =======================
# JSON save directory
# =======================
SAVE_DIR = Path("extracted_json")
SAVE_DIR.mkdir(exist_ok=True)


# =======================
# PDF → images (300 DPI)
# =======================
def pdf_to_images(file_bytes: bytes, dpi: int = 300) -> list[io.BytesIO]:
    """Convert PDF bytes to a list of in-memory PNG images at the given DPI."""
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
    for img in pil_images:
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
    """Try to coerce an LLM response into valid JSON."""
    cleaned = raw.strip()

    # Remove fences if added
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`").replace("json", "", 1).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        fixed = cleaned

        # Try closing last JSON bracket
        if fixed.count("{") > fixed.count("}"):
            fixed += "}" * (fixed.count("{") - fixed.count("}"))

        if fixed.count("[") > fixed.count("]"):
            fixed += "]" * (fixed.count("[") - fixed.count("]"))

        return json.loads(fixed)


def _merge_dicts(base: dict, new: dict) -> dict:
    """Merge two dicts recursively, concatenating lists and preferring existing values."""
    merged = dict(base)
    for key, value in new.items():
        if key not in merged or merged[key] is None:
            merged[key] = value
            continue

        if isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _merge_dicts(merged[key], value)
        elif isinstance(merged[key], list) and isinstance(value, list):
            merged[key] = merged[key] + value
        else:
            # keep existing non-null value; could also choose latest if needed
            pass
    return merged


# =======================
# LLM: Vision per-page JSON
# =======================
def call_openai_page_json(image_b64: str, page_number: int, total_pages: int) -> dict:
    prompt = f"""
You extract structured information from a single PDF page image.

Context:
- Page {page_number} of {total_pages}
- Document types may include invoices, bills, reports, forms, statements, prescriptions, tickets, IDs, etc.
Rules:
- Manufacturing date must always be earlier than expiry date.
- If column order conflicts, correct based on date logic.

Task:
- Read ONLY this page image.
- Extract all meaningful structured data you see (header, parties, dates, line items, totals, references, metadata).
- Be strict with numeric/dates; do not guess missing values.
- If a field is absent, omit it or set null.
- Include "page_number": {page_number} in the JSON.

Output rules (mandatory):
- Return ONLY valid JSON (no markdown, no extra text).
"""

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a vision model that returns ONLY valid JSON extracted from a page image.",
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                ],
            },
        ],
        temperature=0,
    )

    raw = response.choices[0].message.content
    return _parse_json_response(raw)


def call_openai_page_json_text(page_text: str, page_number: int, total_pages: int) -> dict:
    prompt = f"""
You extract structured information from a single PDF page (text only, no image).

Context:
- Page {page_number} of {total_pages}
- Document types may include invoices, bills, reports, forms, statements, prescriptions, tickets, IDs, etc.

Task:
- Read ONLY this page text.
- Extract all meaningful structured data you see (header, parties, dates, line items, totals, references, metadata).
- Be strict with numeric/dates; do not guess missing values.
- If a field is absent, omit it or set null.
- Include "page_number": {page_number} in the JSON.

Output rules (mandatory):
- Return ONLY valid JSON (no markdown, no extra text).

Page text:
{page_text}
"""

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You return ONLY valid JSON extracted from a page text.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )

    raw = response.choices[0].message.content
    return _parse_json_response(raw)


# =======================
# Merge page JSONs (code-based)
# =======================
def merge_pages(per_page_json: list[dict]) -> dict:
    """Merge a list of per-page JSON dicts into one document-level JSON."""
    merged: dict = {}
    for page in per_page_json:
        merged = _merge_dicts(merged, page)
    merged["pages"] = per_page_json
    return merged



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


# =======================
# API Endpoint
# =======================
@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    # Validate file
    if file.content_type not in ["application/pdf", "application/octet-stream"]:
        raise HTTPException(status_code=400, detail="Only PDF files allowed")

    file_bytes = await file.read()

    per_page_json: list[dict] = []

    # Try primary path: PDF → images → vision model
    try:
        images = pdf_to_images(file_bytes, dpi=300)
        print("hello")
        if not images:
            raise RuntimeError("Could not render PDF pages")

        total_pages = len(images)
        for idx, img_buf in enumerate(images, start=1):
            img_b64 = _encode_image_b64(img_buf)
            page_json = call_openai_page_json(img_b64, idx, total_pages)
            per_page_json.append(page_json)
    except Exception as vision_err:
        # Fallback: text extraction per page (no poppler needed)
        pages_text = pdf_to_text_pages(file_bytes)
        print("world")
        if not any(pages_text):
            raise HTTPException(status_code=500, detail=f"PDF processing failed: {vision_err}")

        total_pages = len(pages_text)
        for idx, page_text in enumerate(pages_text, start=1):
            if not page_text:
                continue
            page_json = call_openai_page_json_text(page_text, idx, total_pages)
            per_page_json.append(page_json)

    if not per_page_json:
        raise HTTPException(status_code=400, detail="No pages extracted")

    # Merge results (code-based)
    final_json = merge_pages(per_page_json)

    # Save JSON (keep both per-page and merged)
    payload = {"pages": per_page_json, "final": final_json}
    file_path = save_json(payload)

    return JSONResponse(
        content={
            "message": "PDF processed successfully",
            "saved_to": file_path,
            "data": final_json,
            "pages": per_page_json,
        },
        status_code=200
    )


@app.get("/")
def home():
    return {"message": "Universal PDF → JSON API Running 🚀"}

