import io
import json
import os
import uuid
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from pypdf import PdfReader
from openai import OpenAI


# =======================
# Load environment
# =======================
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

if not OPENAI_API_KEY:
    raise RuntimeError("OpenAI API Key missing in .env")

client = OpenAI(api_key=OPENAI_API_KEY)


# =======================
# JSON save directory
# =======================
SAVE_DIR = Path("extracted_json")
SAVE_DIR.mkdir(exist_ok=True)


# =======================
# Extract text from PDF
# =======================
def extract_text_from_pdf(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
        text += "\n"
    return text.strip()


# =======================
# LLM: Convert PDF text → JSON
# =======================
def call_openai_universal_json(pdf_text: str) -> dict:
    prompt = f"""
You are an AI that extracts structured information from ANY type of PDF.

Return:
- A COMPLETE JSON object
- No explanation, no markdown
- JSON must end with proper closing brackets: }}
- If list, close with ].
- NEVER cut off the response.
- ALWAYS return fully valid JSON.

Extract meaningful fields like:
- invoice numbers
- parties involved
- items or products (if any)
- totals (if found)
- tables as lists
- metadata
- dates, names, identifiers
- bank details (if found)
- summary of the invoice

PDF Text:
{pdf_text}
"""

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": "You are an AI that extracts structured information from PDFs and returns only valid JSON."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    # Get full output
    raw = response.choices[0].message.content.strip()

    # Remove fences if added
    if raw.startswith("```"):
        raw = raw.strip("`").replace("json", "", 1).strip()

    # Try strict JSON load
    try:
        return json.loads(raw)

    except json.JSONDecodeError:
        # Attempt auto-fixing common issue: missing closing braces
        fixed = raw

        # Try closing last JSON bracket
        if fixed.count("{") > fixed.count("}"):
            fixed += "}" * (fixed.count("{") - fixed.count("}"))

        if fixed.count("[") > fixed.count("]"):
            fixed += "]" * (fixed.count("[") - fixed.count("]"))

        try:
            return json.loads(fixed)
        except:
            raise ValueError("❌ OpenAI did not return valid JSON:\n" + raw[:1000])



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

    # Extract text
    pdf_text = extract_text_from_pdf(file_bytes)

    if not pdf_text:
        raise HTTPException(status_code=400, detail="Could not extract PDF text")

    # LLM → JSON
    json_data = call_openai_universal_json(pdf_text)

    # Save JSON
    file_path = save_json(json_data)

    return JSONResponse(
        content={
            "message": "PDF processed successfully",
            "saved_to": file_path,
            "data": json_data
        },
        status_code=200
    )


@app.get("/")
def home():
    return {"message": "Universal PDF → JSON API Running 🚀"}

