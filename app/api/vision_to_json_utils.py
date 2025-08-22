import fitz  # PyMuPDF
from PIL import Image
import pytesseract
import requests
import json
import os
import cv2
import re
import numpy as np
from io import BytesIO
import base64
from typing import List, Dict
from json_repair import repair_json
from rapidfuzz import fuzz
from flask import request, jsonify
from werkzeug.utils import secure_filename
import tempfile

# CONFIGURATION
OLLAMA_API_BASE = "https://ollama2.havenify.ai/v1"
MODEL_NAME = "llama3.2-vision:latest"
ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg', 'tiff', 'bmp', 'gif'}
UPLOAD_FOLDER = tempfile.mkdtemp()

INVOICE_SCHEMA = {
    "invoice_number": "string",
    "issue_date": "string (YYYY-MM-DD)",
    "due_date": "string (YYYY-MM-DD)",
    "vendor_name": "string",
    "vendor_address": "string",
    "vendor_GSTIN_number": "string",
    "customer_name": "string",
    "customer_address": "string",
    "customer_GSTIN_number": "string",
    "items": [
        {
            "product_id": "number",
            "product_name": "string",
            "description": "string",
            "HSN_Code": "string",
            "quantity": "number",
            "unit_price": "number",
            "amount": "number",
            "total": "number",
            "uom": "string (unit of measure)",
            "taxes": [
                {
                    "code": "string (e.g., CGST, SGST, IGST)",
                    "rate": "number (percentage)",
                    "amount": "number"
                }
            ],
            "subTotal": "number",
            "taxTotal": "number",
            "total": "number"
        }
    ],
    "subtotal": "number",
    "tax": "number",
    "total_amount": "number",
    "currency": "string",
    "payment_terms": "string",
    "notes": "string"
}

INVOICE_KEYWORDS = [
    'invoice', 'invoice number','bill', 'amount due', 'total', 'subtotal', 'due date',
    'payment terms', 'tax', 'vat', 'gst', 'invoice no', 'inv-', 'po #',
    'purchase order', 'billing address', 'account number', 'payment due',
    'issued date', 'reference number', 'itemized', 'quantity', 'unit price',
    'description', 'line total', 'balance due', 'receipt', 'debit note'
]

CURRENCY_SYMBOLS = r'[$€£₹¥]'
INVOICE_ID_PATTERNS = [
    r'inv[-_\s]?\d+',
    r'invoice[-_\s]?\d+',
    r'#\d{4,}',
    r'bill[-_\s]?\d+'
]

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def is_supported_image(file_path: str) -> bool:
    ext = os.path.splitext(file_path)[1].lower()
    return ext in {'.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'}

def is_pdf(file_path: str) -> bool:
    return os.path.splitext(file_path)[1].lower() == '.pdf'

def preprocess_image_for_ocr(np_image):
    gray = cv2.cvtColor(np_image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def ocr_image(image: Image.Image) -> str:
    cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    processed = preprocess_image_for_ocr(cv_image)
    return pytesseract.image_to_string(processed)

def load_images_from_file(file_path: str) -> List[Image.Image]:
    images = []
    if is_pdf(file_path):
        doc = fitz.open(file_path)
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            mat = fitz.Matrix(200 / 72, 200 / 72)
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")
            img = Image.open(BytesIO(img_data))
            images.append(img)
        doc.close()
    elif is_supported_image(file_path):
        images = [Image.open(file_path)]
    else:
        raise ValueError(f"Unsupported file type: {file_path}")
    return images

def image_to_base64(image: Image.Image) -> str:
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


def extract_with_llm(image: Image.Image, ocr_text: str, page_num: int, schema: dict) -> Dict:
    url = f"{OLLAMA_API_BASE}/chat/completions"
    headers = {"Content-Type": "application/json"}
    schema_str = json.dumps(schema, indent=2)
    prompt = f"""
You are an expert document parser. Analyze the image and OCR text to extract accurate structured data. This document may have varying layouts, field names, and structures, but your task is to map the information to the provided schema.

**OCR Text (for reference):**
{ocr_text[:2000]}

**Instructions:**
- Return ONLY a JSON object matching the schema below.
- DO NOT wrap the response in markdown (no ```json or ```).
- DO NOT include explanations or notes.
- If a field is missing, use null.
- Format dates as YYYY-MM-DD.
- Parse numbers correctly (e.g., 1,234.56 → 1234.56).
- Escape quotes properly in strings.
- Ensure the JSON is valid and parseable.

**Schema:**
{schema_str}
"""
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_to_base64(image)}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 2048,
        "temperature": 0.1,
        "response_format": {"type": "json_object"}
    }
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        content = response.json()['choices'][0]['message']['content'].strip()
        if content.startswith("```json"):
            content = content[7:-3]
        elif content.startswith("```"):
            content = content[3:-3]
        try:
            result = repair_json(content, return_objects=True)
            if not isinstance(result, dict):
                raise ValueError("Not a valid dict")
        except Exception:
            result = {
                "error": "Failed to parse",
                "raw_response_truncated": content[:500] + "...",
                "source_page": page_num + 1
            }
    except Exception as e:
        result = {"error": "Request failed", "exception": str(e), "source_page": page_num + 1}
    result.setdefault("source_page", page_num + 1)
    return result


def vision_to_json_endpoint():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "File type not allowed"}), 400
    # Accept schema as JSON string in form-data or as a field named 'schema'
    schema = None
    if 'schema' in request.form:
        try:
            schema = json.loads(request.form['schema'])
        except Exception as e:
            return jsonify({"error": "Invalid schema JSON", "details": str(e)}), 400
    elif request.json and 'schema' in request.json:
        schema = request.json['schema']
    else:
        schema = INVOICE_SCHEMA
    try:
        filename = secure_filename(file.filename)
        temp_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(temp_path)
        # Only process first image/page
        if is_pdf(temp_path):
            images = load_images_from_file(temp_path)
            image = images[0]
        elif is_supported_image(temp_path):
            image = Image.open(temp_path)
        else:
            return jsonify({"error": "Unsupported file type"}), 400
        ocr_text = ocr_image(image)
        data = extract_with_llm(image, ocr_text, 0, schema)
        os.remove(temp_path)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": "Processing failed", "details": str(e)}), 500
