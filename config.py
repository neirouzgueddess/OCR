"""
Configuration file for Invoice Extraction System
Contains all constants, paths, and settings
"""

import os
from pathlib import Path

# Model Configuration
MODEL_NAME = "HuggingFaceTB/SmolVLM-Instruct"
DEVICE = "cuda"  # Will be set to "cpu" if CUDA not available
MAX_NEW_TOKENS = 300
BATCH_SIZE = 5

# Data Paths (update these for your environment)
DATA_DIR = Path("/kaggle/input/high-quality-invoice-images-for-ocr/batch_1/batch_1")
CSV_FILE = DATA_DIR / "batch1_1.csv"
IMAGE_DIR = DATA_DIR / "batch1_1"

# API Configuration
API_HOST = "0.0.0.0"
API_PORT = 8000
API_TITLE = "Invoice Extraction API"
API_DESCRIPTION = "Extract structured data from invoice images using VLM"

# Output Paths
OUTPUT_DIR = Path("outputs")
EXTRACTED_DATA_FILE = OUTPUT_DIR / "extracted_data.csv"
RESULTS_FILE = OUTPUT_DIR / "evaluation_results.csv"
MISMATCHES_FILE = OUTPUT_DIR / "mismatches.csv"

# Create output directory if it doesn't exist
OUTPUT_DIR.mkdir(exist_ok=True)

# Field Configuration
ALL_FIELDS = [
    "client_name", "client_address", "seller_name", "seller_address", 
    "invoice_number", "invoice_date", "tax", "discount", "total",
    "cust_number", "total_open_amount", "business_code", 
    "cust_payment_terms", "invoice_currency"
]

# Date formats for normalization
DATE_FORMATS = [
    "%Y-%m-%d", "%d-%m-%Y", "%m/%d/%Y", "%Y/%m/%d", 
    "%d/%m/%Y", "%m-%d-%Y", "%Y.%m.%d", "%d.%m.%Y"
]

# Prompt template
INVOICE_EXTRACTION_PROMPT = """
You are an intelligent invoice parser. From the provided invoice image, extract the following structured information and return it in valid JSON format. 
Follow the descriptions and formatting guidelines for each field exactly. 

Field Descriptions and Format Requirements:

client_name: The full name of the client (person or company) to whom the invoice is issued. Must be a single string without extra labels like "Bill To" or "Client:".
client_address: The complete mailing address of the client. It may span multiple lines. Include line breaks as \\n. Remove any label like "Address:" or "Client Address:".
seller_name: The full name of the seller or vendor (person or company) issuing the invoice. A single string only.
seller_address: The full address of the seller. It may span multiple lines. Include line breaks as \\n.
invoice_number: A unique identifier for the invoice. Often labeled as "Invoice No.", "Invoice #", or simply "No.". Extract the alphanumeric value exactly.
invoice_date: The date the invoice was issued. Must be in YYYY-MM-DD format. If the original format differs (e.g., DD-MM-YYYY, MM/DD/YYYY, or YYYY/MM/DD), convert it to YYYY-MM-DD.
tax: The tax amount applied on the invoice. Extract only the numeric part (e.g., 809.62) and preserve the original formatting for decimals and thousand separators if present (e.g., 1,234.56 or 809,62).
discount: Any discount applied. Use the same formatting as for tax. If not present, use "".
total: The gross total amount after applying tax and discount. This should reflect the final amount the client has to pay. Maintain the original number formatting (e.g., 8 905,77 or 8,905.77).
cust_number: The customer number as shown on the invoice. Use the exact string as it appears.
total_open_amount: The total open amount for the invoice. Extract only the numeric part and preserve the original formatting.
business_code: The business code associated with the invoice. Use the exact string as it appears.
cust_payment_terms: The customer payment terms as shown on the invoice. Use the exact string as it appears.
invoice_currency: The currency code for the invoice (e.g., USD, EUR). Use the exact string as it appears.

Expected JSON Output Format:
{
  "client_name": "",
  "client_address": "",
  "seller_name": "",
  "seller_address": "",
  "invoice_number": "",
  "invoice_date": "",
  "tax": "",
  "discount": "",
  "total": "",
  "cust_number": "",
  "total_open_amount": "",
  "business_code": "",
  "cust_payment_terms": "",
  "invoice_currency": ""
}

Guidelines:
- If any field is missing or not explicitly mentioned in the text, use an empty string ("") as its value.
- Do not hallucinate
""" 