# Invoice Extraction with Vision-Language Model (VLM)

A modular system for extracting structured data from invoice images using pre-trained Vision-Language Models (VLM). This project provides both batch processing capabilities and a REST API for real-time invoice data extraction.

## Features

- 🔍 **VLM-based Extraction**: Uses HuggingFaceTB/SmolVLM-Instruct for accurate invoice data extraction
- 📊 **Batch Processing**: Process multiple invoice images efficiently
- 🌐 **REST API**: Real-time extraction via HTTP endpoints
- 📈 **Evaluation Pipeline**: Compare model results with ground truth data
- 🎯 **Modular Design**: Clean, maintainable code structure
- 📝 **Comprehensive Fields**: Extracts 14 different invoice fields including customer details, amounts, and dates

## Extracted Fields

- `client_name` - Client/company name
- `client_address` - Client mailing address
- `seller_name` - Seller/vendor name
- `seller_address` - Seller address
- `invoice_number` - Unique invoice identifier
- `invoice_date` - Invoice date (YYYY-MM-DD format)
- `tax` - Tax amount
- `discount` - Discount amount
- `total` - Total invoice amount
- `cust_number` - Customer number
- `total_open_amount` - Total open amount
- `business_code` - Business code
- `cust_payment_terms` - Payment terms
- `invoice_currency` - Currency code

## Project Structure

```
invoice-extraction/
├── config.py          # Configuration and constants
├── utils.py           # Utility functions
├── model.py           # VLM model and extraction logic
├── api.py             # FastAPI application
├── extract.py         # Main extraction script
├── evaluate.py        # Evaluation script
├── requirements.txt   # Python dependencies
├── README.md         # This file
└── outputs/          # Generated output files
```

## Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd invoice-extraction
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Update Configuration
Edit `config.py` to set your data paths:
```python
# Update these paths for your environment
DATA_DIR = Path("/path/to/your/data")
CSV_FILE = DATA_DIR / "your_data.csv"
IMAGE_DIR = DATA_DIR / "images"
```

## Usage

### 1. Single Image Extraction

Extract data from a single invoice image:

```bash
python extract.py --mode single --image-path /path/to/invoice.jpg
```

### 2. Batch Processing

Process all images in a directory:

```bash
python extract.py --mode batch --image-dir /path/to/images --output results.csv
```

### 3. Start API Server

Launch the REST API:

```bash
python extract.py --mode api --host 0.0.0.0 --port 8000
```

Or directly:

```bash
python api.py
```

### 4. Evaluate Model Performance

Compare model results with ground truth:

```bash
python evaluate.py --gt-csv ground_truth.csv --model-results extracted_data.csv --output-dir outputs
```

## API Usage

### Start the API
```bash
python api.py
```

The API will be available at `http://localhost:8000`

### API Endpoints

#### 1. Health Check
```bash
curl http://localhost:8000/health
```

#### 2. Extract from Uploaded Image
```bash
curl -X POST "http://localhost:8000/extract_invoice" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@invoice.jpg"
```

#### 3. Extract from Base64 Image
```bash
curl -X POST "http://localhost:8000/extract_invoice_base64" \
     -H "accept: application/json" \
     -H "Content-Type: application/json" \
     -d '{"image": "base64_encoded_image_string"}'
```

#### 4. Extract from File Path
```bash
curl -X POST "http://localhost:8000/extract_invoice_path" \
     -H "accept: application/json" \
     -H "Content-Type: application/json" \
     -d '{"path": "/path/to/invoice.jpg"}'
```

### Interactive API Documentation
Visit `http://localhost:8000/docs` for Swagger UI documentation.

## Python API Usage

### Single Image Extraction
```python
from model import InvoiceExtractor

extractor = InvoiceExtractor()
result = extractor.extract_single_image("invoice.jpg")
print(result)
```

### Batch Processing
```python
from model import process_images_batch

results_df = process_images_batch("/path/to/images", "output.csv")
print(f"Processed {len(results_df)} images")
```

### API Client
```python
import requests

# Upload file
with open('invoice.jpg', 'rb') as f:
    files = {'file': f}
    response = requests.post('http://localhost:8000/extract_invoice', files=files)
    print(response.json())

# Base64
import base64
with open('invoice.jpg', 'rb') as f:
    image_b64 = base64.b64encode(f.read()).decode()
    data = {'image': image_b64}
    response = requests.post('http://localhost:8000/extract_invoice_base64', json=data)
    print(response.json())
```

## Configuration

### Model Settings (`config.py`)
```python
MODEL_NAME = "HuggingFaceTB/SmolVLM-Instruct"
DEVICE = "cuda"  # or "cpu"
MAX_NEW_TOKENS = 300
BATCH_SIZE = 5
```

### API Settings
```python
API_HOST = "0.0.0.0"
API_PORT = 8000
```

### Data Paths
```python
DATA_DIR = Path("/path/to/your/data")
CSV_FILE = DATA_DIR / "batch1_1.csv"
IMAGE_DIR = DATA_DIR / "batch1_1"
```

## Output Files

The system generates several output files:

- `extracted_data.csv` - Raw model extraction results
- `evaluation_results.csv` - Field-by-field accuracy results
- `mismatches.csv` - Detailed mismatch analysis

## Evaluation Metrics

The evaluation system provides:

- **Overall Accuracy**: Percentage of correctly extracted fields
- **Field-wise Accuracy**: Individual field performance
- **Mismatch Analysis**: Detailed comparison of ground truth vs predictions
- **Error Analysis**: Common failure patterns

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce `BATCH_SIZE` in `config.py`
   - Use CPU mode: set `DEVICE = "cpu"`

2. **Model Loading Issues**
   - Ensure internet connection for model download
   - Check available disk space

3. **API Connection Issues**
   - Verify port availability
   - Check firewall settings

4. **Image Processing Errors**
   - Ensure images are in supported formats (JPEG, PNG)
   - Check image file integrity

### Performance Tips

- Use GPU for faster processing
- Adjust batch size based on available memory
- Use SSD storage for large datasets
- Consider model quantization for deployment

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

[Add your license information here]

## Acknowledgments

- HuggingFace for the VLM model
- FastAPI for the web framework
- The open-source community for various dependencies 