"""
Main extraction script for invoice processing
"""

import argparse
import pandas as pd
from pathlib import Path
from typing import List
import json

from config import IMAGE_DIR, EXTRACTED_DATA_FILE, ALL_FIELDS
from model import InvoiceExtractor, process_images_batch
from utils import flatten_invoice_data


def extract_single_image(image_path: str) -> dict:
    """Extract data from a single image."""
    extractor = InvoiceExtractor()
    return extractor.extract_single_image(image_path)


def extract_directory(image_dir: str, output_file: str = None) -> pd.DataFrame:
    """Extract data from all images in a directory."""
    return process_images_batch(image_dir, output_file)


def main():
    parser = argparse.ArgumentParser(description="Invoice Extraction Tool")
    parser.add_argument(
        "--mode",
        choices=["single", "batch", "api"],
        default="batch",
        help="Extraction mode: single image, batch directory, or start API"
    )
    parser.add_argument(
        "--image-path",
        type=str,
        help="Path to single image file (for single mode)"
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        default=IMAGE_DIR,
        help="Directory containing images (for batch mode)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=EXTRACTED_DATA_FILE,
        help="Output CSV file path"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="API host (for api mode)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="API port (for api mode)"
    )
    
    args = parser.parse_args()
    
    if args.mode == "single":
        if not args.image_path:
            raise ValueError("--image-path is required for single mode")
        
        if not Path(args.image_path).exists():
            raise FileNotFoundError(f"Image file not found: {args.image_path}")
        
        print(f"Extracting data from: {args.image_path}")
        result = extract_single_image(args.image_path)
        print("Extracted data:")
        print(json.dumps(result, indent=2))
        
    elif args.mode == "batch":
        if not Path(args.image_dir).exists():
            raise FileNotFoundError(f"Image directory not found: {args.image_dir}")
        
        print(f"Processing images in: {args.image_dir}")
        results_df = extract_directory(args.image_dir, args.output)
        print(f"Processed {len(results_df)} images")
        print(f"Results saved to: {args.output}")
        
    elif args.mode == "api":
        print(f"Starting API server on {args.host}:{args.port}")
        import uvicorn
        from api import app
        uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main() 