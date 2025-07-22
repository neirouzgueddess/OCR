"""
Evaluation script for invoice extraction system
"""

import json
import pandas as pd
import argparse
from pathlib import Path
from typing import Dict, Any

from config import CSV_FILE, ALL_FIELDS, EXTRACTED_DATA_FILE, RESULTS_FILE, MISMATCHES_FILE
from utils import flatten_invoice_data, evaluate_extractions, save_results


def load_ground_truth(csv_file: str) -> pd.DataFrame:
    """Load and process ground truth data from CSV."""
    print(f"Loading ground truth data from {csv_file}...")
    
    df = pd.read_csv(csv_file)
    df["Json Data"] = df.apply(lambda x: json.loads(x["Json Data"]), axis=1)
    
    # Extract and flatten ground truth data
    rows = []
    for idx, row in df.iterrows():
        data = row['Json Data']
        flat_data = flatten_invoice_data(data)
        
        rows.append({
            "file_name": row["File Name"],
            "invoice_no": flat_data.get("invoice_number", ""),
            "requested_parameters": json.dumps(ALL_FIELDS),
            "requested_data": json.dumps(flat_data)
        })
    
    out_df = pd.DataFrame(rows)
    print(f"Loaded {len(out_df)} ground truth records")
    
    return out_df


def load_model_results(results_file: str) -> pd.DataFrame:
    """Load model extraction results."""
    print(f"Loading model results from {results_file}...")
    
    df_resp = pd.read_csv(results_file)
    df_resp.columns = ["file_name", "vlm_response"]
    
    # Parse JSON responses
    from utils import safe_load_json
    df_resp["vlm_response"] = df_resp["vlm_response"].apply(safe_load_json)
    
    print(f"Loaded {len(df_resp)} model results")
    
    return df_resp


def run_evaluation(gt_csv: str, model_results: str, output_dir: str = "outputs"):
    """Run complete evaluation pipeline."""
    print("Starting evaluation...")
    
    # Load data
    ground_truth_df = load_ground_truth(gt_csv)
    model_results_df = load_model_results(model_results)
    
    # Run evaluation
    results = evaluate_extractions(model_results_df, ground_truth_df)
    
    # Save results
    save_results(results, output_dir)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate invoice extraction model")
    parser.add_argument(
        "--gt-csv", 
        type=str, 
        default=CSV_FILE,
        help="Path to ground truth CSV file"
    )
    parser.add_argument(
        "--model-results", 
        type=str, 
        default=EXTRACTED_DATA_FILE,
        help="Path to model results CSV file"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="outputs",
        help="Output directory for results"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not Path(args.gt_csv).exists():
        raise FileNotFoundError(f"Ground truth CSV not found: {args.gt_csv}")
    
    if not Path(args.model_results).exists():
        raise FileNotFoundError(f"Model results CSV not found: {args.model_results}")
    
    # Run evaluation
    results = run_evaluation(args.gt_csv, args.model_results, args.output_dir)
    
    print("\nEvaluation completed successfully!")


if __name__ == "__main__":
    main() 