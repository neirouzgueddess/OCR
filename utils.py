"""
Utility functions for invoice extraction system
"""

import json
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List, Union
from config import ALL_FIELDS, DATE_FORMATS


def normalize_number(val: Union[str, float, int]) -> Union[float, str]:
    """Normalize numeric values by removing currency symbols and converting to float."""
    if isinstance(val, str):
        val = val.replace(',', '.').replace('$', '').strip()
    try:
        return round(float(val), 2)
    except (ValueError, TypeError):
        return val


def normalize_date(val: str) -> str:
    """Convert any date string to YYYY-MM-DD format if possible."""
    if not isinstance(val, str) or not val.strip():
        return ""
    
    for fmt in DATE_FORMATS:
        try:
            dt = datetime.strptime(val.strip(), fmt)
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            continue
    return val  # Return as is if no format matched


def safe_load_json(x: Union[str, dict]) -> Union[dict, None]:
    """Safely load JSON from string or return dict as is."""
    if isinstance(x, str):
        try:
            return json.loads(x)
        except json.JSONDecodeError:
            return None
    return x


def extract_json_from_response(response_text: str) -> Dict[str, Any]:
    """Extract JSON from model response, handling extra text."""
    try:
        response_text = response_text.strip()
        # Find JSON in the response (in case there's extra text)
        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}') + 1
        
        if start_idx != -1 and end_idx != 0:
            json_str = response_text[start_idx:end_idx]
            return json.loads(json_str)
        else:
            return {"error": "No valid JSON found in response"}
    except json.JSONDecodeError as e:
        return {"error": f"Failed to parse JSON: {str(e)}", "raw_response": response_text}


def match_field(gt_val: Any, pred_val: Any, field: str = None) -> bool:
    """Compare ground truth and predicted values with appropriate normalization."""
    if field == "invoice_date":
        gt_val = normalize_date(gt_val)
        pred_val = normalize_date(pred_val)
        return gt_val == pred_val
    
    if field in ['tax', 'discount', 'total', 'total_open_amount']:
        gt_val = normalize_number(gt_val)
        pred_val = normalize_number(pred_val)
        return gt_val == pred_val
    
    if isinstance(gt_val, float) or isinstance(pred_val, float):
        return gt_val == pred_val
    
    return str(gt_val).strip() == str(pred_val).strip()


def flatten_invoice_data(data: Dict[str, Any]) -> Dict[str, str]:
    """Flatten invoice JSON data to extract required fields."""
    invoice = data.get("invoice", {})
    subtotal = data.get("subtotal", {})
    
    flat = {}
    for field in ALL_FIELDS:
        if field in invoice:
            flat[field] = invoice.get(field, "")
        elif field in subtotal:
            flat[field] = subtotal.get(field, "")
        else:
            flat[field] = ""
    
    return flat


def evaluate_extractions(df_resp: pd.DataFrame, out_df: pd.DataFrame) -> Dict[str, Any]:
    """Evaluate model extractions against ground truth."""
    df_resp = df_resp.merge(out_df, how='left', on="file_name")
    
    mismatches = []
    results = []
    total, correct = 0, 0
    
    for idx, row in df_resp.iterrows():
        file_name = row['file_name']
        gt_json = safe_load_json(row['requested_data'])
        pred_json = safe_load_json(row['vlm_response'])
        requested_fields = safe_load_json(row['requested_parameters'])
        
        if not all([gt_json, pred_json, requested_fields]):
            continue
            
        row_result = {'file_name': file_name}
        
        for field in requested_fields:
            try:
                gt_val = gt_json.get(field, "")
                pred_val = pred_json.get(field, "")
                
                matched = match_field(gt_val, pred_val, field)
                row_result[field] = int(matched)
                total += 1
                correct += int(matched)
                
                if not matched:
                    mismatches.append({
                        'file_name': file_name,
                        'field': field,
                        'gt_val': gt_val,
                        'pred_val': pred_val,
                    })
            except Exception as ex:
                row_result[field] = None
        
        results.append(row_result)
    
    accuracy = correct / total if total else 0
    
    return {
        'accuracy': accuracy,
        'total_fields': total,
        'correct_fields': correct,
        'results': pd.DataFrame(results),
        'mismatches': pd.DataFrame(mismatches)
    }


def save_results(results: Dict[str, Any], output_dir: str = "outputs"):
    """Save evaluation results to files."""
    from pathlib import Path
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Save results
    results['results'].to_csv(output_path / "evaluation_results.csv", index=False)
    results['mismatches'].to_csv(output_path / "mismatches.csv", index=False)
    
    # Print summary
    print(f"\nEvaluation Summary:")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Total Fields: {results['total_fields']}")
    print(f"Correct Fields: {results['correct_fields']}")
    print(f"Results saved to: {output_path}") 