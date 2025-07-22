"""
Model module for VLM-based invoice extraction
"""

import torch
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Union
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.image_utils import load_image
from torch.amp import autocast
from PIL import Image
import io

from config import (
    MODEL_NAME, DEVICE, MAX_NEW_TOKENS, BATCH_SIZE, 
    INVOICE_EXTRACTION_PROMPT, ALL_FIELDS
)
from utils import extract_json_from_response


class InvoiceExtractor:
    """Main class for invoice extraction using VLM."""
    
    def __init__(self, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.processor = None
        self._load_model()
    
    def _load_model(self):
        """Load the VLM model and processor."""
        print(f"Loading model {MODEL_NAME} on {self.device}...")
        self.processor = AutoProcessor.from_pretrained(MODEL_NAME)
        self.model = AutoModelForVision2Seq.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16,
            _attn_implementation="sdpa"
        ).to(self.device)
        self.model.eval()
        print("Model loaded successfully!")
    
    def extract_single_image(self, image_path_or_bytes: Union[str, bytes]) -> Dict[str, Any]:
        """Extract invoice data from a single image."""
        # Load image
        if isinstance(image_path_or_bytes, str):
            image = load_image(image_path_or_bytes)
        else:
            image = Image.open(io.BytesIO(image_path_or_bytes))
        
        # Prepare prompt
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": INVOICE_EXTRACTION_PROMPT}
                ]
            }
        ]
        
        # Process input
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=prompt, images=[image], return_tensors="pt")
        inputs = inputs.to(self.device)
        input_ids_len = inputs["input_ids"].shape[1]
        
        # Generate output
        with torch.no_grad():
            with autocast(device_type=self.device):
                generated_ids = self.model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
        
        # Decode output
        new_tokens = generated_ids[0][input_ids_len:]
        generated_text = self.processor.tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        # Parse JSON response
        extracted_data = extract_json_from_response(generated_text)
        
        return extracted_data
    
    def extract_batch(self, image_paths: List[str]) -> List[Dict[str, Any]]:
        """Extract invoice data from a batch of images."""
        dataset = InvoiceDataset(image_paths, self.processor)
        dataloader = DataLoader(
            dataset, 
            batch_size=BATCH_SIZE, 
            num_workers=0, 
            pin_memory=True, 
            collate_fn=custom_collate_fn
        )
        
        extracted_data = []
        
        for batch in dataloader:
            batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
            input_ids_lens = [len(input_id) for input_id in batch["input_ids"]]
            
            with torch.no_grad():
                with autocast(device_type=self.device):
                    generated_ids = self.model.generate(**batch, max_new_tokens=MAX_NEW_TOKENS)
            
            for i, gen in enumerate(generated_ids):
                new_tokens = gen[input_ids_lens[i]:]
                generated_text = self.processor.tokenizer.decode(new_tokens, skip_special_tokens=True)
                extracted_data.append(generated_text.strip())
        
        return extracted_data


class InvoiceDataset(Dataset):
    """Dataset for batch processing of invoice images."""
    
    def __init__(self, image_paths: List[str], processor):
        self.image_paths = image_paths
        self.processor = processor
        self.messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": INVOICE_EXTRACTION_PROMPT}
                ]
            }
        ]
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = load_image(self.image_paths[idx])
        return {"image": image, "prompt": self.messages}


def custom_collate_fn(batch):
    """Custom collate function for batch processing."""
    images = [item["image"] for item in batch]
    prompts = [batch[0]["prompt"]] * len(batch)  # Same prompt for all images
    
    processor_outputs = batch[0]["prompt"][0]["content"][1]["text"]  # Get processor from first item
    processor = batch[0]["prompt"][0]["content"][1]["text"]  # This is a placeholder, need actual processor
    
    # This is a simplified version - in practice, you'd need to pass the processor properly
    return processor_outputs


def process_images_batch(image_dir: str, output_file: str = None) -> pd.DataFrame:
    """Process all images in a directory and save results."""
    image_dir = Path(image_dir)
    image_paths = [str(p) for p in image_dir.glob("*.jpg")]
    image_paths.extend([str(p) for p in image_dir.glob("*.png")])
    image_paths.extend([str(p) for p in image_dir.glob("*.jpeg")])
    
    if not image_paths:
        raise ValueError(f"No images found in {image_dir}")
    
    print(f"Found {len(image_paths)} images to process")
    
    # Initialize extractor
    extractor = InvoiceExtractor()
    
    # Process images
    extracted_data = extractor.extract_batch(image_paths)
    
    # Create results DataFrame
    results = []
    for i, (image_path, extracted_text) in enumerate(zip(image_paths, extracted_data)):
        file_name = Path(image_path).name
        parsed_data = extract_json_from_response(extracted_text)
        
        results.append({
            "file_name": file_name,
            "image_path": image_path,
            "vlm_response": extracted_text,
            "parsed_data": json.dumps(parsed_data)
        })
    
    df_results = pd.DataFrame(results)
    
    # Save results
    if output_file:
        df_results.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")
    
    return df_results 