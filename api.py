"""
FastAPI application for invoice extraction API
"""

import base64
import io
from typing import Dict, Any
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image

from config import API_TITLE, API_DESCRIPTION, API_HOST, API_PORT
from model import InvoiceExtractor

# Initialize FastAPI app
app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version="1.0.0"
)

# Global extractor instance
extractor = None


@app.on_event("startup")
async def startup_event():
    """Initialize the model when the API starts."""
    global extractor
    print("Loading invoice extraction model...")
    extractor = InvoiceExtractor()
    print("API ready!")


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "message": "Invoice Extraction API is running",
        "version": "1.0.0",
        "endpoints": {
            "extract_invoice": "POST /extract_invoice - Upload image file",
            "extract_invoice_base64": "POST /extract_invoice_base64 - Send base64 image",
            "health": "GET /health - Health check"
        }
    }


@app.get("/health")
async def health_check():
    """Detailed health check."""
    return {
        "status": "healthy",
        "model_loaded": extractor is not None,
        "device": extractor.device if extractor else None
    }


# @app.post("/extract_invoice")
# async def extract_invoice(file: UploadFile = File(...)):
#     """
#     Extract structured data from an uploaded invoice image.
    
#     Args:
#         file: Uploaded image file (JPEG, PNG, etc.)
    
#     Returns:
#         JSON with extracted invoice data
#     """
#     if extractor is None:
#         raise HTTPException(status_code=503, detail="Model not loaded")
    
#     # Validate file type
#     if not file.content_type.startswith('image/'):
#         raise HTTPException(
#             status_code=400, 
#             detail=f"File must be an image. Received: {file.content_type}"
#         )
    
#     try:
#         # Read file content
#         image_bytes = await file.read()
        
#         # Validate image can be opened
#         try:
#             Image.open(io.BytesIO(image_bytes))
#         except Exception as e:
#             raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")
        
#         # Extract data
#         result = extractor.extract_single_image(image_bytes)
        
#         return JSONResponse(content=result)
    
#     except HTTPException:
#         raise
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
@app.post("/extract_invoice")
async def extract_invoice(file: UploadFile = File(...)):
    """
    Extract structured data from an uploaded invoice image.
    """
    if extractor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400, 
            detail=f"File must be an image. Received: {file.content_type}"
        )

    try:
        image_bytes = await file.read()

        # Validate image
        try:
            Image.open(io.BytesIO(image_bytes))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")

        # Extract result
        result = extractor.extract_single_image(image_bytes)

        # 🔍 Log result to terminal
        print("Extracted raw result:", result)

        # ✅ Ensure proper JSON response
        if isinstance(result, dict):
            print("RESULT TYPE:", type(result))
            print("RESULT VALUE:", result)

            return JSONResponse(content=result)
        else:
            return JSONResponse(content={"extracted_text": result})

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@app.post("/extract_invoice_base64")
async def extract_invoice_base64(image_data: Dict[str, str]):
    """
    Extract structured data from a base64 encoded image.
    
    Args:
        image_data: {"image": "base64_encoded_image_string"}
    
    Returns:
        JSON with extracted invoice data
    """
    if extractor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Decode base64 image
        image_b64 = image_data.get("image")
        if not image_b64:
            raise HTTPException(status_code=400, detail="No image data provided")
        
        try:
            image_bytes = base64.b64decode(image_b64)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid base64 encoding: {str(e)}")
        
        # Validate image can be opened
        try:
            Image.open(io.BytesIO(image_bytes))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")
        
        # Extract data
        result = extractor.extract_single_image(image_bytes)
        
        return JSONResponse(content=result)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@app.post("/extract_invoice_path")
async def extract_invoice_path(image_path: Dict[str, str]):
    """
    Extract structured data from an image file path.
    
    Args:
        image_path: {"path": "/path/to/image.jpg"}
    
    Returns:
        JSON with extracted invoice data
    """
    if extractor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        path = image_path.get("path")
        if not path:
            raise HTTPException(status_code=400, detail="No image path provided")
        
        # Validate file exists
        import os
        if not os.path.exists(path):
            raise HTTPException(status_code=404, detail=f"Image file not found: {path}")
        
        # Extract data
        result = extractor.extract_single_image(path)
        
        return JSONResponse(content=result)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host=API_HOST, port=API_PORT) 
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
