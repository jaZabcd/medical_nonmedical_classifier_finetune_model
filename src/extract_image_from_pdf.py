import fitz  # PyMuPDF
from PIL import Image
import time
import tracemalloc
import os
import json
from io import BytesIO
from logger import logging
from exception.exception_handing import CustomExceptionHandling
import sys

def extract_images_from_pdf_with_metrics(pdf_data, report_path: str = "output/benchmark_report.json") -> list[Image.Image]:
    """
    Extracts images from PDF data (either file path or bytes), logs timing and memory usage.
    """
    tracemalloc.start()
    t0 = time.perf_counter()
    images = []
    doc = None
    
    try:
        # Handle both file paths and binary data
        if isinstance(pdf_data, bytes):
            logging.logging.info(f"📄 Processing PDF from binary data ({len(pdf_data)} bytes)")
            doc = fitz.open(stream=pdf_data, filetype="pdf")
        elif isinstance(pdf_data, str):
            logging.logging.info(f"📄 Processing PDF from file: {pdf_data}")
            doc = fitz.open(pdf_data)
        else:
            raise ValueError("Unsupported PDF input type. Must be path string or bytes.")
        
        for page_number in range(len(doc)):
            page = doc.load_page(page_number)
            image_list = page.get_images(full=True)
            
            for img_index, img_info in enumerate(image_list):
                xref = img_info[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                
                try:
                    image = Image.open(BytesIO(image_bytes)).convert("RGB")
                    images.append(image)
                    logging.logging.info(f"✅ Extracted image from page {page_number+1}, index {img_index}")
                except Exception as e:
                    logging.logging.warning(f"⚠️ Failed to decode image on page {page_number+1}, index {img_index}: {e}")
        
        t1 = time.perf_counter()
        current, peak = tracemalloc.get_traced_memory()
        
        # Create metrics
        metrics = {
            "source": "binary" if isinstance(pdf_data, bytes) else pdf_data,
            "images_found": len(images),
            "time_sec": round(t1 - t0, 3),
            "mem_current_kb": round(current / 1024, 2),
            "mem_peak_kb": round(peak / 1024, 2),
        }

        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(metrics, f, indent=2)

        logging.logging.info(f"✅ Extracted {len(images)} images")
        logging.logging.info(f"⏱️ Time: {metrics['time_sec']} sec | 💾 Mem: {metrics['mem_current_kb']} KB (current), {metrics['mem_peak_kb']} KB (peak)")
        return images

    except Exception as e:
        logging.logging.exception(f"❌ PDF processing failed")
        raise
    finally:
        if doc:
            doc.close()
        tracemalloc.stop()