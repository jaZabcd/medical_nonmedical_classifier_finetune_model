from typing import Union, List, Tuple
from PIL import Image
import sys
from urllib.parse import urlparse
from src.extrat_image_from_url import extract_images_from_website_async
from src.extract_image_from_pdf import extract_images_from_pdf_with_metrics
from src.load_model import load_model
from src.predict import classify_image
from logger import logging
from exception.exception_handing import CustomExceptionHandling
import asyncio
import os

class ImageClassifier:
    def __init__(self, model_path: str):
        """
        Initialize and load the model once.
        """
        try:
            logging.logging.info("🔧 Initializing ImageClassifier")
            self.model_path = model_path
            self.model = self._load_model_once()
            logging.logging.info("✅ Model initialized and ready for inference.")
        except Exception as e:
            logging.logging.error("❌ Failed to initialize ImageClassifier")
            raise CustomExceptionHandling(e, sys)
        
    def _load_model_once(self):
        """
        Load and cache the model during initialization.
        """
        try:
            return load_model(self.model_path)
        except Exception as e:
            logging.logging.error("❌ Error loading model inside _load_model_once")
            raise CustomExceptionHandling(e, sys)
        
    
        
    def _get_images(self, input_data: Union[str, bytes]) -> List[Image.Image]:
        try:
            # Handle URL input
            if isinstance(input_data, str) and input_data.strip().lower().startswith(("http://", "https://")):
                logging.logging.info("📥 Detected input type: URL")
                return asyncio.run(extract_images_from_website_async(input_data))
            
            # Handle PDF input (both bytes and file paths)
            elif isinstance(input_data, (bytes, str)):
                logging.logging.info("📥 Detected PDF input")
                return extract_images_from_pdf_with_metrics(input_data)
            
            # Handle file-like objects
            elif hasattr(input_data, "read"):
                logging.logging.info("📥 Detected file-like object")
                return extract_images_from_pdf_with_metrics(input_data.read())
            
            else:
                raise ValueError(f"Unsupported input type: {type(input_data)}")
                
        except Exception as e:
            logging.logging.exception("❌ Failed to extract images")
            raise CustomExceptionHandling(e, sys)
        
    
        
    def classify(
    self, input_data: Union[str, bytes], input_type: str = "url"
) -> Tuple[List[Tuple[Image.Image, str, float]], float, float, float, float]:
        """
        Classify all images from the given input (URL or PDF).

        Returns:
            results (List[Tuple[Image.Image, str, float]]): Image, label, confidence.
            total_inference_time (float): Total time for all images (seconds).
            avg_inference_time (float): Average time per image (seconds).
            throughput (float): Images per second.
            model_size_mb (float): Model size in MB.
        """
        try:
            logging.logging.info(f"🔎 Starting classification for input type: {input_type}")

            images = self._get_images(input_data)

            if not images:
                logging.logging.warning("⚠️ No images were found to classify.")
                return [], 0.0, 0.0

            results = []
            import time

            

            start = time.time()

            for i, img in enumerate(images):
                label, confidence = classify_image(img, self.model)
                logging.logging.info(
                    f"📸 Image {i + 1} classified as {label} ({confidence:.2f}%)"
                )
                results.append((img, label, confidence))

            end = time.time()
            total_inference_time = end - start
            avg_inference_time = total_inference_time / len(images)
            

            return results, total_inference_time, avg_inference_time

        except Exception as e:
            logging.logging.error("❌ Error during classification")
            raise CustomExceptionHandling(e, sys)