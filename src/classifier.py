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
        """
        Automatically detects input type (URL or PDF) and extracts images.
        Args:
        input_data: URL string, PDF bytes, or file-like object
        Returns:
        List of PIL.Image.Image objects
        """
        try:
            # Detect input type
            if isinstance(input_data, str) and input_data.strip().lower().startswith(("http://", "https://")):
                logging.logging.info("📥 Detected input type: URL")
                return asyncio.run(extract_images_from_website_async(input_data))
            elif isinstance(input_data, bytes):
                logging.info("📥 Detected input type: PDF (bytes)")
                return extract_images_from_pdf_with_metrics(input_data)

            elif hasattr(input_data, "read"):
                logging.info("📥 Detected input type: PDF (file-like)")
                return extract_images_from_pdf_with_metrics(input_data.read())

            else:
                raise ValueError("Unable to detect input type. Must be a URL string, PDF bytes, or file-like object.")

        except Exception as e:
            logging.error(f"❌ Failed to extract images in _get_images: {e}")
            raise CustomExceptionHandling(e, sys)
        
    def classify(self, input_data: Union[str, bytes], input_type: str = "url") -> List[Tuple[str, str]]:
        """
        Classify all images from the given input (URL or PDF).

        Returns:
            List of tuples with image label and prediction.
        """
        try:
            logging.logging.info(f"🔎 Starting classification for input type: {input_type}")
            images = self._get_images(input_data)

            if not images:
                logging.logging.warning("⚠️ No images were found to classify.")
                return []

            results = []
            for i, img in enumerate(images):
                label = classify_image(img, self.model)
                logging.logging.info(f"📸 Image {i + 1} classified as {label}")
                results.append((f"Image {i + 1}", label))

            logging.logging.info("✅ Classification completed")
            return results
        except Exception as e:
            logging.logging.error("❌ Error during classification")
            raise CustomExceptionHandling(e, sys)