from typing import Union, List, Tuple
from PIL import Image

from load_images import extract_images_from_pdf, extract_images_from_website
from load_model import load_model
from predict import classify_image

class ImageClassifier:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = self._load_model_once()

    def _load_model_once(self):
        """
        Load and cache the model once during initialization.
        """
        return load_model(self.model_path)

    def _get_images(self, input_data: Union[str, bytes], input_type: str) -> List[Image.Image]:
        """
        Extract images from a URL or PDF input.
        """
        if input_type == "url":
            return extract_images_from_website(input_data)
        elif input_type == "pdf":
            if hasattr(input_data, "read"):
                file_bytes = input_data.read()
            else:
                file_bytes = input_data
            return extract_images_from_pdf(file_bytes)
        else:
            raise ValueError("input_type must be 'url' or 'pdf'")

    def classify(self, input_data: Union[str, bytes], input_type: str = "url") -> List[Tuple[str, str]]:
        """
        Classify all images from the given input (URL or PDF).

        Returns:
            List of tuples with image label and prediction.
        """
        print(f"🧪 Processing input type: {input_type}")
        images = self._get_images(input_data, input_type)

        if not images:
            print("⚠️ No images found.")
            return []

        results = []
        for i, img in enumerate(images):
            label = classify_image(img, self.model)
            results.append((f"Image {i + 1}", label))
        return results

# ✅ Example usage:
if __name__ == "__main__":
    model_path = "training_output\efficientnetb0_finetuned.pth"  # Replace this with your actual model path
    classifier = ImageClassifier(model_path)

    url = "https://radiopaedia.org/cases/air-bronchogram-in-pneumonia?lang=us"
    url_results = classifier.classify(url, input_type="url")

    print("Results from URL:")
    for name, label in url_results:
        print(name, ":", label)
