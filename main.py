from src.classifier import ImageClassifier  
from logger import logging
from exception.exception_handing import CustomExceptionHandling
import sys

def main(path):
    try:
        model_path = "training_output/efficientnetb0_finetuned.pth"
        classifier = ImageClassifier(model_path)

        # Example: classify images from a URL
        
        logging.logging.info(f"Classifying images from URL: {path}")
        results, total_time, avg_time = classifier.classify(path)

        for img, name, label in results:
            print(f"{name} => {label}")
            logging.logging.info(f"{name} => {label}")

    except Exception as e:
        logging.logging.error("Unhandled error in main")
        raise CustomExceptionHandling(e, sys)

# if __name__ == "__main__":
#     path = "https://en.wikipedia.org/wiki/Cancer"
#     main(path)
