import asyncio
from PIL import Image
import streamlit as st
from typing import List, Tuple
from src.classifier import ImageClassifier
from src.extract_image_from_pdf import extract_images_from_pdf_with_metrics
from src.extrat_image_from_url import extract_images_from_website_async
from logger.logging import logging
from exception.exception_handing import CustomExceptionHandling
import sys

MODEL_PATH = "training_output/efficientnetb0_finetuned.pth"
classifier = ImageClassifier(MODEL_PATH)

if "history" not in st.session_state:
    st.session_state["history"] = []

def classify_input(input_data: str) -> Tuple[str, List[Image.Image]]:
    try:
        if input_data.lower().endswith(".pdf"):
            logging.info(f"Detected PDF: {input_data}")
            images = extract_images_from_pdf_with_metrics(input_data)
            return "pdf", images
        elif input_data.startswith("http"):
            logging.info(f"Detected URL: {input_data}")
            # FIX: Correct async handling
            images = asyncio.run(extract_images_from_website_async(input_data))
            return "url", images
        else:
            raise ValueError("Input must be a PDF URL or web page URL")
    except Exception as e:
        logging.error("Error in classify_input")
        raise CustomExceptionHandling(e, sys)

def show_image_predictions(predictions: List[Tuple[Image.Image, str, float]]) -> None:
    st.subheader("📸 Image Predictions")
    for idx, (img, label, confidence) in enumerate(predictions):
        col1, col2 = st.columns([2, 3])
        with col1:
            st.image(img, use_column_width=True)
        with col2:
            st.markdown(f"**Prediction:** {label}")
            st.markdown(f"**Confidence:** {confidence:.2f}%")
        st.session_state["history"].append((label, confidence))

# ✅ STREAMLIT WORKAROUND FOR ASYNC HANDLING
def run_async_classification(input_data):
    return asyncio.run(async_classify_and_predict(input_data))

async def async_classify_and_predict(input_data):
    source_type, images = await classify_input_async(input_data)
    predictions = [(img, *classifier.classify_image(img)) for img in images]
    return images, predictions

async def classify_input_async(input_data: str):
    try:
        if input_data.lower().endswith(".pdf"):
            images = extract_images_from_pdf_with_metrics(input_data)
            return "pdf", images
        elif input_data.startswith("http"):
            images = await extract_images_from_website_async(input_data)
            return "url", images
        else:
            raise ValueError("Input must be a PDF URL or web page URL")
    except Exception as e:
        raise CustomExceptionHandling(e, sys)

def main():
    st.set_page_config(page_title="Medical Image Classifier", layout="wide")
    st.title("🧠 Medical Image Classifier (URL or PDF)")
    st.markdown("This app classifies images from a URL or PDF as **Medical** or **Non-Medical** using a fine-tuned EfficientNetB0 model.")

    input_data = st.text_input("Enter a webpage URL or a PDF link")

    if st.button("Classify"):
        try:
            if not input_data:
                st.error("Please enter a valid input URL or PDF path")
                return

            st.info("Extracting and classifying images... Please wait.")

            # Use async-compatible flow
            images, predictions = run_async_classification(input_data)

            if not images:
                st.error("No images found.")
                return

            st.success(f"Found {len(images)} images. Showing results:")
            show_image_predictions(predictions)

        except Exception as e:
            st.error("Something went wrong during classification")
            logging.error("Exception in Streamlit main:")
            logging.error(e)

if __name__ == "__main__":
    main()
