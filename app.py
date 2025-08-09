import time
from PIL import Image
import streamlit as st
from src.classifier import ImageClassifier
from exception.exception_handing import CustomExceptionHandling
import sys
import os
import tempfile
import shutil

MODEL_PATH = "training_output/efficientnetb0_finetuned.pth"

# Initialize classifier
try:
    classifier = ImageClassifier(MODEL_PATH)
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Failed to load model: {str(e)}")
    st.stop()

if "history" not in st.session_state:
    st.session_state["history"] = []

def show_image_predictions(predictions) -> None:
    st.subheader("📸 Image Predictions")
    for img, label, confidence in predictions:
        col1, col2 = st.columns([2, 3])
        with col1:
            st.image(img, use_container_width=True)
        with col2:
            st.markdown(f"**Prediction:** {label}")
            st.markdown(f"**Confidence:** {confidence:.2f}%")
        
        # Track history by image content
        if not any(h[1] == label and h[2] == confidence for h in st.session_state["history"]):
            st.session_state["history"].append((img, label, confidence))

def show_performance_metrics(total_time: float, num_images: int):
    """Display performance metrics"""
    avg_time = total_time / num_images if num_images > 0 else 0
    throughput = num_images / total_time if total_time > 0 else 0
    model_size = os.path.getsize(MODEL_PATH) / (1024 * 1024)  # MB

    st.subheader("📈 Performance Metrics")
    cols = st.columns(4)
    metrics = [
        ("⚡ Total Time", f"{total_time:.2f} sec"),
        ("⏱ Avg Time/Image", f"{avg_time * 1000:.2f} ms"),
        ("📊 Throughput", f"{throughput:.2f} img/sec"),
        ("💾 Model Size", f"{model_size:.2f} MB")
    ]
    
    for col, (label, value) in zip(cols, metrics):
        col.metric(label=label, value=value)

def main():
    st.set_page_config(page_title="Medical Image Classifier", layout="wide")
    st.title("🧠 Medical Image Classifier")
    
    # Tab interface
    url_tab, pdf_tab, history_tab = st.tabs(["URL Classifier", "PDF Classifier", "History"])
    
    # URL Classifier Tab
    with url_tab:
        st.subheader("🌐 Classify from URL")
        url_input = st.text_input("Enter a webpage URL")
        if st.button("Classify URL", key="classify_url"):
            if not url_input.strip():
                st.error("Please enter a valid URL")
            else:
                try:
                    with st.spinner("Extracting and classifying images..."):
                        start_time = time.perf_counter()
                        results, total_time, avg_time = classifier.classify(url_input)
                        process_time = time.perf_counter() - start_time
                    
                    if not results:
                        st.warning("No images found at this URL")
                    else:
                        st.success(f"Found {len(results)} images in {process_time:.2f} seconds")
                        show_performance_metrics(total_time, len(results))
                        show_image_predictions(results)
                except Exception as e:
                    error_details = CustomExceptionHandling(e, sys)
                    st.error(f"URL processing failed: {error_details}")
    
    # PDF Classifier Tab
    with pdf_tab:
        st.subheader("📂 Upload PDF File")
        pdf_file = st.file_uploader("Select a PDF file", type=["pdf"])
        if st.button("Classify PDF", key="classify_pdf"):
            if not pdf_file:
                st.error("Please upload a PDF file")
            else:
                try:
                    with st.spinner("Processing PDF..."):
                        # Create a temporary directory that won't be automatically deleted
                        temp_dir = tempfile.mkdtemp()
                        temp_path = os.path.join(temp_dir, pdf_file.name)
                        
                        # Write the uploaded file to the temp directory
                        with open(temp_path, "wb") as f:
                            f.write(pdf_file.getbuffer())
                        
                        start_time = time.perf_counter()
                        results, total_time, avg_time = classifier.classify(temp_path)
                        process_time = time.perf_counter() - start_time
                    
                    if not results:
                        st.warning("No images found in PDF")
                    else:
                        st.success(f"Extracted {len(results)} images in {process_time:.2f} seconds")
                        show_performance_metrics(total_time, len(results))
                        show_image_predictions(results)
                    
                    # Clean up the temporary directory
                    shutil.rmtree(temp_dir, ignore_errors=True)
                    
                except Exception as e:
                    error_details = CustomExceptionHandling(e, sys)
                    st.error(f"PDF processing failed: {error_details}")
                    # Clean up if error occurs
                    if 'temp_dir' in locals():
                        shutil.rmtree(temp_dir, ignore_errors=True)
    
    # History Tab
    with history_tab:
        if st.session_state["history"]:
            st.subheader("📚 Classification History")
            for idx, (img, label, confidence) in enumerate(st.session_state["history"]):
                st.image(img, caption=f"{label} ({confidence:.2f}%)", width=200)
        else:
            st.info("No classification history yet")
    
    # Clear button
    if st.button("Clear Results and History"):
        st.session_state["history"] = []
        st.rerun()

if __name__ == "__main__":
    main()