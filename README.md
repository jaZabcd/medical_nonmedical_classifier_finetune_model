# 🧠 Medical Image Classifier

## 📌 Overview
The **Medical Image Classifier** is a deep learning–based application that can automatically classify images as **medical** or **non-medical**.  
It supports two input types:
- Extracting and classifying **all images from a webpage URL**
- Extracting and classifying **all images from a PDF file**

This tool is useful for hospitals, research labs, and academic projects where large collections of images need to be sorted quickly and accurately.

---

## 🚀 Features
- **High Accuracy** – Fine-tuned EfficientNet-B0 model with near-perfect classification results.
- **Multi-Input Support** – Classify images from both URLs and PDF files.
- **Performance Metrics** – Displays total inference time, average inference time per image, throughput, and model size.
- **History Tracking** – Stores recent classification results in session history for easy reference.
- **User-Friendly Interface** – Built with Streamlit for easy interaction.

---

## 🏗 Model Architecture
The model is based on **EfficientNet-B0** and has been **fine-tuned** for the binary classification task.  
Key steps in the fine-tuning process:
1. **Base Model Selection** – EfficientNet-B0 pre-trained on ImageNet for strong general image features.
2. **Custom Classification Layer** – Replaced the original classification head with:
   - Global Average Pooling
   - Fully Connected Layer
   - Dropout for regularization
   - Output layer with sigmoid activation for binary classification
3. **Data Augmentation** – Applied random rotations, flips, and normalization to improve generalization.
4. **Training Strategy** – Used Adam optimizer with a learning rate scheduler for stable convergence.
5. **Evaluation** – Tested on ~450 images for both accuracy and speed.

---

## 📊 Evaluation Results
### Classification Report
| Class         | Precision | Recall | F1-score | Support |
|---------------|-----------|--------|----------|---------|
| Medical       | 1.00      | 1.00   | 1.00     | 267     |
| Non-medical   | 1.00      | 0.99   | 1.00     | 198     |
| **Accuracy**  |           |        | **1.00** | 465     |

### Performance Metrics
#### URL Input
- Found: **36 images** in **7.96 sec**
- Avg Time/Image: **221.15 ms**
- Throughput: **4.52 img/sec**
- Model Size: **15.58 MB**

#### PDF Input
- Total Time: **0.26 sec**
- Avg Time/Image: **132.28 ms**
- Throughput: **7.56 img/sec**
- Model Size: **15.58 MB**

---

