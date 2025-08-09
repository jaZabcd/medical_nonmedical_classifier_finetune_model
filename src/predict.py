from torchvision import transforms
import torch
from logger import logging
from exception.exception_handing import CustomExceptionHandling
import sys

# Preprocessing
inference_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])



def classify_image(img, model):
    """Classify image as 'medical' or 'non-medical' using the provided model."""
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logging.logging.info(f"🖥️ Using device: {device}")
        model.to(device)  
        model.eval()  

        img_tensor = inference_transform(img).unsqueeze(0).to(device)  

        with torch.no_grad():  
            output = model(img_tensor)  
            pred = torch.argmax(output, dim=1).item()  

        label = "medical" if pred == 0 else "non-medical"  
        logging.logging.info(f"🔍 Prediction: {label} (class index: {pred})")  
        return label  

    except Exception as e:  
        logging.logging.error(f"❌ Error during image classification: {e}")  
        raise CustomExceptionHandling(e, sys)
