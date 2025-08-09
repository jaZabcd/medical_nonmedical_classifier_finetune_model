import torch
import timm
import sys
from logger import logging
from exception.exception_handing import CustomExceptionHandling

def load_model(path: str):
    """
    Loads a fine-tuned EfficientNet-B0 model for binary classification.

    Args:
        path (str): Path to the .pth or .pt model weights file.

    Returns:
        torch.nn.Module: The loaded and evaluated model on the correct device.
    """
    try:
        logging.logging.info("🔄 Starting model loading process")

        # Detect device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.logging.info(f"📟 Using device: {device}")

        # Initialize model architecture
        model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=2)

        # Load weights
        model.load_state_dict(torch.load(path, map_location=device))
        model.to(device)
        model.eval()

        logging.logging.info(f"✅ Model loaded successfully on {device}")
        print(f"✅ Model loaded successfully on {device}")
        return model

    except FileNotFoundError as fnf:
        logging.logging.error("❌ Model file not found at the specified path.")
        raise CustomExceptionHandling(fnf, sys)
    
    except RuntimeError as re:
        logging.logging.error("❌ Runtime error while loading model. Possibly a mismatch between model structure and weights.")
        raise CustomExceptionHandling(re, sys)
    
    except Exception as e:
        logging.logging.error("❌ Unexpected error during model loading.")
        raise CustomExceptionHandling(e, sys)
