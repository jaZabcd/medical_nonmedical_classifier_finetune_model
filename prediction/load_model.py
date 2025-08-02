import torch
import timm

def load_model(path:str):
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=2)
        model.load_state_dict(torch.load(path, map_location=device))
        model.to(device)
        model.eval()
        print(f"✅ Model loaded successfully on {device.upper()}")
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        raise e


# Load your trained model
# MODEL_PATH = "training_output/efficientnetb0_finetuned.pth"
# moedel = load_model(MODEL_PATH)


