from torchvision import transforms
import torch

# Preprocessing
inference_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])



# Image classification
def classify_image(img, model):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    img_tensor = inference_transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img_tensor)
        pred = torch.argmax(output, dim=1).item()
        return "medical" if pred == 0 else "non-medical"