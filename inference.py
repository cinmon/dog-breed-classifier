import torch
from torchvision import transforms, models
from PIL import Image
import os
import json

with open("class_names.json", "r") as f:
    class_names = json.load(f)


def load_model():

    model = models.resetnet18(pretrained=False) # false when running inference
    model.fc = torch.nn.Sequential(
        torch.nn.Dropout(0.5),
        torch.nn.Linear(model.fc.in_features, len(class_names))
    )

    model.load_state_dict(torch.load("model_best_weights.pt"), map_location=torch.device('cpu'))
    model.eval()

    return model

def predict_image(image_path, model):

    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
        predicted_class = class_names[predicted.item()]
        confidence = torch.nn.functional.softmax(output, dim=1)[0][predicted.item()].item()

    return predicted_class, confidence