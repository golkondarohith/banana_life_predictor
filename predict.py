import torch
import torch.nn as nn
from torchvision.models import resnet18
from PIL import Image
import sys

from image_preprocess import transform


# ============================================================
# DEVICE
# ============================================================

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("Using device:", device)


# ============================================================
# LOAD MODEL
# ============================================================

model = resnet18(weights=None)

num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 1)

model.load_state_dict(
    torch.load(
        "banana_resnet18_best.pth",
        map_location=device,
        weights_only=True
    )
)

model = model.to(device)
model.eval()


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_banana(image_path):

    # Open image
    image = Image.open(image_path).convert("RGB")

    # Same preprocessing used during training
    image_tensor = transform(image)

    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0).to(device)

    # Prediction
    with torch.no_grad():
        prediction = model(image_tensor).item()

    # Remaining life cannot be negative
    prediction = max(0.0, prediction)

    return prediction


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage:")
        print("python3 predict.py path_to_image")
        sys.exit()

    image_path = sys.argv[1]

    predicted_days = predict_banana(image_path)

    print("\n-----------------------------")
    print("BANANA LIFE PREDICTION")
    print("-----------------------------")

    print(f"Estimated days remaining: {predicted_days:.2f} days")

    if predicted_days < 0.5:
        print("Prediction: At or near the inedible stage")
    elif predicted_days < 1.5:
        print("Prediction: About 1 day remaining")
    else:
        print(f"Prediction: About {round(predicted_days)} days remaining")