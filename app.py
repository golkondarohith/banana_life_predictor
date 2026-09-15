import streamlit as st
import torch
import torch.nn as nn
from torchvision.models import resnet18
from PIL import Image

from image_preprocess import transform


# ============================================================
# PAGE SETTINGS
# ============================================================

st.set_page_config(
    page_title="Banana Shelf-Life Predictor",
    page_icon="🍌",
    layout="centered"
)


# ============================================================
# DEVICE
# ============================================================

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

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

    return model


model = load_model()


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_banana(image):

    image = image.convert("RGB")

    image_tensor = transform(image)

    image_tensor = image_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        prediction = model(image_tensor).item()

    # Shelf life cannot be negative
    prediction = max(0.0, prediction)

    return prediction


# ============================================================
# APP
# ============================================================

st.title("🍌 Banana Shelf-Life Predictor")

st.write(
    "Upload a banana image to estimate how many edible days "
    "it may have remaining."
)

st.divider()


uploaded_file = st.file_uploader(
    "Upload a banana image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Banana",
        use_container_width=True
    )

    if st.button("Predict Remaining Life"):

        predicted_days = predict_banana(image)

        st.subheader("Prediction")

        st.metric(
            label="Estimated Edible Life Remaining",
            value=f"{predicted_days:.1f} days"
        )

        if predicted_days < 0.5:

            st.warning(
                "The banana appears to be at or near "
                "the inedible stage."
            )

        elif predicted_days < 1.5:

            st.info(
                "The model estimates about 1 day remaining."
            )

        else:

            st.success(
                f"The model estimates about "
                f"{round(predicted_days)} days remaining."
            )


st.divider()

st.caption(
    "Prediction generated using a ResNet18 "
    "deep-learning regression model."
)