import streamlit as st
import torch
import torch.nn as nn

from torchvision.models import (
    resnet18,
    ResNet18_Weights
)

from PIL import Image


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
# PREPROCESSING
# ============================================================

weights = ResNet18_Weights.DEFAULT
transform = weights.transforms()


# ============================================================
# LOAD BANANA DETECTION MODEL
# ============================================================

@st.cache_resource
def load_banana_detector():

    detector_weights = ResNet18_Weights.DEFAULT

    detector = resnet18(
        weights=detector_weights
    )

    detector = detector.to(device)
    detector.eval()

    categories = detector_weights.meta["categories"]

    return detector, categories


# ============================================================
# LOAD SHELF-LIFE REGRESSION MODEL
# ============================================================

@st.cache_resource
def load_shelf_life_model():

    model = resnet18(weights=None)

    num_features = model.fc.in_features

    model.fc = nn.Linear(
        num_features,
        1
    )

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


banana_detector, categories = load_banana_detector()
shelf_life_model = load_shelf_life_model()


# ============================================================
# CHECK WHETHER IMAGE IS A BANANA
# ============================================================

def is_banana(image):

    image = image.convert("RGB")

    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0).to(device)

    with torch.no_grad():

        output = banana_detector(image_tensor)

        probabilities = torch.softmax(
            output,
            dim=1
        )

        # Check top 5 predictions
        top_probabilities, top_indices = torch.topk(
            probabilities,
            5
        )

    top_probabilities = (
        top_probabilities[0]
        .cpu()
        .tolist()
    )

    top_indices = (
        top_indices[0]
        .cpu()
        .tolist()
    )

    predictions = []

    for probability, index in zip(
        top_probabilities,
        top_indices
    ):

        class_name = categories[index]

        predictions.append(
            (
                class_name,
                probability
            )
        )

    # Check if ImageNet recognizes banana
    for class_name, probability in predictions:

        if class_name.lower() == "banana":

            # Minimum confidence
            if probability >= 0.10:
                return True, probability, predictions

    return False, 0.0, predictions


# ============================================================
# SHELF-LIFE PREDICTION
# ============================================================

def predict_banana_life(image):

    image = image.convert("RGB")

    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0).to(device)

    with torch.no_grad():

        prediction = (
            shelf_life_model(
                image_tensor
            )
            .item()
        )

    prediction = max(
        0.0,
        prediction
    )

    return prediction


# ============================================================
# APPLICATION
# ============================================================

st.title(
    "🍌 Banana Shelf-Life Predictor"
)

st.write(
    "Upload a banana image to estimate "
    "how many edible days it may have remaining."
)

st.divider()


uploaded_file = st.file_uploader(
    "Upload a banana image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)


# ============================================================
# IMAGE UPLOADED
# ============================================================

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )


    # ========================================================
    # PREDICT BUTTON
    # ========================================================

    if st.button(
        "Predict Remaining Life",
        type="primary"
    ):

        with st.spinner(
            "Checking image..."
        ):

            banana_found, confidence, detector_predictions = (
                is_banana(image)
            )


        # ====================================================
        # NOT A BANANA
        # ====================================================

        if not banana_found:

            st.error(
                "❌ No banana detected."
            )

            st.warning(
                "Please upload a clear image "
                "containing a banana."
            )

            st.write(
                "Shelf-life prediction was not performed."
            )


        # ====================================================
        # BANANA DETECTED
        # ====================================================

        else:

            st.success(
                "🍌 Banana detected!"
            )

            st.caption(
                f"Banana detection confidence: "
                f"{confidence * 100:.1f}%"
            )

            with st.spinner(
                "Estimating remaining shelf life..."
            ):

                predicted_days = (
                    predict_banana_life(
                        image
                    )
                )


            # ================================================
            # RESULT
            # ================================================

            st.subheader(
                "Prediction"
            )

            st.metric(
                label=(
                    "Estimated Edible "
                    "Life Remaining"
                ),
                value=(
                    f"{predicted_days:.1f} days"
                )
            )


            if predicted_days < 0.5:

                st.warning(
                    "The banana appears to be "
                    "at or near the inedible stage."
                )


            elif predicted_days < 1.5:

                st.info(
                    "The model estimates about "
                    "1 day remaining."
                )


            else:

                st.success(
                    f"The model estimates about "
                    f"{round(predicted_days)} "
                    f"days remaining."
                )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Banana validation uses a pretrained "
    "ImageNet ResNet18 model. Shelf-life estimation "
    "uses a transfer-learned ResNet18 regression model."
)