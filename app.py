import os
import time
import cv2
import torch
import numpy as np
import streamlit as st
from PIL import Image

from model import (
    BaselineSODModel,
    BatchNormSODModel,
    DropoutSODModel,
    UNetSODModel
)


IMAGE_SIZE = 224


MODEL_OPTIONS = {

    "Baseline Model": {
        "class": BaselineSODModel,
        "path": "saved_models/best_baseline_model.pth"
    },

    "Improvement 1 - Added BatchNorm": {
        "class": BatchNormSODModel,
        "path": "saved_models/batchnorm_model.pth"
    },

    "Improvement 2 - Added Dropout": {
        "class": DropoutSODModel,
        "path": "saved_models/dropout_model.pth"
    },

    "Improvement 3 - Added Augmentation + Lower LR": {
        "class": DropoutSODModel,
        "path": "saved_models/agumentations_model.pth"
    },

    "Final Improved Model - U-Net": {
        "class": UNetSODModel,
        "path": "saved_models/unet_model.pth"
    }
}


def preprocess_image(image, image_size):
    image = np.array(image.convert("RGB"))

    resized = cv2.resize(
        image,
        (image_size, image_size)
    )

    normalized = resized.astype(np.float32) / 255.0

    tensor = torch.from_numpy(normalized)
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)

    return resized, tensor


def create_overlay(image, mask):
    image = image.astype(np.float32) / 255.0

    red_mask = np.zeros_like(image)

    red_mask[:, :, 0] = mask

    overlay = image * 0.7 + red_mask * 0.3

    overlay = np.clip(overlay, 0, 1)

    return overlay


@st.cache_resource
def load_model(model_name):

    model_info = MODEL_OPTIONS[model_name]

    model_class = model_info["class"]
    model_path = model_info["path"]

    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")
        return None, None

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = model_class().to(device)

    model.load_state_dict(
        torch.load(model_path, map_location=device)
    )

    model.eval()

    return model, device


st.title("Salient Object Detection Demo")

st.write(
    "Upload an image and choose a trained model "
    "to generate a saliency mask."
)


selected_model = st.selectbox(
    "Choose Model",
    list(MODEL_OPTIONS.keys())
)


uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    image = Image.open(uploaded_file)

    resized, tensor = preprocess_image(
        image,
        IMAGE_SIZE
    )

    model, device = load_model(selected_model)

    if model is not None:

        tensor = tensor.to(device)

        start_time = time.time()

        with torch.no_grad():
            pred = model(tensor)

        inference_time = time.time() - start_time

        pred = pred.squeeze().cpu().numpy()

        pred = np.clip(pred, 0, 1)

        pred_binary = (pred > 0.5).astype(np.float32)

        overlay = create_overlay(
            resized,
            pred_binary
        )

        st.subheader("Results")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.image(
                resized,
                caption="Input Image",
                use_container_width=True
            )

        with col2:
            st.image(
                pred_binary,
                caption="Predicted Saliency Mask",
                use_container_width=True
            )

        with col3:
            st.image(
                overlay,
                caption="Overlay",
                use_container_width=True
            )

        st.success(
            f"Inference Time: {inference_time:.4f} seconds"
        )