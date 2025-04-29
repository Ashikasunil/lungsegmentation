
import streamlit as st
from PIL import Image
import torch
import numpy as np
from model import QRC_UNet
from utils import preprocess_image, postprocess_mask

st.set_page_config(page_title="Lung CT Scan Segmentation", layout="centered")
st.title("🧬 Lung Nodule Segmentation (QRC-U-Net)")

@st.cache_resource
def load_model():
    model = QRC_UNet()
    model.load_state_dict(torch.load("qrc_unet_trained.pth", map_location='cpu'))
    model.eval()
    return model

model = load_model()

uploaded_file = st.file_uploader("Upload a Lung CT Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    resized_image = image.resize((180, 180))
    st.image(resized_image, caption="Uploaded Image", use_column_width=False)

    input_tensor = preprocess_image(image)
    with torch.no_grad():
        output = model(input_tensor)
        pred_mask = torch.sigmoid(output)
        binary_mask = postprocess_mask(pred_mask)

    # Resize mask to match image for overlay
    binary_mask_resized = Image.fromarray(binary_mask).resize((256, 256))
    binary_mask_resized = np.array(binary_mask_resized)

    st.subheader("🩻 Predicted Segmentation Mask")
    st.image(binary_mask_resized, width=300, caption="Binary Mask")

    overlay = np.array(image.resize((256, 256))).copy()
    overlay[:, :, 1] = np.maximum(overlay[:, :, 1], binary_mask_resized)
    st.subheader("📊 Overlay (Image + Mask)")
    st.image(overlay, width=300, caption="Overlayed Output")

    confidence = pred_mask.mean().item()
    st.success(f"🧠 Confidence Score: {confidence:.2f}")
