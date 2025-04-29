
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torchvision import transforms
from PIL import Image
import numpy as np

# --- Custom CSS for professional styling ---
st.set_page_config(page_title="Lung Nodule Segmentation", layout="wide")
st.markdown("""
    <style>
        body {
            background: linear-gradient(to right, #e3f2fd, #f8f9fa);
        }
        .big-title {
            text-align: center;
            padding: 10px;
            border-radius: 12px;
            background-color: #1565c0;
            color: white;
            font-size: 30px;
            font-weight: bold;
        }
        .footer {
            font-size: 13px;
            text-align: center;
            color: #888;
            padding-top: 30px;
        }
        .card {
            background: white;
            padding: 15px;
            border-radius: 12px;
            box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
            margin-bottom: 20px;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='big-title'>🫁 Lung Nodule Segmentation (QRC-U-Net)</div>", unsafe_allow_html=True)

# --- Model components (QFC, ResCaps, ADSC, QRC_UNet) ---
class QuantumFourierConv(nn.Module):
    def __init__(self, channels):
        super(QuantumFourierConv, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        real = self.conv1(x)
        imag = self.conv2(x)
        out = torch.sqrt(real**2 + imag**2)
        return out

class ResidualCapsuleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualCapsuleBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        skip = self.skip(x)
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        return F.relu(x + skip)

class ADSCBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ADSCBlock, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return F.relu(x)

class QRC_UNet(nn.Module):
    def __init__(self):
        super(QRC_UNet, self).__init__()
        self.encoder = timm.create_model('mobilevit_xxs', pretrained=True, features_only=True)
        enc_channels = self.encoder.feature_info.channels()

        self.qfc = QuantumFourierConv(enc_channels[-1])
        self.rescaps = ResidualCapsuleBlock(enc_channels[-1], 256)

        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.adsc1 = ADSCBlock(128 + enc_channels[3], 128)

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.adsc2 = ADSCBlock(64 + enc_channels[2], 64)

        self.up3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.adsc3 = ADSCBlock(32 + enc_channels[1], 32)

        self.up4 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.adsc4 = ADSCBlock(16 + enc_channels[0], 16)

        self.final_conv = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x):
        e1, e2, e3, e4, e5 = self.encoder(x)
        x = self.qfc(e5)
        x = self.rescaps(x)

        x = self.up1(x)
        x = self.adsc1(torch.cat([x, e4], dim=1))

        x = self.up2(x)
        x = self.adsc2(torch.cat([x, e3], dim=1))

        x = self.up3(x)
        x = self.adsc3(torch.cat([x, e2], dim=1))

        x = self.up4(x)
        x = self.adsc4(torch.cat([x, e1], dim=1))

        x = self.final_conv(x)
        return torch.sigmoid(x)

# --- Image Transforms ---
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# --- Load model ---
@st.cache_resource
def load_model():
    model = QRC_UNet()
    model.load_state_dict(torch.load("qrc_unet_trained.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

# --- Upload and process image ---
uploaded_file = st.file_uploader("📤 Upload a chest CT image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    image = Image.open(uploaded_file).convert("RGB")
    image_resized = transform(image).unsqueeze(0)

    with st.spinner("🧠 Segmenting lung nodules..."):
        with torch.no_grad():
            output = model(image_resized)
            output = F.interpolate(output, size=(image.size[1], image.size[0]), mode='bilinear', align_corners=False)
            mask = (output.squeeze().cpu().numpy() > 0.5).astype(np.uint8)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(image, caption="📷 Original Image", use_column_width=True)

    with col2:
        st.image(mask * 255, caption="🩻 Predicted Mask", use_column_width=True, clamp=True)

    with col3:
        overlay = np.array(image).copy()
        overlay[mask == 1] = [255, 0, 0]  # Red highlight
        st.image(overlay, caption="📊 Overlay", use_column_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("<div class='footer'>Built with ❤️ using QRC-U-Net • Streamlit • PyTorch</div>", unsafe_allow_html=True)
