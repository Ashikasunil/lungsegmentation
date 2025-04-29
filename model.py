
import torch.nn as nn
import torch.nn.functional as F
import timm
import torch

# --- QFC ---
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

# --- ResCaps ---
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

# --- ADSC ---
class ADSCBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ADSCBlock, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return F.relu(x)

# --- QRC-U-Net Full Architecture ---
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
