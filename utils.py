
from torchvision import transforms
from PIL import Image
import numpy as np

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    return transform(image).unsqueeze(0)

def postprocess_mask(mask_tensor):
    mask = mask_tensor.squeeze().detach().cpu().numpy()
    return (mask > 0.5).astype(np.uint8) * 255
