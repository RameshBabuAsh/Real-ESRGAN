import time
import cv2
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseSeparableConv(nn.Module):
    """ Depthwise Separable Convolution for efficiency """
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, padding=padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = DepthwiseSeparableConv(channels, channels)
        self.gn1 = nn.GroupNorm(8, channels)
        self.conv2 = DepthwiseSeparableConv(channels, channels)
        self.gn2 = nn.GroupNorm(8, channels)
        self.act = nn.SiLU()

    def forward(self, x):
        identity = x
        out = self.act(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        return self.act(out + identity)

class ComplexEnhancer(nn.Module):
    def __init__(self, num_residual_blocks=3, channels=32):
        super(ComplexEnhancer, self).__init__()
        self.initial = nn.Sequential(
            DepthwiseSeparableConv(3, channels),
            nn.SiLU()
        )
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(channels) for _ in range(num_residual_blocks)]
        )
        self.conv_mid = DepthwiseSeparableConv(channels, channels)
        self.final = nn.Sequential(
            DepthwiseSeparableConv(channels, 3),
            nn.Tanh()
        )

    def forward(self, x):
        initial_features = self.initial(x)
        out = self.residual_blocks(initial_features)
        out = self.conv_mid(out) + initial_features
        out = self.final(out)
        return (out + 1) / 2  # Scale to [0,1]

# ---------------------------
# Preprocessing Function (same as training)
# ---------------------------
def preprocess_frame(frame):
    frame_float = frame.astype(np.float32)
    gamma = 0.7
    gamma_corrected = np.power(frame_float / 255.0, gamma) * 255.0
    gamma_corrected = np.clip(gamma_corrected, 0, 255)
    exposure_factor = 2
    tone_mapped_image = gamma_corrected * exposure_factor
    tone_mapped_image = np.clip(tone_mapped_image, 0, 255)
    contrast_factor = 0.9
    midpoint = 15
    decreased_contrast_image = (tone_mapped_image - midpoint) * contrast_factor + midpoint
    decreased_contrast_image = np.clip(decreased_contrast_image, 0, 255)
    return decreased_contrast_image.astype(np.uint8)

# ---------------------------
# Load Model
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
model = ComplexEnhancer().to(device)
# Note: Remove or modify the argument `weights_only=True` if your torch version does not support it.
model.load_state_dict(torch.load("complex_enhancer2.pth", map_location=device, weights_only=True))
model.eval()

# ---------------------------
# Open the Videos
# ---------------------------
image_path = "inputs/eval15/low/4.png"  # Change this to your image path
image = cv2.imread(image_path)
if image is None:
    print("Error: Could not load the image.")
    exit()

# Resize image to (320, 240) for consistency
start_time = time.time()
image = cv2.resize(image, (320, 240))

# Apply preprocessing
preprocessed = preprocess_frame(image)

# Convert to RGB, normalize and convert to tensor
preprocessed_rgb = cv2.cvtColor(preprocessed, cv2.COLOR_BGR2RGB)
preprocessed_norm = preprocessed_rgb.astype(np.float32) / 255.0
input_tensor = torch.from_numpy(np.transpose(preprocessed_norm, (2, 0, 1))).unsqueeze(0).to(device)

# Run inference and measure time
with torch.no_grad():
    output_tensor = model(input_tensor)
end_time = time.time()

# Calculate and print processing time
processing_time = (end_time - start_time) * 1000  # Convert to milliseconds
print(f"Processing Time: {processing_time:.2f} ms")

# Convert model output back to image format
output_tensor = output_tensor.squeeze(0).cpu().numpy()
output_tensor = np.transpose(output_tensor, (1, 2, 0))
output_tensor = np.clip(output_tensor, 0, 1)
output_img = (output_tensor * 255).astype(np.uint8)

# Display the processed image
cv2.imshow("Enhanced Image", output_img)
cv2.waitKey(0)
cv2.destroyAllWindows()