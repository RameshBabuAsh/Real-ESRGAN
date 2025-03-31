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
model = ComplexEnhancer().to(device)
# Note: Remove or modify the argument `weights_only=True` if your torch version does not support it.
model.load_state_dict(torch.load("complex_enhancer2.pth", map_location=device, weights_only=True))
model.eval()

# ---------------------------
# Open the Videos
# ---------------------------
cap_test = cv2.VideoCapture("test.mp4")
cap_org = cv2.VideoCapture("org.mp4")
if not cap_test.isOpened() or not cap_org.isOpened():
    print("Error: Could not open one of the videos.")
    exit()

# ---------------------------
# Process Frames and Store the Output
# ---------------------------
output_frames = []
start_time = time.time()
while True:
    ret_test, frame_test = cap_test.read()
    ret_org, frame_org = cap_org.read()
    if not ret_test or not ret_org:
        break
    
    # Resize frames to ensure consistent dimensions
    frame_test = cv2.resize(frame_test, (320, 240))
    frame_org  = cv2.resize(frame_org, (320, 240))
    
    # Apply preprocessing to the test frame
    preprocessed = preprocess_frame(frame_test)
    
    # Convert preprocessed frame to RGB, normalize and convert to tensor
    preprocessed_rgb = cv2.cvtColor(preprocessed, cv2.COLOR_BGR2RGB)
    preprocessed_norm = preprocessed_rgb.astype(np.float32) / 255.0
    input_tensor = torch.from_numpy(np.transpose(preprocessed_norm, (2, 0, 1))).unsqueeze(0).to(device)
    
    # Run inference with the model
    with torch.no_grad():
        output_tensor = model(input_tensor)
    
    # Process model output: remove batch dimension and convert to HWC format
    output_tensor = output_tensor.squeeze(0).cpu().numpy()
    output_tensor = np.transpose(output_tensor, (1, 2, 0))
    output_tensor = np.clip(output_tensor, 0, 1)
    output_img = (output_tensor * 255).astype(np.uint8)
    
    # For display, convert original test frame from BGR to RGB
    org_display = cv2.cvtColor(frame_test, cv2.COLOR_BGR2RGB)
    
    # Concatenate model output and original side by side (output on the left)
    combined = np.hstack((output_img, org_display))
    
    # Convert combined image back to BGR for OpenCV display
    combined_bgr = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
    
    output_frames.append(combined_bgr)
end_time = time.time()
total_time = end_time - start_time
print("Total processing time for the model:", total_time, "seconds")

# Release video resources after processing
cap_test.release()
cap_org.release()

# ---------------------------
# Display the Stored Video
# ---------------------------
# Display each stored frame at ~30 FPS
for frame in output_frames:
    cv2.imshow("Model Output (Left) vs Original (Right)", frame)
    # Wait key delay is calculated based on 30 FPS (i.e. ~33ms per frame)
    if cv2.waitKey(int(1000 / 30)) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
