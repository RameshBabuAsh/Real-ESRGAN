import numpy as np
import torch
import torch.nn as nn
import onnx

# ---------------------------
# Residual Block with two convolutional layers and a skip connection.
# ---------------------------
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(channels)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        out = self.relu(out)
        return out

# ---------------------------
# More complex enhancer network using residual blocks.
# ---------------------------
class ComplexEnhancer(nn.Module):
    def __init__(self, num_residual_blocks=4):
        super(ComplexEnhancer, self).__init__()
        # Initial convolution layer: expand from 3 to 64 channels
        self.initial = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        # A series of residual blocks to increase network capacity.
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(64) for _ in range(num_residual_blocks)]
        )
        # Mid-level convolution to further process features.
        self.conv_mid = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64)
        )
        # Final convolution layer: compress back to 3 channels.
        # Using Tanh activation to produce output in [-1, 1] then scaled to [0,1]
        self.final = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Tanh()
        )
    
    def forward(self, x):
        # Get initial features
        initial_features = self.initial(x)
        # Pass through residual blocks
        out = self.residual_blocks(initial_features)
        out = self.conv_mid(out)
        # Global skip connection adds the initial features
        out += initial_features
        out = self.final(out)
        # Scale output from [-1,1] to [0,1]
        out = (out + 1) / 2
        return out

# Load your trained PyTorch model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ComplexEnhancer().to(device)
model.load_state_dict(torch.load("./complex_enhancer.pth", map_location=device, weights_only=True))
model.eval()

# Dummy input for model tracing (Batch size=1, 3 channels, 320x180 image)
dummy_input = torch.randn(1, 3, 180, 320).to(device)

# Export the model to ONNX
onnx_path = "complex_enhancer.onnx"
torch.onnx.export(model, dummy_input, onnx_path, opset_version=11, input_names=["input"], output_names=["output"])
print(f"ONNX model saved to {onnx_path}")

import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType

onnx_path = "complex_enhancer.onnx"
quantized_onnx_path = "complex_enhancer_quantized.onnx"

# Apply dynamic quantization
quantize_dynamic(onnx_path, quantized_onnx_path, weight_type=QuantType.QInt8)

print(f"Quantized model saved to {quantized_onnx_path}")

