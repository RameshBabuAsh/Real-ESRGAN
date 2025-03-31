import cv2
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

# Residual Block with two convolutional layers and a skip connection.
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

# More complex enhancer network using residual blocks.
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


# Assuming preprocess_frame() and PairedVideoDataset are defined as in the previous example.
# If not, define preprocess_frame() accordingly.

# --- Define preprocess_frame() ---
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

# --- Define PairedVideoDataset ---
class PairedVideoDataset(Dataset):
    def __init__(self, test_video_path, org_video_path, resize_dims=(320, 240)):
        self.test_frames = []
        self.org_frames = []
        cap_test = cv2.VideoCapture(test_video_path)
        cap_org  = cv2.VideoCapture(org_video_path)
        if not cap_test.isOpened() or not cap_org.isOpened():
            raise ValueError("Error: Could not open one of the videos.")
        while True:
            ret_test, frame_test = cap_test.read()
            ret_org, frame_org = cap_org.read()
            if not ret_test or not ret_org:
                break
            frame_test = cv2.resize(frame_test, resize_dims)
            frame_org  = cv2.resize(frame_org, resize_dims)
            self.test_frames.append(frame_test)
            self.org_frames.append(frame_org)
        cap_test.release()
        cap_org.release()
        if len(self.test_frames) != len(self.org_frames):
            raise ValueError("Mismatch in frame counts between the videos.")
    
    def __len__(self):
        return len(self.test_frames)
    
    def __getitem__(self, idx):
        test_frame = self.test_frames[idx]
        org_frame = self.org_frames[idx]
        # Preprocess the test frame
        preprocessed_frame = preprocess_frame(test_frame)
        # Convert from BGR to RGB
        preprocessed_frame = cv2.cvtColor(preprocessed_frame, cv2.COLOR_BGR2RGB)
        org_frame = cv2.cvtColor(org_frame, cv2.COLOR_BGR2RGB)
        # Normalize and rearrange channels
        preprocessed_frame = preprocessed_frame.astype(np.float32) / 255.0
        org_frame = org_frame.astype(np.float32) / 255.0
        preprocessed_frame = np.transpose(preprocessed_frame, (2, 0, 1))
        org_frame = np.transpose(org_frame, (2, 0, 1))
        # Convert to torch tensors
        return torch.tensor(preprocessed_frame), torch.tensor(org_frame)

# --- Training Loop ---
def train_model(model, dataloader, criterion, optimizer, device, num_epochs=100):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {avg_loss:.4f}")
    print("Training complete.")

if __name__ == '__main__':
    # Update video paths as needed.
    test_video_path = "test.mp4"
    org_video_path  = "org.mp4"
    
    dataset = PairedVideoDataset(test_video_path, org_video_path, resize_dims=(320, 240))
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ComplexEnhancer(num_residual_blocks=4).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    print("Starting training...")
    
    # Train the model. Increase epochs as needed.
    train_model(model, dataloader, criterion, optimizer, device, num_epochs=50)
    
    # Save the trained model
    torch.save(model.state_dict(), "complex_enhancer.pth")
