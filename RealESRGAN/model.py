import os
import torch
from torch.nn import functional as F
from torch.optim import Adam
from PIL import Image
import numpy as np
from huggingface_hub import hf_hub_url, cached_download
from tqdm import tqdm
from .rrdbnet_arch import RRDBNet
from .utils import pad_reflect, split_image_into_overlapping_patches, stich_together, unpad_image
from skimage.metrics import peak_signal_noise_ratio as calculate_psnr
from skimage.metrics import structural_similarity as calculate_ssim

HF_MODELS = {
    2: dict(
        repo_id='sberbank-ai/Real-ESRGAN',
        filename='RealESRGAN_x2.pth',
    ),
    4: dict(
        repo_id='sberbank-ai/Real-ESRGAN',
        filename='RealESRGAN_x4.pth',
    ),
    8: dict(
        repo_id='sberbank-ai/Real-ESRGAN',
        filename='RealESRGAN_x8.pth',
    ),
}


class RealESRGAN:
    def __init__(self, device, scale=4):
        self.device = device
        self.scale = scale
        self.model = RRDBNet(
            num_in_ch=3, num_out_ch=3, num_feat=64,
            num_block=23, num_grow_ch=32, scale=scale
        )
        self.model.to(self.device)

    def train(self, dataloader, num_epochs, learning_rate=1e-4, loss_fn=None):
        """
        Train the RealESRGAN model.

        Args:
            dataloader (DataLoader): PyTorch DataLoader providing training data (LR, HR pairs).
            num_epochs (int): Number of epochs to train.
            learning_rate (float): Learning rate for the optimizer.
            loss_fn (callable): Loss function (e.g., L1, L2, or perceptual loss).
        """
        if loss_fn is None:
            loss_fn = torch.nn.L1Loss()  # Default to L1 loss

        optimizer = Adam(self.model.parameters(), lr=learning_rate)

        self.model.train()
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            for lr_images, hr_images in tqdm(dataloader):
                lr_images = lr_images.to(self.device)
                hr_images = hr_images.to(self.device)

                # Forward pass
                sr_images = self.model(lr_images)

                # Calculate loss
                loss = loss_fn(sr_images, hr_images)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(dataloader)
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

    def validate(self, dataloader, win_size=None):
        """
        Validate the RealESRGAN model by calculating PSNR and SSIM on a dataset.

        Args:
            dataloader (DataLoader): PyTorch DataLoader providing low-resolution and high-resolution image pairs.
            win_size (int, optional): Window size for SSIM calculation. Default is None, which uses skimage's default.

        Returns:
            dict: Dictionary containing average PSNR and SSIM for the validation dataset.
        """
        self.model.eval()  # Set the model to evaluation mode
        total_psnr = 0.0
        total_ssim = 0.0
        num_images = 0

        with torch.no_grad():
            for lr_images, hr_images in tqdm(dataloader):
                # Move images to the device
                lr_images = lr_images.to(self.device)
                hr_images = hr_images.to(self.device)

                # Perform super-resolution
                sr_images = self.model(lr_images)

                # Calculate metrics for each image in the batch
                for sr_image, hr_image in zip(sr_images, hr_images):
                    # Convert SR and HR images to NumPy arrays
                    sr_image = sr_image.permute(1, 2, 0).clamp(0, 1).cpu().numpy()
                    hr_image = hr_image.permute(1, 2, 0).clamp(0, 1).cpu().numpy()

                    # Rescale images to [0, 255] for metric calculations
                    sr_image = (sr_image * 255).astype(np.uint8)
                    hr_image = (hr_image * 255).astype(np.uint8)

                    # Determine appropriate win_size for SSIM if not provided
                    effective_win_size = win_size or min(sr_image.shape[0], sr_image.shape[1], 7)

                    # Calculate PSNR and SSIM
                    psnr = calculate_psnr(hr_image, sr_image, data_range=255)
                    ssim = calculate_ssim(hr_image, sr_image, win_size=effective_win_size, multichannel=True, data_range=255)

                    total_psnr += psnr
                    total_ssim += ssim
                    num_images += 1

        # Calculate average metrics
        avg_psnr = total_psnr / num_images
        avg_ssim = total_ssim / num_images

        print(f"Validation Results - PSNR: {avg_psnr:.6f}, SSIM: {avg_ssim:.8f}")
        return {"PSNR": avg_psnr, "SSIM": avg_ssim}

    def predict(self, dataloader):
        """
        Perform super-resolution on a dataset using a DataLoader.

        Args:
            dataloader (DataLoader): PyTorch DataLoader providing low-resolution images.

        Returns:
            List[PIL.Image]: List of high-resolution images corresponding to the input images.
        """
        self.model.eval()  # Set the model to evaluation mode
        high_res_images = []

        with torch.no_grad():
            for lr_images in tqdm(dataloader):
                # Move low-resolution images to the device
                lr_images = lr_images.to(self.device)

                # Perform super-resolution
                sr_images = self.model(lr_images)

                # Process each super-resolved image
                for sr_image in sr_images:
                    # Convert the tensor back to a NumPy array and scale to [0, 255]
                    sr_image = sr_image.permute(1, 2, 0).clamp(0, 1).cpu().numpy()
                    sr_image = (sr_image * 255).astype(np.uint8)

                    # Convert to PIL Image and store
                    high_res_images.append(Image.fromarray(sr_image))

        return high_res_images

    def save_model(self, save_path):
        """Save the trained model."""
        torch.save(self.model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

    def load_for_training(self, model_path):
        """Load pre-trained weights for training."""
        loadnet = torch.load(model_path, weights_only=True, map_location=self.device)
        self.model.load_state_dict(loadnet, strict=False)
        self.model.train()

    def load_for_eval(self, model_path):
        """Load pre-trained weights for evaluation."""
        loadnet = torch.load(model_path, weights_only=True, map_location=self.device)
        self.model.load_state_dict(loadnet, strict=False)
        self.model.eval()
