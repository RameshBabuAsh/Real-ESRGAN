import os
import torch
from torch.nn import functional as F
from torch.optim import Adam
from PIL import Image
import numpy as np
from huggingface_hub import hf_hub_url, cached_download

from .rrdbnet_arch import RRDBNet
from .utils import pad_reflect, split_image_into_overlapping_patches, stich_together, unpad_image


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

    def load_weights(self, model_path, download=True):
        if not os.path.exists(model_path) and download:
            assert self.scale in [2, 4, 8], 'You can download models only with scales: 2, 4, 8'
            config = HF_MODELS[self.scale]
            cache_dir = os.path.dirname(model_path)
            local_filename = os.path.basename(model_path)
            config_file_url = hf_hub_url(repo_id=config['repo_id'], filename=config['filename'])
            cached_download(config_file_url, cache_dir=cache_dir, force_filename=local_filename)
            print('Weights downloaded to:', os.path.join(cache_dir, local_filename))

        loadnet = torch.load(model_path)
        if 'params' in loadnet:
            self.model.load_state_dict(loadnet['params'], strict=True)
        elif 'params_ema' in loadnet:
            self.model.load_state_dict(loadnet['params_ema'], strict=True)
        else:
            self.model.load_state_dict(loadnet, strict=True)
        self.model.to(self.device)

    def predict(self, lr_image, batch_size=4, patches_size=192,
                padding=24, pad_size=15):
        scale = self.scale
        device = self.device
        lr_image = np.array(lr_image)
        lr_image = pad_reflect(lr_image, pad_size)

        patches, p_shape = split_image_into_overlapping_patches(
            lr_image, patch_size=patches_size, padding_size=padding
        )
        img = torch.FloatTensor(patches / 255).permute((0, 3, 1, 2)).to(device).detach()

        with torch.no_grad():
            res = self.model(img[0:batch_size])
            for i in range(batch_size, img.shape[0], batch_size):
                res = torch.cat((res, self.model(img[i:i + batch_size])), 0)

        sr_image = res.permute((0, 2, 3, 1)).clamp_(0, 1).cpu()
        np_sr_image = sr_image.numpy()

        padded_size_scaled = tuple(np.multiply(p_shape[0:2], scale)) + (3,)
        scaled_image_shape = tuple(np.multiply(lr_image.shape[0:2], scale)) + (3,)
        np_sr_image = stich_together(
            np_sr_image, padded_image_shape=padded_size_scaled,
            target_shape=scaled_image_shape, padding_size=padding * scale
        )
        sr_img = (np_sr_image * 255).astype(np.uint8)
        sr_img = unpad_image(sr_img, pad_size * scale)
        sr_img = Image.fromarray(sr_img)

        return sr_img

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
            for lr_images, hr_images in dataloader:
                lr_images = lr_images.to(self.device)
                hr_images = hr_images.to(self.device)

                # Forward pass
                sr_images = self.model(lr_images)
                print(sr_images.shape, hr_images.shape)

                # Calculate loss
                loss = loss_fn(sr_images, hr_images)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(dataloader)
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

    def save_model(self, save_path):
        """Save the trained model."""
        torch.save(self.model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

    def load_for_training(self, model_path):
        """Load pre-trained weights for training."""
        loadnet = torch.load(model_path)
        self.model.load_state_dict(loadnet, strict=False)
        self.model.train()
