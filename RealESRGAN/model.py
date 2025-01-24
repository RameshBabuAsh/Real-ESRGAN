import os
import torch
from torch.nn import functional as F
from torch.optim import Adam
from PIL import Image
import numpy as np
from huggingface_hub import hf_hub_url, cached_download
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as calculate_psnr
from skimage.metrics import structural_similarity as calculate_ssim
from .rrdbnet_arch import RRDBNet

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

    def train(self, dataloader, num_epochs, learning_rate=1e-4, loss_fn=None, batch_size=4, patches_size=192, padding=24, pad_size=15):
        """
        Train the RealESRGAN model using a patch-based approach.

        Args:
            dataloader (DataLoader): PyTorch DataLoader providing (LR, HR) image pairs.
            num_epochs (int): Number of epochs to train the model.
            loss_fn (callable): Loss function to calculate training loss.
            optimizer (torch.optim.Optimizer): Optimizer for model training.
            batch_size (int): Batch size for processing patches.
            patches_size (int): Size of the patches to split the image into for processing.
            padding (int): Padding size for overlapping patches.
            pad_size (int): Padding size for reflective padding of the input images.
            save_path (str): Directory to save intermediate training results (optional).
        """
        if loss_fn is None:
            loss_fn = torch.nn.L1Loss()  # Default to L1 loss

        optimizer = Adam(self.model.parameters(), lr=learning_rate)
        scale = self.scale
        device = self.device

        self.model.train()  # Set the model to training mode

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            for batch_idx, (lr_images_batch, hr_images_batch) in enumerate(tqdm(dataloader)):
                batch_loss = 0.0

                for idx, (lr_image, hr_image) in enumerate(zip(lr_images_batch, hr_images_batch)):
                    # Convert images to NumPy arrays
                    lr_image = np.array(lr_image)
                    hr_image = np.array(hr_image)
        
                    img = torch.FloatTensor(lr_image/255).permute((2,0,1)).unsqueeze(0).to(device).detach()

                    with torch.no_grad():
                        res = self.model(img)

                    res = res.squeeze(0)

                    sr_image = res.permute((1,2,0)).clamp_(0, 1).cpu()

                    np_sr_image = sr_image.numpy()

                    sr_img = (np_sr_image*255).astype(np.uint8)

                    # Convert LR, HR, and SR images to tensors for loss calculation
                    sr_tensor = torch.FloatTensor(sr_img / 255).permute(2, 0, 1).unsqueeze(0).to(device)
                    hr_tensor = torch.FloatTensor(hr_image / 255).permute(2, 0, 1).unsqueeze(0).to(device)

                    # Calculate loss
                    loss = loss_fn(sr_tensor, hr_tensor)
                    batch_loss += loss.item()

                    # Backward pass and optimization
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                # Average loss for the batch
                epoch_loss += batch_loss / len(lr_images_batch)

            # Log average epoch loss
            avg_epoch_loss = epoch_loss / len(dataloader)
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_epoch_loss:.4f}")

        print("Training Complete!")


    def validate(self, dataloader, batch_size=4, patches_size=192, padding=24, pad_size=15, save_path="validation_results"):
        """
        Validate the RealESRGAN model by calculating PSNR and SSIM on a dataset.

        Args:
            dataloader (DataLoader): PyTorch DataLoader providing (LR, HR) image pairs.
            batch_size (int): Batch size for processing patches.
            patches_size (int): Size of the patches to split the image into for processing.
            padding (int): Padding size for overlapping patches.
            pad_size (int): Padding size for reflective padding of the input images.
            save_path (str): Directory to save super-resolved images.

        Returns:
            dict: Dictionary containing average PSNR and SSIM for the validation dataset.
        """
        os.makedirs(save_path, exist_ok=True)
        scale = self.scale
        device = self.device

        total_psnr = 0.0
        # total_ssim = 0.0
        num_images = 0

        self.model.eval()
        with torch.no_grad():
            for batch_idx, (lr_images_batch, hr_images_batch) in enumerate(tqdm(dataloader)):
                for idx, (lr_image, hr_image) in enumerate(zip(lr_images_batch, hr_images_batch)):
                    # Convert images to NumPy arrays
                    lr_image = np.array(lr_image)
                    hr_image = np.array(hr_image)
        
                    img = torch.FloatTensor(lr_image/255).permute((2,0,1)).unsqueeze(0).to(device).detach()

                    with torch.no_grad():
                        res = self.model(img)

                    res = res.squeeze(0)

                    sr_image = res.permute((1,2,0)).clamp_(0, 1).cpu()

                    np_sr_image = sr_image.numpy()

                    sr_img = (np_sr_image*255).astype(np.uint8)
                    hr_image = hr_image.astype(np.uint8)

                    # Calculate PSNR and SSIM
                    psnr = calculate_psnr(hr_image, sr_img, data_range=255)
                    # ssim = calculate_ssim(hr_image, sr_img, multichannel=True, data_range=255)

                    total_psnr += psnr
                    # total_ssim += ssim
                    num_images += 1

                    # Save the super-resolved image
                    sr_pil_image = Image.fromarray(sr_img)
                    sr_pil_image.save(os.path.join(save_path, f"sr_image_{batch_idx}_{idx}.png"))

        # Calculate average metrics
        avg_psnr = total_psnr / num_images
        # avg_ssim = total_ssim / num_images

        print(f"Validation Results - PSNR: {avg_psnr:.4f}")
        return {"PSNR": avg_psnr}



    def predict_org_dataloader(self, dataloader, batch_size=4, patches_size=192, padding=24, pad_size=15):
        """
        Perform super-resolution on a dataset using a DataLoader.

        Args:
            dataloader (DataLoader): PyTorch DataLoader providing low-resolution images.
            batch_size (int): Batch size for model inference.
            patches_size (int): Size of the patches to split the image into for processing.
            padding (int): Padding size for overlapping patches.
            pad_size (int): Padding size for reflective padding of the input images.

        Returns:
            List[PIL.Image]: List of high-resolution images corresponding to the input images.
        """
        scale = self.scale
        device = self.device
        super_resolved_images = []

        self.model.eval()
        with torch.no_grad():
            for lr_images_batch in tqdm(dataloader):
                for lr_image in lr_images_batch:
                    lr_image = np.array(lr_image)
        
                    img = torch.FloatTensor(lr_image/255).permute((2,0,1)).unsqueeze(0).to(device).detach()

                    with torch.no_grad():
                        res = self.model(img)

                    res = res.squeeze(0)

                    sr_image = res.permute((1,2,0)).clamp_(0, 1).cpu()

                    np_sr_image = sr_image.numpy()

                    sr_img = (np_sr_image*255).astype(np.uint8)
                    sr_img = Image.fromarray(sr_img)
                    super_resolved_images.append(sr_img)

        return super_resolved_images


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

    def predict_org(self, lr_image, batch_size=4, patches_size=192,
                padding=24, pad_size=15):
        scale = self.scale
        device = self.device
        lr_image = np.array(lr_image)
        
        img = torch.FloatTensor(lr_image/255).permute((2,0,1)).unsqueeze(0).to(device).detach()

        with torch.no_grad():
            res = self.model(img)

        res = res.squeeze(0)

        sr_image = res.permute((1,2,0)).clamp_(0, 1).cpu()

        np_sr_image = sr_image.numpy()

        sr_img = (np_sr_image*255).astype(np.uint8)

        sr_img = Image.fromarray(sr_img)

        return sr_img