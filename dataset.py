from torch.utils.data import Dataset
from PIL import Image, ImageEnhance
import os
import numpy as np

class LOLDataset(Dataset):
    def __init__(self, dataset_dir, mode='train', transform=None):
        """
        Args:
            dataset_dir (str): Path to the LOL dataset directory.
            mode (str): 'train' for our485 or 'val' for eval15.
            transform (callable, optional): Optional transforms to be applied on a sample.
        """
        self.dataset_dir = dataset_dir
        self.mode = mode
        self.transform = transform

        # Choose the appropriate subfolder
        self.subfolder = 'our485' if mode == 'train' else 'eval15'
        self.lr_dir = os.path.join(dataset_dir, self.subfolder, 'low')
        self.hr_dir = os.path.join(dataset_dir, self.subfolder, 'high')

        # Sort and pair the images
        self.lr_images = sorted(os.listdir(self.lr_dir))
        self.hr_images = sorted(os.listdir(self.hr_dir))

    def __len__(self):
        return len(self.lr_images)
    
    def __getitem__(self, idx):
        # Load low-resolution (LR) and high-resolution (HR) images
        lr_image_path = os.path.join(self.lr_dir, self.lr_images[idx])
        hr_image_path = os.path.join(self.hr_dir, self.hr_images[idx])

        lr_image = Image.open(lr_image_path).convert('RGB')
        hr_image = Image.open(hr_image_path).convert('RGB')

        # Resize HR image to (2*w, 2*h)
        hr_width, hr_height = hr_image.size
        hr_image = hr_image.resize((2 * hr_width, 2 * hr_height))

        # Apply transformations to the LR image
        # Adjust brightness
        brightness_enhancer = ImageEnhance.Brightness(lr_image)
        lr_image = brightness_enhancer.enhance(4)  # Increase brightness

        # Adjust contrast
        contrast_enhancer = ImageEnhance.Contrast(lr_image)
        lr_image = contrast_enhancer.enhance(0.8)  # Decrease contrast

        # Adjust exposure by scaling pixel intensity
        lr_image = np.array(lr_image)
        exposure_factor = 1.5  # Increase exposure by 50%
        lr_image = np.clip(lr_image * exposure_factor, 0, 255).astype(np.uint8)

        # Convert images to NumPy arrays
        hr_image = np.array(hr_image)

        return lr_image, hr_image
