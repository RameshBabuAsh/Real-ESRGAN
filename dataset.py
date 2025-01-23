from torch.utils.data import Dataset
from PIL import Image
import os

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

        # Apply transformations if specified
        if self.transform:
            lr_image = self.transform(lr_image)
            hr_image = self.transform(hr_image)

        return lr_image, hr_image
