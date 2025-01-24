import os 
import torch
from PIL import Image, ImageEnhance
import numpy as np
from RealESRGAN import RealESRGAN
from torchvision import transforms
from torch.utils.data import DataLoader
from dataset import LOLDataset

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# Dataset paths
dataset_dir = 'inputs'

# Training dataset and loader
# train_dataset = LOLDataset(dataset_dir=dataset_dir, mode='train', transform=transform)
# train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)

# Validation dataset and loader
val_dataset = LOLDataset(dataset_dir=dataset_dir, mode='val')
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)



def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RealESRGAN(device=device, scale=2)
    # model.load_for_training('/kaggle/working/Real-ESRGAN/weights/RealESRGAN_x2.pth')

    # model.train(dataloader=train_loader, num_epochs=2, learning_rate=1e-4)
    # model.save_model('/kaggle/working/Real-ESRGAN/weights/fine_tuned_realesrgan.pth')

    model.load_for_eval('weights\RealESRGAN_x2.pth')
    print('Model loaded successfully!')
    # lr_image_path = r'C:\Users\Ramesh Babu\Real-ESRGAN\inputs\eval15\low\1.png'
    # lr_image = Image.open(lr_image_path).convert('RGB')

    # # Apply transformations to the LR image
    # # Adjust brightness
    # brightness_enhancer = ImageEnhance.Brightness(lr_image)
    # lr_image = brightness_enhancer.enhance(4)  # Increase brightness

    # # Adjust contrast
    # contrast_enhancer = ImageEnhance.Contrast(lr_image)
    # lr_image = contrast_enhancer.enhance(0.8)  # Decrease contrast

    # # Adjust exposure by scaling pixel intensity
    # lr_image_array = np.array(lr_image)
    # exposure_factor = 1.5  # Increase exposure by 50%
    # lr_image_array = np.clip(lr_image_array * exposure_factor, 0, 255).astype(np.uint8)
    # model.predict_org(lr_image_array)

    model.validate(dataloader=val_loader)



if __name__ == '__main__':
    main()