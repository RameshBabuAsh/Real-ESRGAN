import os
import time 
import cv2
import torch
from PIL import Image, ImageEnhance
import numpy as np
from RealESRGAN import RealESRGAN
from torchvision import transforms
# from torch.utils.data import DataLoader
# from dataset import LOLDataset

# Define transformations
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
# ])

# Dataset paths
# dataset_dir = 'inputs'

# Training dataset and loader
# train_dataset = LOLDataset(dataset_dir=dataset_dir, mode='train', transform=transform)
# train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)

# Validation dataset and loader
# val_dataset = LOLDataset(dataset_dir=dataset_dir, mode='val')
# val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)

def preprocess_image(input_path):
    # Load image in color (OpenCV)
    image = cv2.imread(input_path)

    # Convert to float32 for processing
    image_float = image.astype(np.float32)

    # Gamma correction (brightness adjustment)
    gamma = 0.7
    gamma_corrected = np.power(image_float / 255.0, gamma) * 255.0
    gamma_corrected = np.clip(gamma_corrected, 0, 255)

    # Exposure fusion/tone mapping
    exposure_factor = 4
    tone_mapped_image = gamma_corrected * exposure_factor
    tone_mapped_image = np.clip(tone_mapped_image, 0, 255)

    # Decrease contrast
    contrast_factor = 0.9
    midpoint = 15
    decreased_contrast_image = (tone_mapped_image - midpoint) * contrast_factor + midpoint
    decreased_contrast_image = np.clip(decreased_contrast_image, 0, 255)

    return image, gamma_corrected, tone_mapped_image, decreased_contrast_image



def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    model = RealESRGAN(device=device, scale=2)
    # model.load_for_training('/kaggle/working/Real-ESRGAN/weights/RealESRGAN_x2.pth')

    # model.train(dataloader=train_loader, num_epochs=2, learning_rate=1e-4)
    # model.save_model('/kaggle/working/Real-ESRGAN/weights/fine_tuned_realesrgan.pth')

    model.load_for_eval('weights/fine_tuned_realesrgan.pth')
    print('Model loaded successfully!')
    start_time = time.time()  # Start timing
    lr_image_path = r'C:/Users/Ramesh Babu/Real-ESRGAN/inputs/eval15/low/4.png'
    lr_image = Image.open(lr_image_path).convert('RGB')

    # Apply transformations to the LR image
    # Adjust brightness
    brightness_enhancer = ImageEnhance.Brightness(lr_image)
    lr_image = brightness_enhancer.enhance(4)  # Increase brightness

    # Adjust contrast
    contrast_enhancer = ImageEnhance.Contrast(lr_image)
    lr_image = contrast_enhancer.enhance(0.8)  # Decrease contrast

    # Adjust exposure by scaling pixel intensity
    lr_image_array = np.array(lr_image)
    exposure_factor = 1.5  # Increase exposure by 50%
    lr_image_array = np.clip(lr_image_array * exposure_factor, 0, 255).astype(np.uint8)

    # _, _, _, decreased_contrast_image = preprocess_image(lr_image_path)
    hr_image = model.predict_org(lr_image_array)
    end_time = time.time()  # Stop timing
    processing_time = end_time - start_time
    print(f"Processing time: {processing_time:.2f} seconds")
    # hr_image = Image.fromarray(hr_image)
    hr_image.show()


    # model.validate(dataloader=val_loader)



if __name__ == '__main__':
    main()