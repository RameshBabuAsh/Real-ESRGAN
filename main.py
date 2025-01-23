import os 
import torch
from PIL import Image
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
dataset_dir = '/kaggle/input/lol-dataset/lol_dataset'

# Training dataset and loader
train_dataset = LOLDataset(dataset_dir=dataset_dir, mode='train', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)

# Validation dataset and loader
val_dataset = LOLDataset(dataset_dir=dataset_dir, mode='val', transform=transform)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)



def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = RealESRGAN(device, scale=2)
    # model.load_weights('weights/RealESRGAN_x2.pth', download=False)
    # for i, image in enumerate(os.listdir("inputs")):
    #     image = Image.open(f"inputs/{image}").convert('RGB')
    #     sr_image = model.predict(image)
    #     sr_image.save(f'results/{i}.png')
    model = RealESRGAN(device=device, scale=2)
    model.load_for_training('weights\RealESRGAN_x2.pth')

    model.train(dataloader=train_loader, num_epochs=1, learning_rate=1e-4)
    model.save_model(r'weights\\fine_tuned_realesrgan.pth')



if __name__ == '__main__':
    main()