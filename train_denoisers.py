import pandas as pd
import subprocess
import shutil
import os
from tqdm import tqdm
# from method import AdaCLIP_Trainer
import torch
import torch.nn as nn
import torch.optim as optim

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
import random
from torch.utils.data import DataLoader
from UNet.unet import UNet

class DiffusionDenoisingDataset(Dataset):
    def __init__(self, csv_file, img_size=518, variants_per_image=3):
        df_1 = pd.read_csv(csv_file)
        self.df = df_1.iloc[: 300, :]
        self.image_paths = self.df['image_path'].tolist()
        self.class_labels = self.df['class'].tolist()
        self.variants_per_image = variants_per_image

        # Define noise levels (std deviations) for each variant
        self.noise_levels = [0, 6, 16]

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_paths) * self.variants_per_image

    def __getitem__(self, idx):
        real_idx = idx // self.variants_per_image
        variant_idx = idx % self.variants_per_image

        img_path = self.image_paths[real_idx]
        class_label = self.class_labels[real_idx]

        clean = self.transform(Image.open(img_path).convert('RGB'))

        # Get noise standard deviation for this variant
        noise_std = self.noise_levels[variant_idx]
        if noise_std == 0:
            noisy = clean.clone()
        else:
            noise = torch.randn_like(clean) * (noise_std / 255.0)  # normalize noise scale
            noisy = clean + noise
            noisy = torch.clamp(noisy, 0.0, 1.0)

        return noisy, clean

csv_path = '/data/rrjha/AP/AdaCLIP_Denoiser/preprocess_training_datasets/mvtec_image_paths.csv'

dataset = DiffusionDenoisingDataset(csv_path)
dataloader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=True,
    num_workers=4,
    pin_memory=True      
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(3).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# === Training ===
import matplotlib.pyplot as plt

num_epochs = 50
epoch_batch_losses = []  # List of lists: one inner list per epoch
for epoch in range(num_epochs):
    cnt = 0
    model.train()
    batch_losses = []  # Store batch loss for this epoch only

    with tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch') as pbar:
        for noisy, clean in pbar:
            noisy = noisy.to(device)
            clean = clean.to(device)

            output = model(noisy)
            loss = criterion(output, clean) * 100.0

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_value = loss.item()
            batch_losses.append(loss_value)
            pbar.set_postfix(batch_loss=loss_value)

    epoch_batch_losses.append(batch_losses)  # Save this epoch's losses
    print(f"Epoch [{epoch+1}/{num_epochs}] - Avg Loss: {sum(batch_losses) / len(batch_losses):.4f}")

    # Save model checkpoint
    weights_dir = './denoisers/Trained_Weights'
    os.makedirs(weights_dir, exist_ok=True)
    model_save_path = os.path.join(weights_dir, f'unet_epoch_{epoch+1}.pth')
    torch.save(model.state_dict(), model_save_path)
    loss = os.path.join(weights_dir, 'Unet_Loss.pth')
    torch.save(epoch_batch_losses, loss)

avg_losses = [sum(batch_losses) / len(batch_losses) for batch_losses in epoch_batch_losses]

plt.figure(figsize=(8, 5))
plt.plot(range(1, num_epochs + 1), avg_losses, marker='o', linestyle='-', color='r')
plt.title('Average Training Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.tight_layout()
plt.savefig('./denoisers/Trained_Weights/loss_vs_epoch.png')  
plt.show()
