from PIL import Image
import torch
import numpy as np
from torchvision import transforms
from unet import UNet
import matplotlib.pyplot as plt

image_path = "Img.jpg"
img = Image.open(image_path).convert("RGB")

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

input_tensor = transform(img).unsqueeze(0)

print("Input shape:", input_tensor.shape)
print(f"Anomaly map min value: {input_tensor.min().item()}")
print(f"Anomaly map max value: {input_tensor.max().item()}")

model = UNet(3)
model.eval()
print(model)


with torch.no_grad():
    output = model(input_tensor)

print("Output shape:", output.shape)
print(f"Anomaly map min value: {output.min().item()}")
print(f"Anomaly map max value: {output.max().item()}")

output_img = output.squeeze(0).permute(1, 2, 0).clamp(0, 1)
output_np = (output_img.numpy() * 255).astype(np.uint8)

# Save as JPG
output_pil = Image.fromarray(output_np)
output_pil.save("output.jpg")
