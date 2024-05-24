import torch
from denoising_diffusion_pytorch import GaussianDiffusion, Trainer
from denoising_diffusion_pytorch.meshnet import MeshNet
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def denormalize(img, mean, std):
    img = img * std[:, None, None] + mean[:, None, None]
    return img

def show_images(images, nrow=4):
    fig, axes = plt.subplots(1, nrow, figsize=(15, 15))
    for i in range(nrow):
        img = images[i]
        img = denormalize(img, np.array([0.5071, 0.4867, 0.4408]), np.array([0.2675, 0.2565, 0.2761]))
        img = np.clip(img, 0, 1)
        axes[i].imshow(np.transpose(img, (1, 2, 0)))
        axes[i].axis('off')
    plt.savefig('figure/sampled_images.png')

# CIFAR-100 dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
])

train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

config_file_path = 'meshnet_config.json'  
in_channels = 3
n_classes = 100
channels = [64, 128, 256, 512] 

model = MeshNet(in_channels, n_classes, channels, config_file_path)

diffusion = GaussianDiffusion(
    model,
    image_size=32,  # CIFAR-100 images are 32x32
    timesteps=1000
)

optimizer = torch.optim.Adam(diffusion.parameters(), lr=2e-4)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model, dataloader, optimizer, epochs=100):
    model.train()
    for epoch in range(epochs):
        for i, (x, _) in enumerate(dataloader):
            optimizer.zero_grad()
            loss = diffusion(x)
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print(f"Epoch [{epoch}/{epochs}] Batch [{i}/{len(dataloader)}] Loss: {loss.item()}")

train(diffusion, train_loader, optimizer, epochs=100)

sampled_images = diffusion.sample(batch_size=4)
sampled_images = sampled_images.cpu().numpy()  # Convert to numpy array

# Show the generated images
show_images(sampled_images, nrow=4)
