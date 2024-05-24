# train.py
import torch
from denoising_diffusion_pytorch import GaussianDiffusion, Trainer
from denoising_diffusion_pytorch.meshnet import MeshNet

config_file_path = 'meshnet_config.json'  # Ensure this points to your configuration file
in_channels = 3
n_classes = 100
channels = [64, 128, 256, 512]  # Example channel sizes, adjust as needed

model = MeshNet(in_channels, n_classes, channels, config_file_path)

diffusion = GaussianDiffusion(
    model,
    image_size = 128,
    timesteps = 1000    # number of steps
)

training_images = torch.rand(8, 3, 128, 128) # images are normalized from 0 to 1
loss = diffusion(training_images)
loss.backward()

# after a lot of training

sampled_images = diffusion.sample(batch_size = 4)
print(sampled_images.shape) # (4, 3, 128, 128)