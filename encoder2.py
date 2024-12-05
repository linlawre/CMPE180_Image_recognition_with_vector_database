import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image


# Simple implementation of a diffusion model
class DiffusionModel(nn.Module):
    def __init__(self, image_size, channels):
        super(DiffusionModel, self).__init__()
        self.image_size = image_size
        self.channels = channels

        # U-Net model
        self.encoder = nn.Conv2d(channels, 64, kernel_size=3, stride=1, padding=1)
        self.middle = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.decoder = nn.Conv2d(64, channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.relu(self.encoder(x))
        x = F.relu(self.middle(x))
        x = self.decoder(x)
        return x


# Simple noise addition function
def add_noise(image, t, sigma):
    """Add noise to the image."""
    noise = torch.randn_like(image) * sigma * np.sqrt(t)
    return image + noise


# Function to reverse the noise (this is what the model learns)
def reverse_noise(model, image, t, sigma):
    """Use the model to reverse the noise."""
    return model(image)


# Training loop for diffusion model
def train_diffusion_model(model, data_loader, epochs, device):
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    model.to(device)
    model.train()

    for epoch in range(epochs):
        for images, _ in data_loader:
            images = images.to(device)

            # Add noise at random time step
            t = np.random.randint(1, 1000)  # Time step for noise schedule
            sigma = 1.0  # Noise strength parameter
            noisy_images = add_noise(images, t, sigma)

            # Forward pass to denoise
            denoised_images = model(noisy_images)

            # Loss function - Mean Squared Error
            loss = F.mse_loss(denoised_images, images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item()}")


# Example usage with random images
if __name__ == '__main__':
    # Simulate loading dataset (use actual dataset in practice)
    transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
    dataset = [transform(Image.new('RGB', (64, 64))) for _ in range(1000)]
    data_loader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Initialize the model
    model = DiffusionModel(image_size=64, channels=3)

    # Training the diffusion model
    train_diffusion_model(model, data_loader, epochs=10, device='cuda' if torch.cuda.is_available() else 'cpu')
