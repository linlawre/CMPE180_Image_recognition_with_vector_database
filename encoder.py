import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# Define the Generator Network
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(100, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 1024)
        self.fc4 = nn.Linear(1024, 28*28)  # Assuming we are generating MNIST images (28x28)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, z):
        x = self.relu(self.fc1(z))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.tanh(self.fc4(x))  # Output in range [-1, 1]
        return x.view(-1, 28, 28)  # Reshape to 28x28 image

# Define the Discriminator Network
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(28*28, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 1)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.view(-1, 28*28)  # Flatten the image
        x = self.leaky_relu(self.fc1(x))
        x = self.leaky_relu(self.fc2(x))
        x = self.leaky_relu(self.fc3(x))
        x = self.fc4(x)
        return self.sigmoid(x)  # Output a probability of real/fake

# Initialize the models
generator = Generator()
discriminator = Discriminator()

# Set the device (GPU or CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
generator.to(device)
discriminator.to(device)

# Loss function and optimizers
criterion = nn.BCELoss()
lr = 0.0002
beta1 = 0.5
optimizer_g = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
optimizer_d = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))

# Prepare dataset (MNIST in this case)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
])

dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

# Training the GAN
num_epochs = 10
for epoch in range(num_epochs):
    for i, (imgs, _) in enumerate(dataloader):
        # Prepare labels
        real_labels = torch.ones(imgs.size(0), 1).to(device)
        fake_labels = torch.zeros(imgs.size(0), 1).to(device)

        # Move images to the GPU
        imgs = imgs.to(device)

        # Train the discriminator: maximize log(D(x)) + log(1 - D(G(z)))
        optimizer_d.zero_grad()

        # Real images
        outputs = discriminator(imgs)
        d_loss_real = criterion(outputs, real_labels)
        d_loss_real.backward()

        # Fake images
        z = torch.randn(imgs.size(0), 100).to(device)  # Random noise
        fake_imgs = generator(z)
        outputs = discriminator(fake_imgs.detach())
        d_loss_fake = criterion(outputs, fake_labels)
        d_loss_fake.backward()

        # Update discriminator
        d_loss = d_loss_real + d_loss_fake
        optimizer_d.step()

        # Train the generator: maximize log(D(G(z)))
        optimizer_g.zero_grad()
        outputs = discriminator(fake_imgs)
        g_loss = criterion(outputs, real_labels)  # Want to fool the discriminator
        g_loss.backward()

        # Update generator
        optimizer_g.step()

        # Print the losses every few steps
        if i % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], '
                  f'D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}')

    # Save generated images every epoch
    with torch.no_grad():
        z = torch.randn(64, 100).to(device)
        generated_images = generator(z)
        generated_images = generated_images.cpu().data.numpy()

        # Plot the generated images
        plt.figure(figsize=(8, 8))
        for j in range(64):
            plt.subplot(8, 8, j+1)
            plt.imshow(generated_images[j], cmap='gray')
            plt.axis('off')
        plt.tight_layout()
        plt.savefig(f'generated_images_epoch_{epoch+1}.png')
        plt.close()

print("Training completed!")
