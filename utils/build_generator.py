import torch
import torch.nn as nn
from torchvision import datasets, transforms
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np
import os

# MNIST is grayscale (1 channel), so we use a single mean/std value
transform = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)) 
])

class ConditionalGenerator(nn.Module):
    def __init__(self, nz=100, ngf=64, num_classes=10):
        super(ConditionalGenerator, self).__init__()
        
        # 1. Label Embedding: Turns class 0-9 into a 50-dim vector
        self.label_emb = nn.Embedding(num_classes, 50)
        
        # 2. Linear layer to process the label embedding
        self.label_linear = nn.Linear(50, 4*4) # Reshape to patch for conv
        
        # 3. Process the noise vector
        self.noise_linear = nn.Linear(nz, ngf*4 * 4*4)

        # 4. Generator Backbone (similar upsampling structure)
        self.main = nn.Sequential(
            # Input: concatenated features (Label patch + Noise patch)
            # Size should match: (ngf*4 + 1) channels
            nn.ConvTranspose2d(257,ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True),
            # (ngf*2) x 8 x 8
            nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # (ngf) x 16 x 16
            nn.ConvTranspose2d(ngf, 1, 4, 2, 1, bias=False),
            nn.Tanh()
            # 3 x 32 x 32
        )

    def forward(self, noise, labels):
        # Process Label: Embed, Reshape to (Batch, 1, 4, 4)
        c = self.label_emb(labels)
        c = self.label_linear(c).view(-1, 1, 4, 4)
        
        # Flatten noise for the linear layer
        noise = noise.view(noise.size(0), -1) 
        
        # FIX: Use the specific number of channels your linear layer produces.
        # If ngf=64, then ngf*4 = 256. 
        # Calculation: 256 channels * 4 * 4 pixels = 4096.
        z = self.noise_linear(noise).view(-1, 256, 4, 4) # Hard-code to 256
        
        # Concatenate: (Batch, 256, 4, 4) + (Batch, 1, 4, 4) -> (Batch, 257, 4, 4)
        x = torch.cat([z, c], 1) 
        return self.main(x)


class ConditionalDiscriminator(nn.Module):
    def __init__(self, ndf=64, num_classes=10):
        super(ConditionalDiscriminator, self).__init__()
        
        # 1. Label Embedding (for input size 32x32)
        self.label_emb = nn.Embedding(num_classes, 50)
        self.label_linear = nn.Linear(50, 32*32) # Match image resolution
        
        # 2. Initial convolution for the image (Input: 1+1 channels)
        self.conv1 = nn.Sequential(
            nn.Conv2d(2, ndf, 4, 2, 1, bias=False), 
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # 3. Discriminator Backbone
        self.main = nn.Sequential(
            # (ndf) x 16 x 16
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # (ndf*2) x 8 x 8
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # (ndf*4) x 4 x 4
            nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False)
            # Remove Sigmoid since we use BCEWithLogitsLoss
        )

    def forward(self, image, labels):
        # Process Label: Embed, Reshape to (Batch, 1, 32, 32)
        c = self.label_emb(labels)
        c = self.label_linear(c).view(-1, 1, 32, 32)
        
        # Concatenate Label Map with Image Channels
        x = torch.cat([image, c], 1) # [Batch, 2, 32, 32]
        
        x = self.conv1(x)
        return self.main(x).view(-1)


def train():
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(root='./data', train=True, download=True, transform=transform),
        batch_size=128, shuffle=True
    )

    # --- Configuration ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nz = 100 
    epochs = 25
    lr = 0.0002
    beta1 = 0.5
    scaler = torch.amp.GradScaler('cuda') # Modern API

    # Fixed noise AND fixed labels to track specific digit progress
    # We'll generate 8 images for each of the 10 digits (80 total)
    fixed_noise = torch.randn(80, nz, 1, 1, device=device)
    fixed_labels = torch.tensor([i for i in range(10) for _ in range(8)], device=device)

    netG = ConditionalGenerator().to(device)
    netD = ConditionalDiscriminator().to(device)
    criterion = nn.BCEWithLogitsLoss() # Safe for autocast

    optimizerD = torch.optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = torch.optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    print("Starting Conditional Training Loop...")

    for epoch in range(epochs):
        # Notice 'real_labels' is now captured from the loader
        for i, (data, real_labels) in enumerate(train_loader):
            
            batch_size = data.size(0)
            real_cpu = data.to(device)
            real_labels = real_labels.to(device)
            
            # Valid and Fake ground truth for the Discriminator
            valid = torch.full((batch_size,), 1.0, device=device)
            fake_label = torch.full((batch_size,), 0.0, device=device)

            ############################
            # (1) Update Discriminator
            ############################
            optimizerD.zero_grad()
            
            with torch.amp.autocast(device_type='cuda'):
                # D looks at real images AND real labels
                output_real = netD(real_cpu, real_labels)
                loss_real = criterion(output_real, valid)
                
                # D looks at fake images AND the labels G was told to make
                gen_labels = torch.randint(0, 10, (batch_size,), device=device)
                noise = torch.randn(batch_size, nz, 1, 1, device=device)
                fake_images = netG(noise, gen_labels)
                
                output_fake = netD(fake_images.detach(), gen_labels)
                loss_fake = criterion(output_fake, fake_label)
                
                errD = (loss_real + loss_fake) / 2
            
            scaler.scale(errD).backward()
            scaler.step(optimizerD)

            ############################
            # (2) Update Generator
            ############################
            optimizerG.zero_grad()
            
            with torch.amp.autocast(device_type='cuda'):
                # G wants D to think its fake images match the 'gen_labels'
                output = netD(fake_images, gen_labels)
                errG = criterion(output, valid)
            
            scaler.scale(errG).backward()
            scaler.step(optimizerG)
            scaler.update()

            if i % 100 == 0:
                print(f'[{epoch}/{epochs}][{i}/{len(train_loader)}] Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f}')

        # Save progress snapshot
        with torch.no_grad():
            fake_display = netG(fixed_noise, fixed_labels).detach().cpu()

        plt.figure(figsize=(10, 8))
        plt.axis("off")
        plt.title(f"Generated Digits at Epoch {epoch}")
        grid = vutils.make_grid(fake_display, nrow=8, padding=2, normalize=True)
        plt.imshow(np.transpose(grid, (1, 2, 0)))
        plt.savefig(f"mnist_cgan_epoch_{epoch}.png")
        plt.close()

    os.makedirs("../models/generator", exist_ok=True)
    # Save final model weights
    torch.save(netG.state_dict(), "../models/generator/final_generator.pth")
    print("Training Complete. Snapshots saved to directory.")


if __name__ == "__main__":
    train()
