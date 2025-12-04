import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from tqdm import tqdm

# ---------- Residual Block ----------
class ResBlock(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(h, h),
            nn.LeakyReLU(),
            nn.Linear(h, h),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return x + self.net(x)

# ---------- Generator ----------
class Generator(nn.Module):
    def __init__(self, z=2, h=24, d=2, n_res=4):
        super().__init__()
        self.z_dim = z   

        self.input = nn.Sequential(
            nn.Linear(z, h),
            nn.LeakyReLU()
        )
        self.body = nn.Sequential(*[ResBlock(h) for _ in range(n_res)])
        self.output = nn.Linear(h, d)

    def forward(self, z):
        h = self.input(z)
        h = self.body(h)
        return self.output(h)

# ---------- Discriminator ----------
class Discriminator(nn.Module):
    def __init__(self, d=2, h=24, n_res=3):
        super().__init__()
        self.input = nn.Sequential(
            nn.Linear(d, h),
            nn.LeakyReLU(),
        )
        self.body = nn.Sequential(*[ResBlock(h) for _ in range(n_res)])
        self.output = nn.Sequential(
            nn.Linear(h, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        h = self.input(x)
        h = self.body(h)
        return self.output(h)

# ---------- Sampling function ----------
@torch.no_grad()
def sample_gan(G, n=2000, device="cpu"):
    G.eval()
    z = torch.randn(n, G.z_dim, device=device)   
    return G(z).cpu().numpy()

def train_gan_(loader, X, epochs=1500, device="cpu"):
    print("\nTraining GAN...")
    
    # Initialize models
    G = Generator(z=2, h=32, d=2, n_res=4).to(device)
    D = Discriminator(d=2, h=64, n_res=3).to(device)
    
    optG = optim.Adam(G.parameters(), lr=1e-3, betas=(0.5, 0.9))
    optD = optim.Adam(D.parameters(), lr=1e-3, betas=(0.5, 0.9))
    
    bce = nn.BCELoss()
    
    for _ in tqdm(range(epochs), desc="Epochs"):
        for (x,) in loader:
            x = x.to(device).float()
            bs = x.size(0)
    
            # ---------------------
            # Train Discriminator
            # ---------------------
            z = torch.randn(bs, 2, device=device)
            fake = G(z)
    
            optD.zero_grad()
            loss_real = bce(D(x), torch.ones(bs, 1, device=device))
            loss_fake = bce(D(fake.detach()), torch.zeros(bs, 1, device=device))
            loss_D = loss_real + loss_fake
            loss_D.backward()
            optD.step()
    
            # ---------------------
            # Train Generator
            # ---------------------
            optG.zero_grad()
            loss_G = bce(D(fake), torch.ones(bs, 1, device=device))
            loss_G.backward()
            optG.step()
    
    # Sample from trained GAN
    G.eval()
    gan_samples = sample_gan(G, n=2000, device=device)
    
    plt.figure(figsize=(5,5))
    plt.scatter(X[:,0], X[:,1], s=2, alpha=0.2)
    plt.scatter(gan_samples[:,0], gan_samples[:,1], s=3)
    plt.title(f"GAN")
    plt.show()

def train_gan(loader, X, epochs=1500, device="cpu"):
    print("\nTraining GAN...")
    
    # Initialize models
    G = Generator(z=2, h=32, d=2, n_res=2).to(device)
    D = Discriminator(d=2, h=42, n_res=4).to(device)
    
    optG = optim.Adam(G.parameters(), lr=1e-3, betas=(0.5, 0.9))
    optD = optim.Adam(D.parameters(), lr=1e-3, betas=(0.5, 0.9))
    
    bce = nn.BCELoss()

    # --- Loss storage ---
    G_losses = []
    D_losses = []

    for epoch in tqdm(range(epochs), desc="Epochs"):
        running_G = 0.0
        running_D = 0.0
        total_samples = 0

        for (x,) in loader:
            x = x.to(device).float()
            bs = x.size(0)
            total_samples += bs

            # -------------------------
            # Train Discriminator
            # -------------------------
            z = torch.randn(bs, 2, device=device)
            fake = G(z)

            optD.zero_grad()
            loss_real = bce(D(x), torch.ones(bs, 1, device=device))
            loss_fake = bce(D(fake.detach()), torch.zeros(bs, 1, device=device))
            loss_D = loss_real + loss_fake
            loss_D.backward()
            optD.step()

            # -------------------------
            # Train Generator
            # -------------------------
            optG.zero_grad()
            loss_G = bce(D(fake), torch.ones(bs, 1, device=device))
            loss_G.backward()
            optG.step()

            running_D += loss_D.item() * bs
            running_G += loss_G.item() * bs

        # Epoch averages
        D_losses.append(running_D / total_samples)
        G_losses.append(running_G / total_samples)

    # -----------------------------
    # Plot Loss Curves
    # -----------------------------
    plt.figure(figsize=(7,5))
    plt.plot(D_losses, label="Discriminator Loss")
    plt.plot(G_losses, label="Generator Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("GAN Training Losses")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.show()

    # -----------------------------
    # Plot Generated Samples
    # -----------------------------
    G.eval()
    gan_samples = sample_gan(G, n=2000, device=device)

    plt.figure(figsize=(5,5))
    plt.scatter(X[:,0], X[:,1], s=2, alpha=0.2)
    plt.scatter(gan_samples[:,0], gan_samples[:,1], s=3)
    plt.title("GAN")
    plt.show()
