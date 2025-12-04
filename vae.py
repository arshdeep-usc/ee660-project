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
            nn.ReLU(),
            nn.Linear(h, h),
        )
    def forward(self, x):
        return x + self.net(x)

# ---------- VAE Model ----------
class VAE(nn.Module):
    def __init__(self, d=2, h=256, z=2, n_res=3):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(d, h),
            nn.ReLU(),
            nn.LayerNorm(h),
            *[ResBlock(h) for _ in range(n_res)],
        )
        self.fc_mu = nn.Linear(h, z)
        self.fc_logvar = nn.Linear(h, z)

        # Decoder
        self.dec_in = nn.Sequential(
            nn.Linear(z, h),
            nn.ReLU(),
            nn.LayerNorm(h),
        )
        self.decoder = nn.Sequential(
            *[ResBlock(h) for _ in range(n_res)],
            nn.Linear(h, d)  # output is unbounded
        )

    def forward(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        h = self.dec_in(z)
        return self.decoder(h), mu, logvar

    @torch.no_grad()
    def sample(self, n=2000, device="cpu"):
        z_ = torch.randn(n, self.fc_mu.out_features, device=device)
        h = self.dec_in(z_)
        return self.decoder(h).cpu().numpy()

# ---------- Loss Function ----------
def vae_loss(x, x_recon, mu, logvar, beta=0.2):
    recon = nn.MSELoss(reduction="mean")(x_recon, x)
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + beta * kl

def train_vae_(loader, X, epochs=1500, device="cpu"):
    print("\nTraining VAE...")
    
    # Initialize model and optimizer
    vae = VAE(d=2, h=128, z=24, n_res=3).to(device)
    opt = optim.Adam(vae.parameters(), lr=1e-3)
    
    for _ in tqdm(range(epochs), desc="Epochs"):
        for (x,) in loader:
            x = x.to(device).float()
            opt.zero_grad()
            x_recon, mu, logvar = vae(x)
            loss = vae_loss(x, x_recon, mu, logvar, beta=0.2)
            loss.backward()
            opt.step()
    
    # Generate samples from the trained VAE
    vae.eval()
    vae_samples = vae.sample(n=2000, device=device)
    
    
    plt.figure(figsize=(5,5))
    plt.scatter(X[:,0], X[:,1], s=2, alpha=0.2)
    plt.scatter(vae_samples[:,0], vae_samples[:,1], s=3)
    plt.title(f"VAE")
    plt.show()

def train_vae(loader, X, epochs=1500, device="cpu"):
    print("\nTraining VAE...")

    vae = VAE(d=2, h=128, z=24, n_res=3).to(device)
    opt = optim.Adam(vae.parameters(), lr=1e-3)

    # Lists for logging
    epoch_losses = []
    recon_losses = []
    kl_losses = []

    for epoch in tqdm(range(epochs), desc="Epochs"):
        running_loss = 0
        running_recon = 0
        running_kl = 0
        count = 0

        for (x,) in loader:
            x = x.to(device).float()
            opt.zero_grad()

            x_recon, mu, logvar = vae(x)

            # compute losses
            recon = nn.MSELoss(reduction="mean")(x_recon, x)
            kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon + 0.2 * kl

            loss.backward()
            opt.step()

            # accumulate
            batch_size = x.size(0)
            running_loss += loss.item() * batch_size
            running_recon += recon.item() * batch_size
            running_kl += kl.item() * batch_size
            count += batch_size

        epoch_losses.append(running_loss / count)
        recon_losses.append(running_recon / count)
        kl_losses.append(running_kl / count)

    # --- Plot losses ---
    plt.figure(figsize=(7,5))
    plt.plot(epoch_losses, label="Total Loss")
    plt.plot(recon_losses, label="Reconstruction Loss")
    plt.plot(kl_losses, label="KL Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("VAE Training Losses")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

    # ----- Sampling -----
    vae.eval()
    vae_samples = vae.sample(n=2000, device=device)

    plt.figure(figsize=(5,5))
    plt.scatter(X[:,0], X[:,1], s=2, alpha=0.2)
    plt.scatter(vae_samples[:,0], vae_samples[:,1], s=3)
    plt.title("VAE Samples vs Data")
    plt.show()
