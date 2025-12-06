import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import gaussian_kde
import datetime  

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

# ---------- Training Function ----------
def train_vae(loader, X, hold_out, hold_out_data, epochs=1500, device="cpu"):
    print("\nTraining VAE...")
    vae = VAE(d=2, h=128, z=24, n_res=3).to(device)
    opt = optim.Adam(vae.parameters(), lr=1e-3)

    # Lists for logging
    epoch_losses = []
    recon_losses = []
    kl_losses = []
    hold_out_losses = []

    for epoch in tqdm(range(epochs), desc="Epochs"):
        running_loss = 0
        running_recon = 0
        running_kl = 0
        count = 0

        ho_running_loss = 0
        ho_count = 0

        vae.eval()
        for (x,) in hold_out:
            x = x.to(device).float()
            x_recon, mu, logvar = vae(x)

            # compute losses
            recon = nn.MSELoss(reduction="mean")(x_recon, x)
            kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon + 0.2 * kl

            # accumulate
            batch_size = x.size(0)
            ho_running_loss += loss.item() * batch_size
            ho_count += batch_size
            
        hold_out_losses.append(ho_running_loss / ho_count)
        vae.train()

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



    # -----------------------------
    # Plot Loss Curves
    # -----------------------------
    print(f"Avg Trainig Loss = {np.mean(epoch_losses)}")
    print(f"Avg Hold Out Loss = {np.mean(hold_out_losses)}")
    plt.figure(figsize=(7,5))
    plt.plot(epoch_losses, label="Total Loss")
    plt.plot(recon_losses, label="Reconstruction Loss")
    plt.plot(kl_losses, label="KL Loss")
    plt.plot(hold_out_losses, label="Hold Out Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("VAE Training Losses")
    plt.legend()
    plt.grid(alpha=0.3)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f"vae_loss_curves_{timestamp}.png")
    plt.show()

    # ----- Sampling -----
    vae.eval()
    vae_samples = vae.sample(n=2000, device=device)

    plt.figure(figsize=(5,5))
    plt.scatter(X[:,0], X[:,1], s=2, alpha=0.2, label="Training")
    plt.scatter(vae_samples[:,0], vae_samples[:,1], s=3, label="Samples")
    plt.scatter(hold_out_data[:,0], hold_out_data[:,1], s=2, alpha=0.5, color='green', label="Hold Out")
    plt.legend()
    plt.title("VAE Samples vs Data")
    plt.savefig(f"vae_scatter_samples_{timestamp}.png")
    plt.show()

    # If vae_samples is already a numpy array:
    if isinstance(vae_samples, np.ndarray):
        vae_samples_np = vae_samples.T
    else:
        vae_samples_np = vae_samples.cpu().numpy().T

    # ------- KDE on VAE samples -------
    kde = gaussian_kde(vae_samples_np)

    # Create grid for density evaluation
    x_min, x_max = X[:,0].min(), X[:,0].max()
    y_min, y_max = X[:,1].min(), X[:,1].max()
    xx, yy = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
    grid = np.vstack([xx.ravel(), yy.ravel()])
    zz = kde(grid).reshape(xx.shape)

    plt.figure(figsize=(6,5))
    plt.contourf(xx, yy, zz, levels=50, cmap="viridis")
    plt.scatter(vae_samples_np[0], vae_samples_np[1], s=5, alpha=0.3, color="red")
    plt.title("KDE of VAE-Generated Samples")
    plt.colorbar(label="Density")
    plt.savefig(f"vae_kde_density_{timestamp}.png")
    plt.show()

    # Fit KDE on model samples (already computed above as "kde")
    logpdf_vals = kde.logpdf(X.T)
    kde_ll = float(logpdf_vals.mean())
    print(f"\nKDE Estimated Log-Likelihood (VAE): {kde_ll:.4f}\n")
