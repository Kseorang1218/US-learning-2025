# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
    
class Conv1DVAE(nn.Module):
    def __init__(self, input_length=4096, latent_dim=32):
        super(Conv1DVAE, self).__init__()
        self.input_length = input_length

        # Encoder: [B, 1, 4096] → [B, 256, 16]
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=4, stride=2, padding=1),   # [B, 32, 2048]
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=4, stride=2, padding=1),  # [B, 64, 1024]
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=4, stride=2, padding=1), # [B, 128, 512]
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=4, stride=2, padding=1),# [B, 256, 256]
            nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size=4, stride=2, padding=1),# [B, 256, 128]
            nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size=4, stride=2, padding=1),# [B, 256, 64]
            nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size=4, stride=2, padding=1),# [B, 256, 32]
            nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size=4, stride=2, padding=1),# [B, 256, 16]
            nn.ReLU(),
        )

        self.flatten = nn.Flatten()
        self.fc_mu = nn.Linear(256 * 16, latent_dim)
        self.fc_logvar = nn.Linear(256 * 16, latent_dim)

        # Decoder
        self.fc_z = nn.Linear(latent_dim, 256 * 16)
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(256, 256, kernel_size=4, stride=2, padding=1), # [B, 256, 32]
            nn.ReLU(),
            nn.ConvTranspose1d(256, 256, kernel_size=4, stride=2, padding=1), # [B, 256, 64]
            nn.ReLU(),
            nn.ConvTranspose1d(256, 256, kernel_size=4, stride=2, padding=1), # [B, 256, 128]
            nn.ReLU(),
            nn.ConvTranspose1d(256, 256, kernel_size=4, stride=2, padding=1), # [B, 256, 256]
            nn.ReLU(),
            nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2, padding=1), # [B, 128, 512]
            nn.ReLU(),
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),  # [B, 64, 1024]
            nn.ReLU(),
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),   # [B, 32, 2048]
            nn.ReLU(),
            nn.ConvTranspose1d(32, 1, kernel_size=4, stride=2, padding=1),    # [B, 1, 4096]
            nn.ReLU(), 
        )

    def encode(self, x):
        h = self.encoder(x)                      # [B, 256, 16]
        h_flat = self.flatten(h)                 # [B, 256*16]
        mu = self.fc_mu(h_flat)                  # [B, latent_dim]
        logvar = self.fc_logvar(h_flat)          # [B, latent_dim]
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std                    # [B, latent_dim]

    def decode(self, z):
        h = self.fc_z(z)                         # [B, 256*16]
        h = h.view(-1, 256, 16)  
        out = self.decoder(h)                # [B, 256, 16]
        return out.squeeze(1)                   # [B, 1, 4096]

    def forward(self, x):
        x = x.reshape(x.size(0), 1, -1)
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar
