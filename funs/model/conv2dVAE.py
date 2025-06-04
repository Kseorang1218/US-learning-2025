import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv2DVAE(nn.Module):
    def __init__(self, input_shape=(1, 257, 626), latent_dim=32):
        super(Conv2DVAE, self).__init__()
        self.input_shape = input_shape
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),   # [B, 32, 129, 313]
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # [B, 64, 65, 157]
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # [B, 128, 33, 79]
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),# [B, 256, 17, 40]
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=4, stride=2, padding=1),# [B, 256, 9, 20]
            nn.ReLU()
        )

        self.flatten = nn.Flatten()
        self.encoder_output_dim = 256 * 9 * 20
        self.fc_mu = nn.Linear(self.encoder_output_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.encoder_output_dim, latent_dim)

        # Decoder
        self.fc_z = nn.Linear(latent_dim, self.encoder_output_dim)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1),  # [B, 256, 17, 40]
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # [B, 128, 33, 79]
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),   # [B, 64, 65, 158]
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),    # [B, 32, 130, 316]
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),     # [B, 1, 260, 632]
            nn.ReLU()  # or ReLU depending on normalization
        )

    def encode(self, x):
        h = self.encoder(x)                      # [B, 256, 9, 20]
        h_flat = self.flatten(h)                 # [B, 256*9*20]
        mu = self.fc_mu(h_flat)                  # [B, latent_dim]
        logvar = self.fc_logvar(h_flat)          # [B, latent_dim]
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std                    # [B, latent_dim]

    def decode(self, z):
        h = self.fc_z(z)                         # [B, 256*9*20]
        h = h.view(-1, 256, 9, 20)               # [B, 256, 9, 20]
        out = self.decoder(h)                    # [B, 1, ~260, ~632]
        return out

    def forward(self, x):
        if x.ndim == 3:
            x = x.unsqueeze(1)  # [B, 1, H, W] 보장
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar