# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, input_dim=4096, latent_dim=32):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(input_dim, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

        self.fc3 = nn.Linear(latent_dim, 256)
        self.fc4 = nn.Linear(256, 1024)
        self.fc5 = nn.Linear(1024, input_dim)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        mu = self.fc_mu(h2)
        logvar = self.fc_logvar(h2)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        h4 = F.relu(self.fc4(h3))
        return self.fc5(h4)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar
    
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

class AE(nn.Module):
    def __init__(self, input_dim=4096, latent_dim=32):
        super(AE, self).__init__()
        self.fc1 = nn.Linear(input_dim, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, latent_dim)

        self.fc4 = nn.Linear(latent_dim, 256)
        self.fc5 = nn.Linear(256, 1024)
        self.fc6 = nn.Linear(1024, input_dim)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        h3 = F.relu(self.fc3(h2))
        return h3

    def decode(self, z):
        h4 = F.relu(self.fc4(z))
        h5 = F.relu(self.fc5(h4))
        h6 = F.relu(self.fc6(h5))
        return h6

    def forward(self, x):
        z = self.encode(x)
        recon = self.decode(z)
        return recon

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