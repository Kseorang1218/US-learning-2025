import torch
import torch.nn as nn
import torch.nn.functional as F

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