# loss.py

from torch.nn import functional as F
import torch

class VAELoss:
    def __init__(self, reduction='sum'):
        self.reduction = reduction

    def __call__(self, recon_x, x, mu, log_var):
        if self.reduction == 'sum':
            recon_loss = F.mse_loss(recon_x, x, reduction=self.reduction)
            kl_d = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        else:
            recon_loss = F.mse_loss(recon_x, x, reduction=self.reduction)
            recon_loss = recon_loss.view(recon_loss.size(0), -1).sum(dim=1)  # shape: (batch,)
            kl_d = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)

        return recon_loss + kl_d