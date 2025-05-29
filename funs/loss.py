# loss.py

from torch.nn import functional as F
import torch

class Loss:
    def __init__(self, reduction='sum'):
        self.reduction = reduction

    def __call__(self, recon_x, x, mu, log_var):
        recon_loss = F.mse_loss(recon_x, x, reduction=self.reduction)
        kl_d = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return recon_loss + kl_d