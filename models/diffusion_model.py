import torch
from torch import nn
from torch.nn import functional as F
from diffusers import UNet2DModel, DDPMScheduler
from diffusers.utils.torch_utils import randn_tensor

class DiffusionModel(nn.Module):
    def __init__(self, unet: UNet2DModel, noise_scheduler: DDPMScheduler):
        super().__init__()
        self.model = unet
        self.noise_scheduler = noise_scheduler

    def forward(self, x, t):
        noise_pred = self.model(x, t).sample
        return noise_pred

    def diffusion_loss(
        self, x: torch.Tensor, t: torch.IntTensor,
    ):
        """
        Do an epoch for training the diffusion model
        Generate random tensor, do a `forward` and compute the loss

        return the loss
        """
        self.train()
        noise = randn_tensor(x.shape, device=x.device)  # [B, C, H, W]
        x_noisy = self.noise_scheduler.add_noise(x, noise, t)
        noise_pred = self(x_noisy, t)
        loss = F.mse_loss(noise_pred, noise)
        return x_noisy, loss

