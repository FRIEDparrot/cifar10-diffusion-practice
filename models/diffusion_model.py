import torch
from torch import nn
from torch.nn import functional as F
from diffusers import UNet2DModel, DDPMScheduler
from diffusers.utils.torch_utils import randn_tensor

class DiffusionModel(nn.Module):
    """
    Simple diffusion model that wraps a UNet2DModel
        and DDPMScheduler to compute noise predictions and diffusion loss.
    """
    def __init__(self, unet: UNet2DModel, noise_scheduler: DDPMScheduler):
        super().__init__()
        self.unet = unet
        self.noise_scheduler = noise_scheduler

    def forward(self, x, t):
        noise_pred = self.unet(x, t).sample
        return noise_pred

    def diffusion_loss(
        self, x: torch.Tensor, t: torch.IntTensor,
    ):
        """
        Compute diffusion loss: MSE between predicted and actual noise.

        Note: Does not modify model's train/eval mode - caller should set appropriately.

        :param x: Input images [B, C, H, W]
        :param t: Timesteps [B]
        :return: (noised_images, loss)
        """
        noise = randn_tensor(x.shape, device=x.device)  # [B, C, H, W]
        x_noisy = self.noise_scheduler.add_noise(x, noise, t)
        noise_pred = self(x_noisy, t)
        loss = F.mse_loss(noise_pred, noise)
        return x_noisy, loss

