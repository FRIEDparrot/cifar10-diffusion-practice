import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid, save_image
from diffusers import DDPMScheduler
import os

def denormalize(x) -> torch.Tensor:
    return (x + 1) * 0.5

def show_grid_images(
        x,
        nrow=4,
        save_path=None,
        show_image=True):
    """
    Show batch images in a grid format.
        Image [B, C, H, W] -> grid image
    :param x:
    :param nrow:
    :param save_path:
    :param show_image: show image instantly
    :return:
    """
    x = denormalize(x)  # Rescale to [0, 1]
    grid = make_grid(x, nrow=nrow, normalize=False)  # [C, H_new, W_new] - already normalized
    grid_img = grid.detach().cpu().permute(1, 2, 0).numpy() # [H_new, W_new, C]
    img = Image.fromarray((grid_img * 255).astype(np.uint8))
    plt.imshow(img)
    plt.axis('off')
    if show_image:
        plt.show()
    if save_path is not None:
        save_image(grid, save_path)

def show_batch_compare(
        x, x_cmp,
        title1="Original", title2="Comparison", max_compare=5, save_path=None,
        show_image=False):
    x     = denormalize(x.detach().cpu()).clamp(0, 1)[:max_compare]
    x_cmp = denormalize(x_cmp.detach().cpu()).clamp(0, 1)[:max_compare]

    grid1 = make_grid(x,     nrow=max_compare, padding=2)
    grid2 = make_grid(x_cmp, nrow=max_compare, padding=2)

    fig, axes = plt.subplots(2, 1, figsize=(max_compare * 2, 5))

    for ax, grid, title in zip(axes, [grid1, grid2], [title1, title2]):
        ax.imshow(grid.permute(1, 2, 0).numpy())
        ax.set_title(title, fontsize=13, fontweight='bold', pad=6)
        ax.axis('off')

    plt.tight_layout()

    if save_path is not None:
        # check if has dir first
        save_path = os.path.abspath(save_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=150)

    if show_image:
        plt.show()
