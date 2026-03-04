import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator
from diffusers.optimization import get_cosine_schedule_with_warmup
from models import DiffusionModel
from configs import TrainConfigs
from tqdm import tqdm
from typing import Optional, Tuple
import os

from .image_functions import show_grid_images


def setup_training(
    config: TrainConfigs,
    model: DiffusionModel,
    accelerator: Accelerator,
    train_loader: DataLoader,
    val_loader: DataLoader,
) -> Tuple[DiffusionModel, torch.optim.Optimizer, object, DataLoader, DataLoader]:
    """
    Setup optimizer, lr_scheduler, and prepare all components with accelerator.
    This eliminates the repetitive boilerplate in training scripts.

    :param config: Training configuration
    :param model: DiffusionModel instance (unconditional or conditional)
    :param accelerator: Accelerator instance
    :param train_loader: Training DataLoader
    :param val_loader: Validation DataLoader
    :return: Tuple of (model, optimizer, lr_scheduler, train_loader, val_loader) - all prepared by accelerator
    """
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
        eps=1e-8,
    )

    # Create learning rate scheduler with warmup
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=(config.max_epoch * len(train_loader)),
    )

    # Prepare all components with accelerator (handles device placement, distributed training, mixed precision)
    model, optimizer, train_loader, val_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, lr_scheduler
    )

    return model, optimizer, lr_scheduler, train_loader, val_loader


def train_step(
    model: DiffusionModel,
    config: TrainConfigs,
    accelerator: Accelerator,
    train_loader: DataLoader,
    optimizer,
    lr_scheduler,
    current_epoch: Optional[int] = 0
):
    model.train()
    num_timesteps = model.noise_scheduler.config["num_train_timesteps"]

    progress_bar = tqdm(
        range(len(train_loader)),
        disable=not accelerator.is_main_process,  # Only show on the main GPU
        desc=f"Train epoch {current_epoch + 1} / {config.max_epoch}"
    )
    avg_loss = 0
    for batch in train_loader:
        x = batch[config.image_field]
        batch_size = x.shape[0]
        t = torch.randint(0, num_timesteps, (batch_size,), device=accelerator.device).long()
        with accelerator.accumulate(model):
            optimizer.zero_grad()
            x_noisy, loss = model.diffusion_loss(x, t)
            # standard loss backward step
            accelerator.backward(loss)  # compute gradient

            # Only clip gradients on actual optimization steps (not during accumulation)
            if accelerator.sync_gradients and config.gradient_clipping is not None:
                accelerator.clip_grad_norm_(model.parameters(), max_norm=config.gradient_clipping)

            optimizer.step()
            lr_scheduler.step()

        avg_loss += loss.item()
        progress_bar.update(1)
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss /= len(train_loader)
    return avg_loss

@torch.no_grad()
def validate_step(
    model: DiffusionModel,
    config: TrainConfigs,
    accelerator: Accelerator,
    val_loader: DataLoader,
    current_epoch: Optional[int] = 0,
) -> float:
    model.eval()
    total_loss = 0.0
    progress_bar = tqdm(
        range(len(val_loader)),
        disable=not accelerator.is_main_process,  # Only show on the main GPU
        desc=f"Validation epoch {current_epoch + 1} / {config.max_epoch}"
    )
    for i, batch in enumerate(val_loader):
        x = batch[config.image_field]
        batch_size = x.shape[0]
        t = torch.randint(
            0, model.noise_scheduler.config["num_train_timesteps"],
            (batch_size,), device=accelerator.device
        ).long()
        x_noisy, loss = model.diffusion_loss(x, t)
        total_loss += loss
        progress_bar.update(1)
        progress_bar.set_postfix({"val_loss": f"{loss.item():.4f}"})
    return total_loss / len(val_loader)


@torch.no_grad()
def generate_images(
    model: DiffusionModel,
    config: TrainConfigs,
    accelerator: Accelerator,
    save_dir: str,
    epoch: int,
    num_samples: int = 16,
):
    """
    Make full reverse diffusion, generate images from pure noise via full reverse diffusion and save a grid.
    Returns the saved image path and the generated image tensor.
    """
    model.eval()
    scheduler = model.noise_scheduler
    device = accelerator.device
    # Start from pure Gaussian noise
    x = torch.randn(num_samples, 3, config.image_size, config.image_size, device=device)
    # Reverse diffusion loop (use fewer steps for faster validation)
    scheduler.set_timesteps(config.reverse_diffusion_steps)  # Use 50-100 steps instead of 1000 for faster inference
    for t in scheduler.timesteps:
        t_batch = torch.full((num_samples,), t, device=device, dtype=torch.long)
        noise_pred = model(x, t_batch)
        x = scheduler.step(noise_pred, t, x).prev_sample

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"samples_epoch_{epoch:03d}.png")
    show_grid_images(x, nrow=4, save_path=save_path, show_image=False)
    return save_path, x
