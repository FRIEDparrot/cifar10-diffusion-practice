import json
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader
from datasets import load_dataset
from diffusers import UNet2DModel, DDPMScheduler
from diffusers.utils.torch_utils import randn_tensor  # random generator
from accelerate import Accelerator
import matplotlib.pyplot as plt
from torchvision.utils import make_grid, save_image
from torchvision import transforms
from dataclasses import dataclass, asdict
from huggingface_hub import HfApi
from tqdm import tqdm
import wandb
import os

"""
Practice for diffusion model on CIFAR10 dataset. 

Remove the clamp in forward_diffusion
Fix noise generation in test_denoising_results — move it inside the loop
Add importance sampling for timesteps (optional but helps high-t learning)
"""
#region Utility functions
def denormalize(x) -> torch.Tensor:
    return (x + 1) * 0.5

def show_grid_images(x, nrow=4, save_path=None, show_image=False):
    """
    Show batch images in a grid format.
    Image [B, C, H, W] -> grid image
    :param x:
    :param nrow:
    :param save_path:
    :param show_image
    :return:
    """
    x = denormalize(x)  # Rescale to [0, 1]
    grid = make_grid(x, nrow=nrow, normalize=True)  # [C, H_new, W_new]
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

def forward_diffusion(x, t, noise, scheduler:DDPMScheduler):
    x_t = scheduler.add_noise(x, noise, t)
    return x_t.clamp(min=-1, max=1)
#endregion

class DiffusionModel(nn.Module):
    def __init__(self, unet: UNet2DModel, scheduler: DDPMScheduler):
        super().__init__()
        self.model = unet
        self.scheduler = scheduler

    def forward(self, x, t):
        noise_pred = self.model(x, t).sample
        return noise_pred

preprocess = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# Custom preprocess function that handles multi-field datasets
def transform(batch):
    """Apply transforms to each image in the batch"""
    batch["img"] = [preprocess(img) for img in batch["img"]]
    return batch

def train_step(
    model: DiffusionModel,
    x,
):
    """
    epoch for training the diffusion model

    Sample a random timestep t for each image in the batch.
    Note: use `num_train_timesteps` from the scheduler config to determine the range of t.
    """
    model.train()
    batch_size = x.shape[0]
    t = torch.randint(
        0, model.scheduler.config["num_train_timesteps"],
        (batch_size,), device=x.device
    ).long()

    noise = randn_tensor(x.shape, device=x.device)  # [B, C, H, W]
    x_noisy = forward_diffusion(x, t, noise, scheduler=model.scheduler)
    noise_pred = model(x_noisy, t)
    loss = F.mse_loss(noise_pred, noise)
    return x_noisy, loss

@torch.no_grad()
def validate(
    model: DiffusionModel,
    val_loader: DataLoader,
    accelerator: Accelerator,
    max_batches: int = 20,  # don't run full test set, just a subset
) -> float:
    model.eval()
    total_loss = 0.0
    for i, batch in enumerate(val_loader):
        if i >= max_batches:
            break
        x = batch["img"]
        batch_size = batch["img"].shape[0]
        t = torch.randint(
            0, model.scheduler.config["num_train_timesteps"],
            (batch_size,), device=x.device
        ).long()
        noise = randn_tensor(x.shape, device=x.device)  # [B, C, H, W]
        x_noisy = forward_diffusion(x, t, noise, scheduler=model.scheduler)
        noise_pred = model(x_noisy, t)
        loss = F.mse_loss(noise_pred, noise)
        total_loss += loss.item()
    return total_loss / min(max_batches, len(val_loader))

@torch.no_grad()
def test_denoising_results(
    model: DiffusionModel,
    scheduler: DDPMScheduler,
    accelerator: Accelerator,
    batch: torch.Tensor,
    test_num=4,
    save_path=None
):
    """
    Instead, prepare all data on the correct device upfront and reuse it.
    :return:
    """
    model.eval()
    device = accelerator.device
    T = scheduler.config.get("num_train_timesteps")
    scheduler.set_timesteps(T)

    diffuse_times = torch.linspace(0, T - 1, test_num).long()
    # Prepare all data upfront with batching
    denoised_images = []

    noise = randn_tensor(batch["img"][0].shape, device=device)
    for i, diffuse_time in enumerate(diffuse_times):
        x = batch["img"][i].unsqueeze(0).to(device)
        x_noisy = forward_diffusion(x, diffuse_time, noise, scheduler=scheduler)
        x_denoised = x_noisy.clone()
        timesteps = scheduler.timesteps[scheduler.timesteps <= diffuse_time]
        for t in timesteps:
            noise_pred = model(x_denoised, t)
            x_denoised = scheduler.step(noise_pred, t, x_denoised).prev_sample
        denoised_images.append(x_denoised)

    denoised_grid = torch.cat(denoised_images, dim=0)  # [test_num, C, H, W]
    show_batch_compare(
        batch["img"][:test_num].cpu(),
        denoised_grid.cpu(),
        title1=f"Original Images",
        title2=f"Denoised (full reverse)",
        max_compare=test_num,
        save_path=save_path
    )

@torch.no_grad()
def generate_images(model: DiffusionModel, save_dir: str, epoch: int, num_samples: int = 16, device="cuda"):
    """
    Generate images from pure noise via full reverse diffusion and save a grid.
    Returns the saved image path and the generated image tensor.
    """
    model.eval()
    scheduler = model.scheduler
    # Ensure scheduler is set to the correct number of timesteps
    scheduler.set_timesteps(scheduler.config["num_train_timesteps"])

    # Start from pure Gaussian noise
    x = torch.randn(num_samples, 3, 32, 32, device=device)
    # Reverse diffusion loop
    for t in scheduler.timesteps:
        t_batch = torch.full((num_samples,), t, device=device, dtype=torch.long)
        noise_pred = model(x, t_batch)
        x = scheduler.step(noise_pred, t, x).prev_sample

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"samples_epoch_{epoch:03d}.png")
    show_grid_images(x, nrow=4, save_path=save_path)
    return save_path, x

@dataclass
class TrainConfigs:
    max_epoch:int = 400   # 300 - 400  for real training
    lr: float = 3e-4
    weight_decay: float = 1e-5
    batch_size: int = 128
    repo_id: str = "FriedParrot/ddpm-cifar10-diffusion"
    local_save_dir: str = "./checkpoints"
    checkpoint_epoch = 10

def main():
    accelerator = Accelerator(
        device_placement=True,
        mixed_precision="fp16",
    )
    save_dir = "validation_samples"
    configs = TrainConfigs()  # all process needs it
    api = HfApi()

    if accelerator.is_main_process:
        # save train configs
        with open("train_config.json", "w") as f:
            json.dump(asdict(configs), f, indent=2)
        api.create_repo(
            repo_id=configs.repo_id,
            private=False,
            exist_ok=True,
        )
        accelerator.print("Repo was successfully created")
        api.upload_file(
            path_or_fileobj="train_config.json",
            path_in_repo="train_config.json",
            repo_id=configs.repo_id,
        )
        wandb.init(
            project="diffusion-cifar10",
            name="diffusion-cifar10-train-optimized",
            config=configs.__dict__,
        )

    dataset_link = "uoft-cs/cifar10"
    train_set = load_dataset(dataset_link, split="train").with_format("torch")
    val_set = load_dataset(dataset_link, split="test").with_format("torch")
    train_set.set_transform(transform)
    val_set.set_transform(transform)

    model_repo = "google/ddpm-cifar10-32"
    # not load weights since we will train from scratch, but use the same architecture as the pretrained model
    unet = UNet2DModel.from_config(model_repo)
    scheduler:DDPMScheduler = DDPMScheduler.from_pretrained(model_repo)

    optimizer = torch.optim.Adam(
        unet.parameters(),
        lr=configs.lr,
        weight_decay=configs.weight_decay,
        eps=1e-8,
    )
    lr_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=configs.max_epoch,
        eta_min=1e-7
    )

    dataloader = DataLoader(train_set, batch_size=configs.batch_size)
    val_loader = DataLoader(val_set, batch_size=configs.batch_size)
    test_loader = DataLoader(train_set, batch_size=8)

    model = DiffusionModel(unet, scheduler)
    model, optimizer, dataloader, val_loader, test_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, dataloader, val_loader, test_loader, lr_scheduler
    )
    global_step = 0
    max_epoch = configs.max_epoch

    for epoch in range(max_epoch):
        avg_loss = 0.0
        progress_bar = tqdm(
            range(len(dataloader)),
            disable=not accelerator.is_main_process,  # Only show on the main GPU
            desc=f"Epoch {epoch} / {max_epoch}"
        )
        for batch in dataloader:
            images = batch["img"]
            optimizer.zero_grad()
            with accelerator.autocast():
                output, loss = train_step(model, images)
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            avg_loss += loss.item()
            if accelerator.is_main_process:
                wandb.log({"train/loss_step": loss.item(), "step": global_step})
            global_step += 1
            progress_bar.update(1)
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss /= len(dataloader)
        lr_scheduler.step()  # step lr scheduler by epoch

        val_loss = validate(model, val_loader, accelerator)
        # Also test the denoising results on a few test images

        if accelerator.is_main_process:
            wandb.log({
                "train/loss_epoch": avg_loss,
                "val/loss": val_loss,
                "epoch": epoch
            })

        if epoch % configs.checkpoint_epoch == 0 :
            # save checkpoints
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                os.makedirs("./checkpoints", exist_ok=True)
                # Validation: generate and save images
                save_path, generated = generate_images(
                    model, save_dir=save_dir, epoch=epoch, device=accelerator.device
                )
                test_batch = next(iter(test_loader))
                test_image_path = "test_denoising.png"
                test_denoising_results(
                    model=model,
                    scheduler=scheduler,
                    accelerator=accelerator,
                    batch=test_batch,
                    test_num=8,
                    save_path=test_image_path
                )
                wandb.log({
                    "train/test_image": wandb.Image(test_image_path, caption=f"Epoch {epoch} - Denoising Test"),
                    "val/generated_samples": wandb.Image(save_path, caption=f"Epoch {epoch} - Generated Samples"),
                    "epoch": epoch,
                })
                accelerator.save_model(model, f"./checkpoints/checkpoint_epoch_{epoch:03d}.pt")

    accelerator.end_training()  # ensures all processes synchronize before the main process handles

    if accelerator.is_main_process:
        wandb.finish()
        model_save_path = "./checkpoints/cifar_diffusion_model.pt"
        accelerator.save_model(model, model_save_path)


if __name__ == "__main__":
    main()