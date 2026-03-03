from typing import Optional

import torch
from huggingface_hub import HfApi
import wandb
from torch.utils.data import DataLoader
from diffusers import UNet2DModel, DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
from accelerate import Accelerator
from tqdm import tqdm
import os

from configs import TrainConfigs, load_dataloaders
from utils import show_grid_images
from models import DiffusionModel

"""
Practice for diffusion model

Fix noise generation in test_denoising_results — move it inside the loop
Add importance sampling for timesteps (optional but helps high-t learning)
"""

#region "Train and Validation"
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
            x_noisy, loss = model.diffusion_loss(x, t)
            # standard loss backward step
            optimizer.zero_grad()
            accelerator.backward(loss)  # compute gradient
            accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
    scheduler.set_timesteps(config.reverse_diffusion_steps)  # Use 50 steps instead of 1000 for faster inference
    for t in scheduler.timesteps:
        t_batch = torch.full((num_samples,), t, device=device, dtype=torch.long)
        noise_pred = model(x, t_batch)
        x = scheduler.step(noise_pred, t, x).prev_sample

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"samples_epoch_{epoch:03d}.png")
    show_grid_images(x, nrow=4, save_path=save_path, show_image=False)
    return save_path, x
#endregion

def main():
    # Config auto-saves to config_save_dir upon creation
    config = TrainConfigs(
        max_epoch=2,  # pretrained model
        dataset_name="huggan/few-shot-dog",
        image_size=128,
        train_batch_size=8,
        eval_batch_size=8,
        # unconditional diffusion model
        model_repo = "anton-l/ddpm-butterflies-128",
        image_field="image",
        remote_repo_id="FriedParrot/ddpm-few-shot-dog-128",
        checkpoint_epoch=5,
    )  # all process needs it
    accelerator = Accelerator(
        device_placement=True,
        mixed_precision="fp16",
        gradient_accumulation_steps=config.gradient_accumulation_steps,
    )
    train_loader, val_loader = load_dataloaders(config)
    # load pretrained model from butterflies-128.
    unet = UNet2DModel.from_pretrained(config.model_repo, subfolder="unet")  # not load weights firstly, use `from_pretrained` to use
    try:
        noise_scheduler: DDPMScheduler = DDPMScheduler.from_pretrained(
            config.model_repo,
            subfolder="scheduler"
        )
    except Exception as e:
        # create a noise scheduler
        accelerator.print(f"Error: {e}, created a new one with default config.")
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=1000,
        )

    model = DiffusionModel(unet, noise_scheduler)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
        eps=1e-8,
    )
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=(config.max_epoch * len(train_loader)),
    )
    model, optimizer, train_loader, val_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, lr_scheduler
    )

    # init logging process
    api = HfApi()
    if accelerator.is_main_process:
        api.create_repo(
            repo_id=config.remote_repo_id,
            private=False,
            exist_ok=True,
        )
        accelerator.print("Repo was successfully created")
        # Upload the auto-saved config file
        if os.path.exists(config.config_save_dir):
            api.upload_file(
                path_or_fileobj=config.config_save_dir,
                path_in_repo="train_configs.json",
                repo_id=config.remote_repo_id,
            )
        wandb.init(
            project="diffusion-model-dogs", # "diffusion-cifar10" for cifar10
            name="initial run",
            config=config.__dict__,
        )

    for epoch in range(config.max_epoch):
        avg_loss = train_step(
            model=model,
            config=config,
            accelerator=accelerator,
            train_loader=train_loader,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            current_epoch=epoch,
        )
        val_loss = validate_step(
            model=model,
            config=config,
            accelerator=accelerator,
            val_loader=val_loader,
            current_epoch=epoch,
        )
        # ============= logging =================
        if accelerator.is_main_process:
            wandb.log({
                "train/epoch": epoch,
                "train/lr": lr_scheduler.get_last_lr()[0],
                "train/loss": avg_loss,
                "validation/loss": val_loss,
            })

        if epoch % config.checkpoint_epoch == 0:
            save_path, _ = generate_images(
                model=model,
                config=config,
                accelerator=accelerator,
                save_dir=config.validation_save_dir,
                epoch=epoch,
                num_samples=16,

            )
            # before saving checkpoints, need to wait for all processes to finish generating images.
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                # log the image to wandb
                wandb.log({
                    "validation/samples": wandb.Image(save_path),
                })
                accelerator.save_model(
                    model,
                    save_directory=os.path.join(
                        config.checkpoint_save_dir, f"checkpoint_{epoch}.pt"
                    )
                )

    accelerator.end_training()  # ensures all processes synchronize before the main process handles
    if accelerator.is_main_process:
        wandb.finish()
        # save the final model to Hugging Face Hub
        accelerator.save_model(
            model,
            save_directory=os.path.join(config.checkpoint_save_dir, "final_model.pt")
        )
        # since hf-mirror.com don't support uploading, not upload, only save model
        # api.upload_file(
        #     path_or_fileobj=os.path.join(config.checkpoint_save_dir, "final_model.pt"),
        #     path_in_repo="final_model.pt",
        #     repo_id=config.remote_repo_id,
        # )
        accelerator.print(f"Final model was successfully saved to {config.checkpoint_save_dir}.")

if __name__ == "__main__":
    main()