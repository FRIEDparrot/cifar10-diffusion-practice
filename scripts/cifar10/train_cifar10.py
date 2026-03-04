import os
import wandb
from diffusers import UNet2DModel, DDPMScheduler
from accelerate import Accelerator

from models import DiffusionModel
from configs import TrainConfigs, load_dataloaders
from utils import train_step, validate_step, generate_images, setup_training

os.chdir(os.path.dirname(__file__))  # change to current directory for saving configs and checkpoints


def main():
    """
    This script not do anything about api, so remote_repo_id takes no use,
    But it still should be set.
    :return:
    """
    config = TrainConfigs(
        max_epoch=50,
        checkpoint_epoch=5,
        model_repo="google/ddpm-cifar10-32",
        dataset_name="uoft-cs/cifar10",
        image_size=32,
        image_field="img",
        gradient_accumulation_steps=2,
        train_batch_size=64,
        eval_batch_size=64,
        lr_warmup_steps=500,
        remote_repo_id="FriedParrot/ddpm-cifar10-diffusion",
        reverse_diffusion_steps=100,
        lr=2e-5,
    )
    train_loader, val_loader = load_dataloaders(config)
    # Load pretrained model without any weights
    unet = UNet2DModel.from_config(
        UNet2DModel.load_config(config.model_repo)
    )
    noise_scheduler = DDPMScheduler.from_pretrained(config.model_repo)
    model = DiffusionModel(unet=unet, noise_scheduler=noise_scheduler)

    # Setup accelerator
    accelerator = Accelerator(
        device_placement=True,
        mixed_precision="fp16",
        gradient_accumulation_steps=config.gradient_accumulation_steps
    )

    # Setup training (creates optimizer, lr_scheduler, and prepares all components)
    model, optimizer, lr_scheduler, train_loader, val_loader = setup_training(
        config, model, accelerator, train_loader, val_loader
    )
    if accelerator.is_main_process:
        wandb.init(
            project="ddpm-cifar10-diffusion",
            name="first training run",
            dir="./wandb_logs",
            config=config.__dict__,
        )

    # Training loop
    for epoch in range(config.max_epoch):
        avg_loss = train_step(
            model, config, accelerator, train_loader,
            optimizer, lr_scheduler, current_epoch=epoch
        )
        val_loss = validate_step(
            model, config, accelerator,
            val_loader, current_epoch=epoch
        )

        # Print progress
        if accelerator.is_main_process:
            wandb.log({
                "epoch": epoch + 1,
                "train/epoch-loss": avg_loss,
                "val/epoch-loss": val_loss,
                "train/lr": lr_scheduler.get_last_lr()[0],
            }, step=epoch)

        # Generate validation samples periodically
        if (epoch + 1) % config.checkpoint_epoch == 0:
            save_path, _ = generate_images(
                model, config, accelerator,
                config.validation_save_dir, epoch, num_samples=16
            )
            if accelerator.is_main_process:
                # not save model, just log the image
                wandb.log({
                    "val/samples": wandb.Image(save_path),
                }, step=epoch)

    # Finalize training
    accelerator.end_training()
    if accelerator.is_main_process:
        wandb.finish()
        # save the model
        accelerator.save_model(
            model,
            save_directory=os.path.join(config.checkpoint_save_dir, "final_model.pt"),
        )
        accelerator.print(f"Final model was successfully saved to {config.checkpoint_save_dir}.")

if __name__ == "__main__":
    main()
