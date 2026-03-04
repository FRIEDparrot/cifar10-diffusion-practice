import os
import wandb
from huggingface_hub import HfApi
from diffusers import UNet2DModel, DDPMScheduler
from accelerate import Accelerator

from configs import TrainConfigs, load_dataloaders
from models import DiffusionModel
from utils import train_step, validate_step, generate_images, setup_training

"""
Model : "anton-l/ddpm-butterflies-128"
Dataset :  huggan/few-shot-dog (contains 389 images of dogs, 90% for training and 10 for validation)
Train on resolution 128

originally drafted model, contains most training details. 
"""

os.chdir(os.path.dirname(__file__))  # change to current directory for saving configs and checkpoints


def main():
    # Config auto-saves to config_save_dir upon creation
    config = TrainConfigs(
        max_epoch=250,  # pretrained model
        dataset_name="huggan/few-shot-dog",
        image_size=128,
        train_batch_size=12,
        eval_batch_size=12,
        lr=1e-4,
        lr_warmup_steps=500,
        # unconditional diffusion model
        model_repo = "anton-l/ddpm-butterflies-128",
        image_field="image",
        remote_repo_id="FriedParrot/ddpm-few-shot-dog-128",
        checkpoint_epoch=5,
        reverse_diffusion_steps=500, # higher quality
    )  # all process needs it
    accelerator = Accelerator(
        device_placement=True,
        mixed_precision="fp16",
        gradient_accumulation_steps=config.gradient_accumulation_steps,
    )
    # since the dataset is very small, use higher train_size.
    train_loader, val_loader = load_dataloaders(config, auto_split=True, train_size=0.9)
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

    # Setup training (creates optimizer, lr_scheduler, and prepares all components)
    model, optimizer, lr_scheduler, train_loader, val_loader = setup_training(
        config, model, accelerator, train_loader, val_loader
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
            dir="./wandb_logs",
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
        accelerator.print(f"Final model was successfully saved to {config.checkpoint_save_dir}.")

if __name__ == "__main__":
    main()