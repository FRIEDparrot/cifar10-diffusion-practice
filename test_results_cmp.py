import json
import time

from datasets import load_dataset
from accelerate import Accelerator
from unconditional import UNet2DModel, DiffusionModel, DDPMScheduler
from unconditional import test_denoising_results, transform, generate_images
from torch.utils.data import DataLoader

model_repo = "google/ddpm-cifar10-32"
unet = UNet2DModel.from_pretrained(model_repo)

scheduler_config = json.load(open("ddpm_scheduler_cfg.json"))
scheduler:DDPMScheduler = DDPMScheduler.from_config(scheduler_config)
model = DiffusionModel(unet=unet, scheduler=scheduler)
dataset_link = "uoft-cs/cifar10"
dataset = load_dataset(dataset_link, split="train").with_format("torch")
dataset.set_transform(transform)

dataloader = DataLoader(dataset, batch_size=8)
accelerator = Accelerator(device_placement=True)

model, scheduler, dataloader = accelerator.prepare(model, scheduler, dataloader)
batch = next(iter(dataloader))


t2 = time.time()
generate_images(model=model, save_dir="./test_evaluate", epoch=0, num_samples=16, device=accelerator.device)
print(f"Time taken for test_evaluate: {time.time() - t2:.2f} seconds")

t = time.time()
test_denoising_results(
    model=model,
    scheduler=scheduler,
    accelerator=accelerator,
    batch=batch,
    save_path="./test_result.png",
    test_num=4,
)
print(f"Time taken for test_denoising_results: {time.time() - t:.2f} seconds")
