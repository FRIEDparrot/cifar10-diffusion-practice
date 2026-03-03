import torch
from datasets import load_dataset
from diffusers import DDPMScheduler
import matplotlib.pyplot as plt
from diffusers.utils.torch_utils import randn_tensor
from models.diffusion_model import forward_diffusion, transform, denormalize

model_repo = "google/ddpm-cifar10-32"
dataset = load_dataset("uoft-cs/cifar10", split="test")
scheduler:DDPMScheduler = DDPMScheduler.from_pretrained(model_repo)

max_step = 800
show_step = 100
sample_num = max_step // show_step
subset = dataset.select(range(sample_num))  # use select to return a proper subset
subset.set_transform(transform)

fig, axes = plt.subplots(2, sample_num, figsize=(sample_num * 3, 7))
fig.suptitle("Forward Diffusion Process", fontsize=16, fontweight='bold', y=1.01)

# Row labels
axes[0, 0].set_ylabel("Original", fontsize=13, fontweight='bold', labelpad=10)
axes[1, 0].set_ylabel("Noised", fontsize=13, fontweight='bold', labelpad=10)

for i in range(sample_num):
    x = subset[i]["img"]  # [C, H, W]
    # make step a torch integer tensor on the same device as the image
    step = torch.tensor(i * show_step, dtype=torch.int, device=x.device)

    noise = randn_tensor(x.shape, device=x.device)  # [B, C, H, W]
    x_t = forward_diffusion(x, step, noise, scheduler)

    # ======== plot ========
    x_img = denormalize(x).permute(1, 2, 0).cpu().numpy()
    axes[0, i].imshow(x_img)
    axes[0, i].axis("off")
    axes[0, i].set_title(f"Step {step}", fontsize=9)

    x_t_img = denormalize(x_t).permute(1, 2, 0).cpu().numpy()
    axes[1, i].imshow(x_t_img)
    axes[1, i].set_title(f"Step {step}", fontsize=9)
    axes[1, i].axis("off")
plt.tight_layout()
plt.show()

