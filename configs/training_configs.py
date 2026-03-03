import torch
import json
import os
from datasets import load_dataset
from torchvision import transforms
from dataclasses import dataclass, field, asdict
from torch.utils.data import DataLoader

@dataclass
class TrainConfigs:
    max_epoch: int = field(default=350)  # 300 - 400  for real training
    dataset_name: str = field(default="uoft-cs/cifar10")  # use uoft-cs/cifar10 for cifar10 data training
    model_repo: str = field(default="google/ddpm-cifar10-32")  # use google/ddpm-cifar10-32 for cifar10 data training
    image_size: int = field(default=32)  # generated image size (use low-precision)
    image_field: str = field(default="img")  # field name for image in the dataset
    lr: float = field(default=1e-4)
    lr_warmup_steps: int = field(default=500)  # learning rate warmup steps
    weight_decay: float = field(default=1e-5)
    train_batch_size: int = field(default=64)
    eval_batch_size: int = field(default=64)
    gradient_accumulation_steps: int = field(default=1)  # for larger effective batch size
    remote_repo_id: str = field(default="FriedParrot/ddpm-cifar10-diffusion")
    checkpoint_save_dir: str = field(default="./checkpoints")
    validation_save_dir: str = field(default="./validation_samples")
    checkpoint_epoch: int = field(default=10)
    overwrite_output_dir: bool = field(default=True)
    reverse_diffusion_steps: int = field(default=100)  # number of steps for reverse diffusion during validation
    config_save_dir: str = field(default="./training_configs.json")

    def __post_init__(self):
        """Automatically save config after initialization"""
        self.save()

    def save(self, path: str = None):
        """
        Save the configuration to a JSON file.

        :param path: Optional custom path to save to. If None, uses config_save_dir
        """
        save_path = path if path is not None else self.config_save_dir
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)

        with open(save_path, 'w') as f:
            json.dump(asdict(self), f, indent=2)

        return save_path

    @classmethod
    def load(cls, path: str):
        """
        Load configuration from a JSON file.

        :param path: Path to the JSON config file
        :return: TrainConfigs instance
        """
        with open(path, 'r') as f:
            config_dict = json.load(f)

        return cls(**config_dict)


def load_dataloaders(
    config: TrainConfigs,
    auto_split: bool = True,
    train_size: float = 0.8,
):
    """
    A helper function to load and preprocess the dataset,
    returning PyTorch DataLoaders for training and validation.

    :param config: TrainConfigs instance with dataset and training parameters
    :param auto_split: Whether to automatically split the training set if no test set exists
    :param train_size: Fraction of data to use for training when auto_split is True (default: 0.8)
    :return: Tuple of (train_loader, val_loader)
    """
    all_set = load_dataset(config.dataset_name)

    # Determine train/validation split
    if "test" in all_set.keys():
        train_set = all_set["train"]
        val_set = all_set["test"]
    else:
        if auto_split:
            # Use train_size parameter for dynamic splitting
            full_train = all_set["train"]
            total_samples = len(full_train)
            train_samples = int(total_samples * train_size)

            shuffled_dataset = full_train.shuffle(seed=42)
            train_set = shuffled_dataset.select(range(train_samples))
            val_set = shuffled_dataset.select(range(train_samples, total_samples))
            print("auto splitted train/val sets with train_size =", train_size)
        else:
            train_set = all_set["train"]
            val_set = all_set["train"]

    preprocess = transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    # Custom preprocess function that handles multi-field datasets
    def transform(batch):
        """Apply transforms to each image in the batch"""
        batch[config.image_field] = [preprocess(img) for img in batch[config.image_field]]
        return batch

    train_set.set_transform(transform)
    val_set.set_transform(transform)

    train_loader = DataLoader(train_set, batch_size=config.train_batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=config.eval_batch_size, shuffle=False)

    # Validate preprocessed images are in correct range
    test_batch = torch.stack(list(train_set[:config.eval_batch_size][config.image_field]))
    max_value = test_batch.max()
    min_value = test_batch.min()
    assert max_value <= 1.0 and min_value >= -1.0, \
        f"Preprocessed images should be in [-1, 1], but got max {max_value} and min {min_value}"

    return train_loader, val_loader