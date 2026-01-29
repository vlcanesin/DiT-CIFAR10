import os
from torchvision.datasets import CIFAR10
from torchvision import transforms
from PIL import Image

def save_split(split, root):
    dataset = CIFAR10(
        root="./data",
        train=(split == "train"),
        download=True,
        transform=None
    )

    out_root = os.path.join(root, split)
    os.makedirs(out_root, exist_ok=True)

    for idx, (img, label) in enumerate(dataset):
        class_dir = os.path.join(out_root, str(label))
        os.makedirs(class_dir, exist_ok=True)
        img.save(os.path.join(class_dir, f"{idx}.png"))

if __name__ == "__main__":
    out_dir = "cifar10"
    save_split("train", out_dir)
    save_split("test", out_dir)
