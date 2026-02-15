# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
A minimal training script for DiT using PyTorch DDP
(with resume-from-latest-checkpoint support).
"""

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms

import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os

from models import DiT_models
from diffusion import create_diffusion

#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    dist.destroy_process_group()


def create_logger(logging_dir):
    if dist.get_rank() == 0:
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(f"{logging_dir}/log.txt") if logging_dir else logging.NullHandler()
            ],
        )
        logger = logging.getLogger(__name__)
    else:
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def center_crop_arr(pil_image, image_size):
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    assert torch.cuda.is_available(), "Training requires at least one GPU."

    # ------------------ DDP setup ------------------
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = rank % torch.cuda.device_count()
    torch.cuda.set_device(device)

    seed = args.global_seed * world_size + rank
    torch.manual_seed(seed)

    if rank == 0:
        print(f"Starting rank={rank}, seed={seed}, world_size={world_size}")

    # ------------------ Experiment dirs ------------------
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_name = args.model.replace("/", "-")
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_name}"
        checkpoint_dir = f"{experiment_dir}/checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
    else:
        experiment_dir = None
        checkpoint_dir = None

    # Broadcast paths
    experiment_dir = experiment_dir if rank == 0 else ""
    checkpoint_dir = checkpoint_dir if rank == 0 else ""
    experiment_dir = broadcast_string(experiment_dir)
    checkpoint_dir = broadcast_string(checkpoint_dir)

    logger = create_logger(experiment_dir)

    if rank == 0:
        logger.info(f"Experiment directory created at {experiment_dir}")

    # ------------------ Model ------------------
    model = DiT_models[args.model](
        input_size=args.image_size,
        num_classes=args.num_classes
    )

    ema = deepcopy(model).to(device)
    requires_grad(ema, False)

    model = DDP(model.to(device), device_ids=[device])
    diffusion = create_diffusion()

    logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)

    # ------------------ Resume logic ------------------
    train_steps = 0

    if args.resume:
        if rank == 0:
            ckpts = sorted(glob(f"{checkpoint_dir}/*.pt"))
            if not ckpts:
                raise RuntimeError("Resume flag set but no checkpoints found.")
            latest_ckpt = ckpts[-1]
            logger.info(f"Resuming from checkpoint: {latest_ckpt}")
            checkpoint = torch.load(latest_ckpt, map_location="cpu")

            model.module.load_state_dict(checkpoint["model"])
            ema.load_state_dict(checkpoint["ema"])
            opt.load_state_dict(checkpoint["opt"])
            train_steps = int(os.path.basename(latest_ckpt).split(".")[0])
        else:
            checkpoint = None

        # Sync model + optimizer across ranks
        dist.barrier()
        for p in model.parameters():
            dist.broadcast(p.data, src=0)
        for p in ema.parameters():
            dist.broadcast(p.data, src=0)

    # ------------------ Data ------------------
    transform = transforms.Compose([
        transforms.Lambda(lambda img: center_crop_arr(img, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3, inplace=True)
    ])

    dataset = ImageFolder(args.data_path, transform=transform)
    sampler = DistributedSampler(dataset, world_size, rank, seed=args.global_seed)

    loader = DataLoader(
        dataset,
        batch_size=args.global_batch_size // world_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    logger.info(f"Dataset contains {len(dataset):,} images")

    # ------------------ Training ------------------
    model.train()
    ema.eval()

    running_loss = 0
    log_steps = 0
    start_time = time()

    logger.info(f"Training for {args.epochs} epochs...")

    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")

        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            loss = diffusion.training_losses(model, x, t, dict(y=y))["loss"].mean()

            opt.zero_grad()
            loss.backward()
            opt.step()
            update_ema(ema, model.module)

            running_loss += loss.item()
            log_steps += 1
            train_steps += 1

            if train_steps % args.log_every == 0:
                torch.cuda.synchronize()
                elapsed = time() - start_time
                steps_per_sec = log_steps / elapsed

                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss)
                avg_loss /= world_size

                logger.info(
                    f"(step={train_steps:07d}) "
                    f"Train Loss: {avg_loss:.4f}, "
                    f"Train Steps/Sec: {steps_per_sec:.2f}"
                )

                running_loss = 0
                log_steps = 0
                start_time = time()

            if train_steps % args.ckpt_every == 0 and rank == 0:
                ckpt = {
                    "model": model.module.state_dict(),
                    "ema": ema.state_dict(),
                    "opt": opt.state_dict(),
                    "args": args,
                }
                path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                torch.save(ckpt, path)
                logger.info(f"Saved checkpoint to {path}")

    logger.info("Done!")
    cleanup()


def broadcast_string(s):
    """Broadcast a Python string from rank 0 to all ranks."""
    if dist.get_rank() == 0:
        data = bytearray(s.encode())
        length = torch.tensor([len(data)], device="cuda")
    else:
        data = bytearray()
        length = torch.tensor([0], device="cuda")

    dist.broadcast(length, 0)
    if dist.get_rank() != 0:
        data = bytearray(length.item())

    tensor = torch.ByteTensor(list(data)).cuda()
    dist.broadcast(tensor, 0)
    return tensor.cpu().numpy().tobytes().decode()


#################################################################################
#                                   Args                                       #
#################################################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--image-size", type=int, choices=[32, 256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=50_000)
    parser.add_argument("--resume", action="store_true")

    args = parser.parse_args()
    main(args)
