from __future__ import annotations

import argparse

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from diffusers.models import DeCoPixelAutoencoder, DeCoTransformer2DModel
from diffusers.schedulers import DeCoFlowMatchEulerDiscreteScheduler
from .trainer import DeCoTrainConfig, DeCoTrainer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train DeCo in diffusers-native style")
    parser.add_argument("--train-data-dir", type=str, required=True, help="ImageFolder-style directory")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)

    parser.add_argument("--conditioning-type", type=str, default="class", choices=["class", "text"])
    parser.add_argument("--num-classes", type=int, default=1000)

    parser.add_argument("--in-channels", type=int, default=3)
    parser.add_argument("--patch-size", type=int, default=2)
    parser.add_argument("--num-groups", type=int, default=12)
    parser.add_argument("--hidden-size", type=int, default=1152)
    parser.add_argument("--hidden-size-x", type=int, default=64)
    parser.add_argument("--nerf-mlpratio", type=int, default=4)
    parser.add_argument("--num-blocks", type=int, default=18)
    parser.add_argument("--num-cond-blocks", type=int, default=4)

    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--max-train-steps", type=int, default=100000)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--save-every-steps", type=int, default=5000)
    parser.add_argument("--log-every-steps", type=int, default=20)
    parser.add_argument("--mixed-precision", type=str, choices=["no", "fp16", "bf16"], default="no")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def run_train_from_args(args: argparse.Namespace):
    if args.conditioning_type != "class":
        raise ValueError("This trainer currently supports class-conditioned DeCo training only")

    transform = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(args.resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    dataset = datasets.ImageFolder(root=args.train_data_dir, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    transformer = DeCoTransformer2DModel(
        conditioning_type=args.conditioning_type,
        in_channels=args.in_channels,
        patch_size=args.patch_size,
        num_groups=args.num_groups,
        hidden_size=args.hidden_size,
        hidden_size_x=args.hidden_size_x,
        nerf_mlpratio=args.nerf_mlpratio,
        num_blocks=args.num_blocks,
        num_cond_blocks=args.num_cond_blocks,
        num_classes=args.num_classes,
    )
    scheduler = DeCoFlowMatchEulerDiscreteScheduler()
    vae = DeCoPixelAutoencoder(scale=1.0, shift=0.0)

    train_config = DeCoTrainConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_train_steps=args.max_train_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        save_every_steps=args.save_every_steps,
        log_every_steps=args.log_every_steps,
        mixed_precision=args.mixed_precision,
        seed=args.seed,
    )

    trainer = DeCoTrainer(transformer=transformer, scheduler=scheduler, vae=vae, config=train_config)
    trainer.train(dataloader)


def main():
    parser = build_parser()
    args = parser.parse_args()
    run_train_from_args(args)


if __name__ == "__main__":
    main()
