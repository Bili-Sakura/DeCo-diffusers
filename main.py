from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Optional

import torch

from deco_diffusers import (
    DeCoFlowMatchEulerDiscreteScheduler,
    DeCoPipeline,
    DeCoPixelAutoencoder,
    DeCoTransformer2DModel,
    load_transformer_from_legacy_lightning_checkpoint,
)
from diffusers.training.cli import run_train_from_args as run_diffusers_train


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DeCo diffusers-native entrypoint")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train a class-conditioned DeCo model")
    train_parser.add_argument("--train-data-dir", type=str, required=True)
    train_parser.add_argument("--output-dir", type=str, required=True)
    train_parser.add_argument("--resolution", type=int, default=256)
    train_parser.add_argument("--batch-size", type=int, default=8)
    train_parser.add_argument("--num-workers", type=int, default=4)
    train_parser.add_argument("--conditioning-type", type=str, default="class", choices=["class", "text"])
    train_parser.add_argument("--num-classes", type=int, default=1000)
    train_parser.add_argument("--in-channels", type=int, default=3)
    train_parser.add_argument("--patch-size", type=int, default=2)
    train_parser.add_argument("--num-groups", type=int, default=12)
    train_parser.add_argument("--hidden-size", type=int, default=1152)
    train_parser.add_argument("--hidden-size-x", type=int, default=64)
    train_parser.add_argument("--nerf-mlpratio", type=int, default=4)
    train_parser.add_argument("--num-blocks", type=int, default=18)
    train_parser.add_argument("--num-cond-blocks", type=int, default=4)
    train_parser.add_argument("--learning-rate", type=float, default=1e-4)
    train_parser.add_argument("--weight-decay", type=float, default=1e-2)
    train_parser.add_argument("--max-train-steps", type=int, default=100000)
    train_parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    train_parser.add_argument("--max-grad-norm", type=float, default=1.0)
    train_parser.add_argument("--save-every-steps", type=int, default=5000)
    train_parser.add_argument("--log-every-steps", type=int, default=20)
    train_parser.add_argument("--mixed-precision", type=str, choices=["no", "fp16", "bf16"], default="no")
    train_parser.add_argument("--seed", type=int, default=42)

    sample_parser = subparsers.add_parser("sample", help="Generate images with DeCoPipeline")
    sample_source = sample_parser.add_mutually_exclusive_group(required=True)
    sample_source.add_argument("--pretrained-model-path", type=str)
    sample_source.add_argument("--legacy-ckpt-path", type=str)
    sample_parser.add_argument("--conditioning-type", type=str, default="class", choices=["class", "text"])
    sample_parser.add_argument("--num-classes", type=int, default=1000)
    sample_parser.add_argument("--in-channels", type=int, default=3)
    sample_parser.add_argument("--class-label", type=int, default=0)
    sample_parser.add_argument("--batch-size", type=int, default=1)
    sample_parser.add_argument("--height", type=int, default=256)
    sample_parser.add_argument("--width", type=int, default=256)
    sample_parser.add_argument("--num-inference-steps", type=int, default=50)
    sample_parser.add_argument("--guidance-scale", type=float, default=1.0)
    sample_parser.add_argument("--seed", type=int, default=42)
    sample_parser.add_argument("--output-dir", type=str, required=True)
    sample_parser.add_argument("--prompt-embeds-path", type=str, default=None)
    sample_parser.add_argument("--negative-prompt-embeds-path", type=str, default=None)

    return parser


def _build_pipeline_from_legacy_ckpt(
    ckpt_path: str,
    conditioning_type: str,
    num_classes: int,
    in_channels: int,
) -> DeCoPipeline:
    transformer = load_transformer_from_legacy_lightning_checkpoint(
        ckpt_path,
        conditioning_type=conditioning_type,
        num_classes=num_classes,
        in_channels=in_channels,
    )
    scheduler = DeCoFlowMatchEulerDiscreteScheduler()
    vae = DeCoPixelAutoencoder(scale=1.0, shift=0.0)
    return DeCoPipeline(transformer=transformer, scheduler=scheduler, vae=vae)


def _sample(args: argparse.Namespace):
    if args.pretrained_model_path is not None:
        pipe = DeCoPipeline.from_pretrained(args.pretrained_model_path)
    else:
        pipe = _build_pipeline_from_legacy_ckpt(
            ckpt_path=args.legacy_ckpt_path,
            conditioning_type=args.conditioning_type,
            num_classes=args.num_classes,
            in_channels=args.in_channels,
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = pipe.to(device)

    generator = torch.Generator(device=device).manual_seed(args.seed)

    kwargs: dict[str, Any] = {
        "batch_size": args.batch_size,
        "height": args.height,
        "width": args.width,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "generator": generator,
        "output_type": "pil",
    }

    if pipe.transformer.config.conditioning_type == "class":
        kwargs["class_labels"] = torch.full((args.batch_size,), args.class_label, dtype=torch.long, device=device)
    else:
        if args.prompt_embeds_path is None:
            raise ValueError("--prompt-embeds-path is required for text-conditioned sampling")
        prompt_embeds = torch.load(args.prompt_embeds_path, map_location=device)
        if not isinstance(prompt_embeds, torch.Tensor):
            raise ValueError("prompt embeddings file must contain a torch.Tensor")
        kwargs["prompt_embeds"] = prompt_embeds

        if args.negative_prompt_embeds_path is not None:
            negative_prompt_embeds = torch.load(args.negative_prompt_embeds_path, map_location=device)
            if not isinstance(negative_prompt_embeds, torch.Tensor):
                raise ValueError("negative prompt embeddings file must contain a torch.Tensor")
            kwargs["negative_prompt_embeds"] = negative_prompt_embeds

    images = pipe(**kwargs).images

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for idx, image in enumerate(images):
        image.save(output_dir / f"sample_{idx:03d}.png")

    print(f"Saved {len(images)} images to {output_dir}")


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "train":
        run_diffusers_train(args)
    elif args.command == "sample":
        _sample(args)


if __name__ == "__main__":
    main()
