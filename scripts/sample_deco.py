#!/usr/bin/env python3
# Copyright 2026 The HuggingFace Team. All rights reserved.

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from deco_diffusers import DeCoPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample images with a Diffusers-native DeCo pipeline.")
    parser.add_argument("--model", required=True, help="Path to a converted Diffusers model directory.")
    parser.add_argument("--class-label", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--prompt-embeds-path", type=str, default=None)
    parser.add_argument("--negative-prompt-embeds-path", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = DeCoPipeline.from_pretrained(args.model).to(device)
    generator = torch.Generator(device=device).manual_seed(args.seed)

    kwargs = {
        "batch_size": args.batch_size,
        "height": args.height,
        "width": args.width,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "generator": generator,
        "output_type": "pil",
    }

    if pipe.transformer.config.conditioning_type == "class":
        kwargs["class_labels"] = torch.full((args.batch_size,), args.class_label, device=device, dtype=torch.long)
    else:
        if args.prompt_embeds_path is None:
            raise ValueError("--prompt-embeds-path is required for text-conditioned models")
        kwargs["prompt_embeds"] = torch.load(args.prompt_embeds_path, map_location=device)
        if args.negative_prompt_embeds_path is not None:
            kwargs["negative_prompt_embeds"] = torch.load(args.negative_prompt_embeds_path, map_location=device)

    images = pipe(**kwargs).images
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, image in enumerate(images):
        image.save(output_dir / f"sample_{index:03d}.png")
    print(f"Saved {len(images)} images to {output_dir}")


if __name__ == "__main__":
    main()
