#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from diffusers import DiffusionPipeline


def resolve_custom_pipeline_path(model_path: str) -> str:
    local_model_path = Path(model_path)
    bundled_pipeline = local_model_path / "pipeline.py"
    if bundled_pipeline.exists():
        return str(bundled_pipeline)
    repo_root = Path(__file__).resolve().parents[1]
    return str(repo_root / "deco_diffusers" / "pipelines" / "deco" / "pipeline_deco.py")


def load_prompt_embeds(path: str, device: str, dtype: torch.dtype) -> torch.Tensor:
    prompt_embeds = torch.load(path, map_location=device)
    if not isinstance(prompt_embeds, torch.Tensor):
        raise ValueError("Prompt embeddings file must contain a torch.Tensor")
    return prompt_embeds.to(device=device, dtype=dtype)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample images with the DeCo Diffusers pipeline.")
    parser.add_argument("--model", required=True, help="Path or Hub id of a Diffusers-style DeCo pipeline.")
    parser.add_argument(
        "--class-label",
        type=int,
        action="append",
        default=None,
        help="Class id. Repeat the flag (e.g. --class-label 1 --class-label 2) to build a batch.",
    )
    parser.add_argument("--prompt-embeds-path", type=str, default=None, help="Path to a torch.Tensor of prompt embeds.")
    parser.add_argument(
        "--negative-prompt-embeds-path",
        type=str,
        default=None,
        help="Optional path to a torch.Tensor of negative prompt embeds.",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=4.0)
    parser.add_argument("--torch-dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", default="samples")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[args.torch_dtype]

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    custom_pipeline = resolve_custom_pipeline_path(args.model)
    pipe = DiffusionPipeline.from_pretrained(
        args.model,
        custom_pipeline=custom_pipeline,
        torch_dtype=dtype,
    ).to(device)

    generator = torch.Generator(device=device)
    if args.seed is not None:
        generator.manual_seed(args.seed)

    kwargs: dict[str, object] = {
        "height": args.height,
        "width": args.width,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "generator": generator,
        "output_type": "pil",
    }

    if args.class_label is not None:
        class_labels = torch.tensor(args.class_label, device=device, dtype=torch.long)
        kwargs["class_labels"] = class_labels
        if args.batch_size is not None:
            kwargs["batch_size"] = args.batch_size
    elif args.prompt_embeds_path is not None:
        prompt_embeds = load_prompt_embeds(args.prompt_embeds_path, device, dtype)
        kwargs["prompt_embeds"] = prompt_embeds
        if args.negative_prompt_embeds_path is not None:
            kwargs["negative_prompt_embeds"] = load_prompt_embeds(args.negative_prompt_embeds_path, device, dtype)
        if args.batch_size is not None:
            kwargs["batch_size"] = args.batch_size
    else:
        raise ValueError("Provide either --class-label or --prompt-embeds-path for sampling.")

    output = pipe(**kwargs).images

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, image in enumerate(output):
        image.save(output_dir / f"{index:06d}.png")


if __name__ == "__main__":
    main()
