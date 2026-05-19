#!/usr/bin/env python3
# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Convert original DeCo Lightning checkpoints into a Diffusers pipeline directory."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

try:
    from safetensors.torch import load_file as safe_load_file
    from safetensors.torch import save_file as safe_save_file
except Exception:  # pragma: no cover
    safe_load_file = None
    safe_save_file = None

from deco_diffusers.models import DeCoPixelAutoencoder, DeCoTransformer2DModel
from deco_diffusers.pipelines import DeCoPipeline
from deco_diffusers.schedulers import DeCoFlowMatchEulerDiscreteScheduler


MODEL_PRESETS: dict[str, dict[str, Any]] = {
    "deco-xl-c2i": {
        "conditioning_type": "class",
        "in_channels": 3,
        "patch_size": 2,
        "num_groups": 12,
        "hidden_size": 1152,
        "hidden_size_x": 64,
        "num_blocks": 18,
        "num_cond_blocks": 4,
        "num_classes": 1000,
    },
    "deco-xxl-t2i": {
        "conditioning_type": "text",
        "in_channels": 3,
        "patch_size": 2,
        "num_groups": 12,
        "hidden_size": 1152,
        "decoder_hidden_size": 64,
        "num_encoder_blocks": 18,
        "num_decoder_blocks": 4,
        "num_text_blocks": 4,
        "txt_embed_dim": 1024,
        "txt_max_length": 100,
    },
}


def _load_lightning_state_dict(checkpoint_path: str, use_ema: bool) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    prefix = "ema_denoiser." if use_ema else "denoiser."
    mapped = {key[len(prefix) :]: value for key, value in state_dict.items() if key.startswith(prefix)}
    if len(mapped) == 0:
        raise ValueError(f"No parameters found in checkpoint with prefix '{prefix}'")
    return mapped


def _save_config(output_dir: Path, config: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "config.json", "w", encoding="utf-8") as file:
        json.dump(config, file, indent=2, sort_keys=True)
        file.write("\n")


def _save_weights(output_dir: Path, state_dict: dict[str, torch.Tensor], safe_serialization: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if safe_serialization:
        if safe_save_file is None:
            raise ImportError("Install safetensors or pass --no-safe-serialization.")
        safe_save_file(state_dict, str(output_dir / "diffusion_pytorch_model.safetensors"), metadata={"format": "pt"})
    else:
        torch.save(state_dict, output_dir / "diffusion_pytorch_model.bin")


def _write_model_index(output_dir: Path) -> None:
    model_index = {
        "_class_name": "DeCoPipeline",
        "_diffusers_version": "0.30.1",
        "scheduler": ["diffusers", "DeCoFlowMatchEulerDiscreteScheduler"],
        "transformer": ["diffusers", "DeCoTransformer2DModel"],
        "vae": ["diffusers", "DeCoPixelAutoencoder"],
    }
    with open(output_dir / "model_index.json", "w", encoding="utf-8") as file:
        json.dump(model_index, file, indent=2, sort_keys=True)
        file.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert original DeCo Lightning checkpoints to Diffusers layout.")
    parser.add_argument("--checkpoint", required=True, help="Path to a DeCo Lightning .ckpt checkpoint.")
    parser.add_argument("--output", required=True, help="Output Diffusers model directory.")
    parser.add_argument("--model-size", choices=sorted(MODEL_PRESETS), default="deco-xl-c2i")
    parser.add_argument("--use-ema", action="store_true", help="Load EMA weights (ema_denoiser.* prefix).")
    parser.add_argument("--shift", type=float, default=1.0)
    parser.add_argument("--last-step", type=float, default=None)
    parser.add_argument("--safe-serialization", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--check-load", action="store_true", help="Instantiate pipeline and verify weight loading.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output)
    preset = MODEL_PRESETS[args.model_size]

    transformer = DeCoTransformer2DModel(**preset)
    state_dict = _load_lightning_state_dict(args.checkpoint, use_ema=args.use_ema)
    incompatible = transformer.backbone.load_state_dict(state_dict, strict=False)
    if incompatible.missing_keys:
        print(f"Missing keys: {len(incompatible.missing_keys)}")
    if incompatible.unexpected_keys:
        print(f"Unexpected keys: {len(incompatible.unexpected_keys)}")

    scheduler = DeCoFlowMatchEulerDiscreteScheduler(shift=args.shift, last_step=args.last_step)
    vae = DeCoPixelAutoencoder(scale=1.0, shift=0.0)

    transformer_dir = output_dir / "transformer"
    scheduler_dir = output_dir / "scheduler"
    vae_dir = output_dir / "vae"

    _save_config(transformer_dir, transformer.config)
    _save_weights(transformer_dir, transformer.state_dict(), args.safe_serialization)
    _save_config(scheduler_dir, scheduler.config)
    _save_config(vae_dir, vae.config)
    _save_weights(vae_dir, vae.state_dict(), args.safe_serialization)
    _write_model_index(output_dir)

    if args.check_load:
        pipe = DeCoPipeline.from_pretrained(output_dir)
        print(f"Loaded pipeline with conditioning_type={pipe.transformer.config.conditioning_type}")

    print(f"Wrote Diffusers pipeline to {output_dir}")


if __name__ == "__main__":
    main()
