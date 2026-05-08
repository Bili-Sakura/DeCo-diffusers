#!/usr/bin/env python3

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from deco_diffusers import (
    DeCoClassPipeline,
    DeCoFlowMatchEulerDiscreteScheduler,
    DeCoTextPipeline,
    load_transformer_from_legacy_lightning_checkpoint,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert a legacy DeCo checkpoint to Diffusers format.")
    parser.add_argument("--checkpoint", required=True, help="Path to the legacy Lightning checkpoint.")
    parser.add_argument("--output-dir", required=True, help="Output directory for the Diffusers checkpoint.")
    parser.add_argument("--conditioning-type", choices=["class", "text"], required=True)
    parser.add_argument("--use-ema", action="store_true", help="Load EMA weights from the checkpoint.")
    parser.add_argument("--strict", action="store_true", help="Enable strict state dict loading.")

    parser.add_argument("--in-channels", type=int, default=3)
    parser.add_argument("--patch-size", type=int, default=2)
    parser.add_argument("--num-groups", type=int, default=12)
    parser.add_argument("--hidden-size", type=int, default=1152)
    parser.add_argument("--hidden-size-x", type=int, default=64)
    parser.add_argument("--nerf-mlpratio", type=int, default=4)
    parser.add_argument("--num-blocks", type=int, default=18)
    parser.add_argument("--num-cond-blocks", type=int, default=4)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--no-learn-sigma", action="store_true", help="Disable learn_sigma in the transformer config.")
    parser.add_argument("--deep-supervision", type=int, default=0)

    parser.add_argument("--decoder-hidden-size", type=int, default=64)
    parser.add_argument("--num-encoder-blocks", type=int, default=18)
    parser.add_argument("--num-decoder-blocks", type=int, default=4)
    parser.add_argument("--num-text-blocks", type=int, default=4)
    parser.add_argument("--txt-embed-dim", type=int, default=1024)
    parser.add_argument("--txt-max-length", type=int, default=100)

    parser.add_argument(
        "--skip-pipeline-copy",
        action="store_true",
        help="Skip copying the pipeline.py file into the output directory.",
    )
    return parser


def resolve_pipeline_path(conditioning_type: str) -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    if conditioning_type == "class":
        return repo_root / "deco_diffusers" / "pipelines" / "deco" / "pipeline_deco_class.py"
    return repo_root / "deco_diffusers" / "pipelines" / "deco" / "pipeline_deco_text.py"


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    transformer = load_transformer_from_legacy_lightning_checkpoint(
        args.checkpoint,
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
        learn_sigma=not args.no_learn_sigma,
        deep_supervision=args.deep_supervision,
        decoder_hidden_size=args.decoder_hidden_size,
        num_encoder_blocks=args.num_encoder_blocks,
        num_decoder_blocks=args.num_decoder_blocks,
        num_text_blocks=args.num_text_blocks,
        txt_embed_dim=args.txt_embed_dim,
        txt_max_length=args.txt_max_length,
        use_ema=args.use_ema,
        strict=args.strict,
    )
    scheduler = DeCoFlowMatchEulerDiscreteScheduler()

    pipeline_cls = DeCoClassPipeline if args.conditioning_type == "class" else DeCoTextPipeline
    pipe = pipeline_cls(transformer=transformer, scheduler=scheduler)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pipe.save_pretrained(output_dir)

    if not args.skip_pipeline_copy:
        shutil.copyfile(resolve_pipeline_path(args.conditioning_type), output_dir / "pipeline.py")

    print(f"Saved Diffusers checkpoint to {output_dir}")


if __name__ == "__main__":
    main()
