from __future__ import annotations

import argparse
from pathlib import Path

import gradio as gr
import torch
from diffusers import DiffusionPipeline

from deco_diffusers import (
    DeCoFlowMatchEulerDiscreteScheduler,
    DeCoPipeline,
    load_transformer_from_legacy_lightning_checkpoint,
)


def _resolve_custom_pipeline_path(model_path: str) -> str:
    local_model_path = Path(model_path)
    bundled_pipeline = local_model_path / "pipeline.py"
    if bundled_pipeline.exists():
        return str(bundled_pipeline)
    repo_root = Path(__file__).resolve().parent
    return str(repo_root / "deco_diffusers" / "pipelines" / "deco" / "pipeline_deco.py")


def _load_pipeline(pretrained_model_path: str | None, legacy_ckpt_path: str | None, num_classes: int | None = None) -> DiffusionPipeline:
    if pretrained_model_path is not None:
        custom_pipeline = _resolve_custom_pipeline_path(pretrained_model_path)
        return DiffusionPipeline.from_pretrained(pretrained_model_path, custom_pipeline=custom_pipeline)

    if legacy_ckpt_path is None:
        raise ValueError("Either --pretrained-model-path or --legacy-ckpt-path must be provided")

    if num_classes is None:
        num_classes = 1000

    transformer = load_transformer_from_legacy_lightning_checkpoint(
        legacy_ckpt_path,
        conditioning_type="class",
        num_classes=num_classes,
        in_channels=3,
    )
    scheduler = DeCoFlowMatchEulerDiscreteScheduler()
    return DeCoPipeline(transformer=transformer, scheduler=scheduler)


def main():
    parser = argparse.ArgumentParser(description="DeCo diffusers-native Gradio demo")
    parser.add_argument("--pretrained-model-path", type=str, default=None)
    parser.add_argument("--legacy-ckpt-path", type=str, default=None)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--server-name", type=str, default="0.0.0.0")
    parser.add_argument("--server-port", type=int, default=23231)
    args = parser.parse_args()

    pipe = _load_pipeline(
        pretrained_model_path=args.pretrained_model_path,
        legacy_ckpt_path=args.legacy_ckpt_path,
        num_classes=args.num_classes,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = pipe.to(device)

    def generate(class_label: int, batch_size: int, seed: int, steps: int, guidance: float, height: int, width: int):
        generator = torch.Generator(device=device).manual_seed(int(seed))
        class_labels = torch.full((batch_size,), int(class_label), device=device, dtype=torch.long)

        images = pipe(
            batch_size=batch_size,
            class_labels=class_labels,
            num_inference_steps=int(steps),
            guidance_scale=float(guidance),
            height=int(height),
            width=int(width),
            generator=generator,
            output_type="pil",
        ).images
        return images

    with gr.Blocks() as demo:
        gr.Markdown("## DeCo Diffusers Demo (Class-to-Image)")
        with gr.Row():
            with gr.Column(scale=1):
                class_label = gr.Number(label="Class label", value=0, precision=0)
                batch_size = gr.Slider(minimum=1, maximum=8, value=4, step=1, label="Batch size")
                seed = gr.Slider(minimum=0, maximum=1_000_000, value=42, step=1, label="Seed")
                steps = gr.Slider(minimum=1, maximum=100, value=50, step=1, label="Inference steps")
                guidance = gr.Slider(minimum=1.0, maximum=10.0, value=4.0, step=0.1, label="Guidance scale")
                height = gr.Slider(minimum=128, maximum=1024, value=256, step=32, label="Height")
                width = gr.Slider(minimum=128, maximum=1024, value=256, step=32, label="Width")
                run_button = gr.Button("Generate")
            with gr.Column(scale=2):
                gallery = gr.Gallery(label="Generated images", columns=2, rows=2)

        run_button.click(
            fn=generate,
            inputs=[class_label, batch_size, seed, steps, guidance, height, width],
            outputs=[gallery],
        )

    demo.launch(server_name=args.server_name, server_port=args.server_port)


if __name__ == "__main__":
    main()
