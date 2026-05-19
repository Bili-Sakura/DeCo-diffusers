from __future__ import annotations

import argparse
import sys
from pathlib import Path

import gradio as gr
import torch

REPO_SRC = Path(__file__).resolve().parent / "src"
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))

from diffusers import DeCoFlowMatchEulerDiscreteScheduler, DeCoPipeline, DeCoPixelAutoencoder


def _load_pipeline(pretrained_model_path: str) -> DeCoPipeline:
    return DeCoPipeline.from_pretrained(pretrained_model_path)


def main():
    parser = argparse.ArgumentParser(description="DeCo Gradio demo (Diffusers pipeline)")
    parser.add_argument("--pretrained-model-path", type=str, required=True)
    parser.add_argument("--server-name", type=str, default="0.0.0.0")
    parser.add_argument("--server-port", type=int, default=23231)
    args = parser.parse_args()

    pipe = _load_pipeline(args.pretrained_model_path)
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
