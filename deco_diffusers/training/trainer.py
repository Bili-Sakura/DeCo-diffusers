from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from deco_diffusers.models import DeCoPixelAutoencoder, DeCoTransformer2DModel
from deco_diffusers.pipelines import DeCoPipeline
from deco_diffusers.schedulers import DeCoFlowMatchEulerDiscreteScheduler


@dataclass
class DeCoTrainConfig:
    output_dir: str
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    max_train_steps: int = 100000
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    save_every_steps: int = 5000
    log_every_steps: int = 20
    mixed_precision: str = "no"  # no | fp16 | bf16
    seed: int = 42


class DeCoTrainer:
    def __init__(
        self,
        transformer: DeCoTransformer2DModel,
        scheduler: DeCoFlowMatchEulerDiscreteScheduler,
        vae: Optional[DeCoPixelAutoencoder],
        config: DeCoTrainConfig,
    ):
        self.transformer = transformer
        self.scheduler = scheduler
        self.vae = vae
        self.config = config

        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transformer.to(self.device)
        if self.vae is not None:
            self.vae.to(self.device)
            self.vae.eval()

        self.optimizer = AdamW(
            self.transformer.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.grad_scaler = torch.amp.GradScaler("cuda", enabled=self._use_amp_scaler)

    @property
    def _use_amp_scaler(self) -> bool:
        return self.config.mixed_precision == "fp16" and self.device.type == "cuda"

    @property
    def _use_autocast(self) -> bool:
        return self._autocast_dtype is not None and self.device.type == "cuda"

    @property
    def _autocast_dtype(self) -> Optional[torch.dtype]:
        if self.config.mixed_precision == "fp16":
            return torch.float16
        if self.config.mixed_precision == "bf16":
            return torch.bfloat16
        return None

    def _compute_loss(self, pixel_values: torch.Tensor, class_labels: Optional[torch.Tensor]) -> torch.Tensor:
        pixel_values = pixel_values.to(self.device)
        if self.vae is not None:
            with torch.no_grad():
                latents = self.vae.encode(pixel_values).sample
        else:
            latents = pixel_values

        noise = torch.randn_like(latents)
        timesteps = torch.rand((latents.shape[0],), device=self.device, dtype=latents.dtype)
        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)

        if self.transformer.config.conditioning_type == "class":
            if class_labels is None:
                raise ValueError("class_labels are required for class-conditioned training")
            class_labels = class_labels.to(self.device, dtype=torch.long)
            model_pred = self.transformer(noisy_latents, timesteps, class_labels=class_labels).sample
        else:
            raise ValueError("text-conditioned training is not implemented in this lightweight trainer")

        target = latents - noise
        return F.mse_loss(model_pred.float(), target.float(), reduction="mean")

    def _save_pipeline(self, output_dir: Path, step: int):
        save_dir = output_dir / f"checkpoint-{step}"
        pipe = DeCoPipeline(transformer=self.transformer, scheduler=self.scheduler, vae=self.vae)
        pipe.save_pretrained(save_dir)

    def train(self, dataloader: DataLoader):
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self.transformer.train()
        step = 0
        running_loss = 0.0

        while step < self.config.max_train_steps:
            for batch in dataloader:
                if step >= self.config.max_train_steps:
                    break

                pixel_values = batch[0]
                class_labels = batch[1] if len(batch) > 1 else None

                with torch.autocast(
                    device_type=self.device.type,
                    dtype=self._autocast_dtype,
                    enabled=self._use_autocast,
                ):
                    loss = self._compute_loss(pixel_values=pixel_values, class_labels=class_labels)
                    scaled_loss = loss / self.config.gradient_accumulation_steps

                self.grad_scaler.scale(scaled_loss).backward()

                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    self.grad_scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.transformer.parameters(), self.config.max_grad_norm)
                    self.grad_scaler.step(self.optimizer)
                    self.grad_scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)

                running_loss += loss.detach().item()
                step += 1

                if step % self.config.log_every_steps == 0:
                    avg_loss = running_loss / self.config.log_every_steps
                    print(f"[step {step}] loss={avg_loss:.6f}")
                    running_loss = 0.0

                if step % self.config.save_every_steps == 0:
                    self._save_pipeline(output_dir=output_dir, step=step)

        self._save_pipeline(output_dir=output_dir, step=step)
        print(f"Training finished. Final checkpoint saved to {output_dir}/checkpoint-{step}")
