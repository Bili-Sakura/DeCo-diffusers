from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import torch

from diffusers.dependency import BaseOutput, ConfigMixin, ModelMixin, register_to_config


@dataclass
class DeCoAutoencoderOutput(BaseOutput):
    sample: torch.Tensor


class DeCoPixelAutoencoder(ModelMixin, ConfigMixin):
    config_name = "config.json"

    @register_to_config
    def __init__(self, scale: float = 1.0, shift: float = 0.0):
        super().__init__()

    def encode(self, sample: torch.Tensor) -> DeCoAutoencoderOutput:
        latents = sample / float(self.config.scale) + float(self.config.shift)
        return DeCoAutoencoderOutput(sample=latents)

    def decode(self, latents: torch.Tensor) -> DeCoAutoencoderOutput:
        sample = (latents - float(self.config.shift)) * float(self.config.scale)
        return DeCoAutoencoderOutput(sample=sample)

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        return self.decode(sample).sample
