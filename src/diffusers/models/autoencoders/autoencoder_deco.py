from __future__ import annotations

from dataclasses import dataclass

import torch

try:
    from diffusers.configuration_utils import ConfigMixin, register_to_config
    from diffusers.models.modeling_utils import ModelMixin
    from diffusers.utils import BaseOutput
except Exception:  # pragma: no cover - importable without a full diffusers install.
    class BaseOutput(dict):
        def __post_init__(self):
            self.update(self.__dict__)

    class _Config(dict):
        def __getattr__(self, key):
            try:
                return self[key]
            except KeyError as error:
                raise AttributeError(key) from error

    class ConfigMixin:
        config_name = "config.json"

    class ModelMixin(torch.nn.Module):
        pass

    def register_to_config(init):
        def wrapper(self, *args, **kwargs):
            import inspect

            signature = inspect.signature(init)
            bound = signature.bind(self, *args, **kwargs)
            bound.apply_defaults()
            self.config = _Config({key: value for key, value in bound.arguments.items() if key != "self"})
            init(self, *args, **kwargs)

        return wrapper


@dataclass
class DeCoPixelAutoencoderOutput(BaseOutput):
    sample: torch.Tensor


class DeCoPixelAutoencoder(ModelMixin, ConfigMixin):
    config_name = "config.json"

    @register_to_config
    def __init__(self, scale: float = 1.0, shift: float = 0.0):
        super().__init__()

    def encode(self, sample: torch.Tensor) -> DeCoPixelAutoencoderOutput:
        latents = sample / float(self.config.scale) + float(self.config.shift)
        return DeCoPixelAutoencoderOutput(sample=latents)

    def decode(self, latents: torch.Tensor) -> DeCoPixelAutoencoderOutput:
        sample = (latents - float(self.config.shift)) * float(self.config.scale)
        return DeCoPixelAutoencoderOutput(sample=sample)

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        return self.decode(sample).sample
