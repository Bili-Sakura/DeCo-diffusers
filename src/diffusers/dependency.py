# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Import Hugging Face diffusers primitives with fallbacks for local src/ testing."""

from __future__ import annotations

import torch.nn as nn

try:
    from diffusers.configuration_utils import ConfigMixin, register_to_config
    from diffusers.image_processor import VaeImageProcessor
    from diffusers.models.modeling_utils import ModelMixin
    from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput
    from diffusers.schedulers.scheduling_utils import SchedulerMixin, SchedulerOutput
    from diffusers.utils import BaseOutput
    from diffusers.utils.torch_utils import randn_tensor
except Exception:  # pragma: no cover
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

    class ModelMixin(nn.Module):
        pass

    class SchedulerMixin:
        pass

    class SchedulerOutput:
        def __init__(self, prev_sample):
            self.prev_sample = prev_sample

    class DiffusionPipeline:
        def register_modules(self, **kwargs):
            for name, module in kwargs.items():
                setattr(self, name, module)

        @property
        def _execution_device(self):
            return "cpu"

        def maybe_free_model_hooks(self):
            pass

    class ImagePipelineOutput:
        def __init__(self, images):
            self.images = images

    class VaeImageProcessor:
        def __init__(self, vae_scale_factor=1):
            self.vae_scale_factor = vae_scale_factor

        def postprocess(self, image, output_type="pil"):
            return image

    def randn_tensor(shape, generator=None, device=None, dtype=None):
        import torch

        return torch.randn(shape, generator=generator, device=device, dtype=dtype)

    def register_to_config(init):
        def wrapper(self, *args, **kwargs):
            import inspect

            signature = inspect.signature(init)
            bound = signature.bind(self, *args, **kwargs)
            bound.apply_defaults()
            self.config = _Config({key: value for key, value in bound.arguments.items() if key != "self"})
            return init(self, *args, **kwargs)

        return wrapper
