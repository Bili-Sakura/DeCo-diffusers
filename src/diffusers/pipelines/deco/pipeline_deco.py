from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch

try:
    from diffusers.image_processor import VaeImageProcessor
    from diffusers.pipelines.pipeline_utils import DiffusionPipeline
    from diffusers.utils import BaseOutput
    from diffusers.utils.torch_utils import randn_tensor
except Exception:  # pragma: no cover - importable without a full diffusers install.
    class BaseOutput(dict):
        def __post_init__(self):
            self.update(self.__dict__)

    class DiffusionPipeline:
        def register_modules(self, **kwargs):
            for name, module in kwargs.items():
                setattr(self, name, module)

        @property
        def _execution_device(self):
            return torch.device("cpu")

        def maybe_free_model_hooks(self):
            pass

    class VaeImageProcessor:
        def __init__(self, vae_scale_factor: int = 1):
            self.vae_scale_factor = vae_scale_factor

        def postprocess(self, image, output_type="pil"):
            return image

    def randn_tensor(shape, generator=None, device=None, dtype=None):
        return torch.randn(shape, generator=generator, device=device, dtype=dtype)

from diffusers.models.autoencoders import DeCoPixelAutoencoder
from diffusers.models.transformers import DeCoTransformer2DModel
from diffusers.schedulers import DeCoFlowMatchEulerDiscreteScheduler


@dataclass
class DeCoPipelineOutput(BaseOutput):
    images: Union[torch.FloatTensor, List]


class DeCoPipeline(DiffusionPipeline):
    model_cpu_offload_seq = "transformer->vae"
    _optional_components = ["vae"]

    def __init__(
        self,
        transformer: DeCoTransformer2DModel,
        scheduler: DeCoFlowMatchEulerDiscreteScheduler,
        vae: Optional[DeCoPixelAutoencoder] = None,
    ):
        super().__init__()
        self.register_modules(transformer=transformer, scheduler=scheduler, vae=vae)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self._get_vae_scale_factor())

    def _get_vae_scale_factor(self) -> int:
        if self.vae is None:
            return 1
        if self.vae.__class__.__name__ == "AutoencoderDC" or "dc-ae" in getattr(self.vae.config, "_name_or_path", ""):
            return 32
        block_out_channels = getattr(self.vae.config, "block_out_channels", None)
        if block_out_channels:
            return 2 ** (len(block_out_channels) - 1)
        return 1

    def _prepare_latents(
        self,
        batch_size: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]],
    ) -> Tuple[torch.Tensor, int, int]:
        vae_scale_factor = self._get_vae_scale_factor()
        if height % vae_scale_factor != 0 or width % vae_scale_factor != 0:
            raise ValueError(f"height and width must be divisible by VAE scale factor {vae_scale_factor}.")
        latent_height = height // vae_scale_factor
        latent_width = width // vae_scale_factor
        patch_size = int(self.transformer.config.patch_size)
        if latent_height % patch_size != 0 or latent_width % patch_size != 0:
            raise ValueError("Latent height and width must be divisible by the transformer's patch_size.")
        latents = randn_tensor(
            (batch_size, int(self.transformer.config.in_channels), latent_height, latent_width),
            generator=generator,
            device=device,
            dtype=dtype,
        )
        return latents, latent_height, latent_width

    def _decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        if self.vae is None:
            return latents
        scaling_factor = getattr(self.vae.config, "scaling_factor", 1.0)
        latents = latents / float(scaling_factor)
        decoded = self.vae.decode(latents)
        return decoded.sample if hasattr(decoded, "sample") else decoded

    @torch.no_grad()
    def __call__(
        self,
        batch_size: int = 1,
        height: int = 256,
        width: int = 256,
        num_inference_steps: int = 50,
        guidance_scale: float = 1.0,
        class_labels: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        generator: Optional[Union[torch.Generator, list[torch.Generator]]] = None,
        output_type: str = "pil",
        return_dict: bool = True,
    ):
        device = self._execution_device
        dtype = next(self.transformer.parameters()).dtype

        conditioning_type = self.transformer.config.conditioning_type
        do_cfg = guidance_scale is not None and float(guidance_scale) > 1.0

        if conditioning_type == "class":
            if class_labels is None:
                raise ValueError("class_labels must be provided for class-conditioned DeCo models")
            class_labels = class_labels.to(device=device, dtype=torch.long)
            if class_labels.ndim == 0:
                class_labels = class_labels[None]
            if class_labels.shape[0] != batch_size:
                if class_labels.shape[0] == 1:
                    class_labels = class_labels.repeat(batch_size)
                else:
                    raise ValueError("class_labels batch size must match batch_size")

            if do_cfg:
                null_label = int(self.transformer.config.num_classes)
                uncond_labels = torch.full((batch_size,), null_label, device=device, dtype=torch.long)
        else:
            if prompt_embeds is None:
                raise ValueError("prompt_embeds must be provided for text-conditioned DeCo models")
            prompt_embeds = prompt_embeds.to(device=device, dtype=dtype)
            if prompt_embeds.shape[0] != batch_size:
                if prompt_embeds.shape[0] == 1:
                    prompt_embeds = prompt_embeds.repeat(batch_size, 1, 1)
                else:
                    raise ValueError("prompt_embeds batch size must match batch_size")

            if do_cfg:
                if negative_prompt_embeds is None:
                    negative_prompt_embeds = torch.zeros_like(prompt_embeds)
                negative_prompt_embeds = negative_prompt_embeds.to(device=device, dtype=dtype)
                if negative_prompt_embeds.shape[0] != batch_size:
                    if negative_prompt_embeds.shape[0] == 1:
                        negative_prompt_embeds = negative_prompt_embeds.repeat(batch_size, 1, 1)
                    else:
                        raise ValueError("negative_prompt_embeds batch size must match batch_size")

        latents, _, _ = self._prepare_latents(
            batch_size=batch_size,
            height=height,
            width=width,
            dtype=dtype,
            device=device,
            generator=generator,
        )

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        for timestep in timesteps[:-1]:
            latent_model_input = self.scheduler.scale_model_input(latents, timestep)

            if do_cfg:
                latent_model_input = torch.cat([latent_model_input, latent_model_input], dim=0)

                if conditioning_type == "class":
                    model_output = self.transformer(
                        latent_model_input,
                        timestep,
                        class_labels=torch.cat([uncond_labels, class_labels], dim=0),
                    ).sample
                else:
                    model_output = self.transformer(
                        latent_model_input,
                        timestep,
                        encoder_hidden_states=torch.cat([negative_prompt_embeds, prompt_embeds], dim=0),
                    ).sample

                model_output_uncond, model_output_text = model_output.chunk(2)
                model_output = model_output_uncond + float(guidance_scale) * (model_output_text - model_output_uncond)
            else:
                if conditioning_type == "class":
                    model_output = self.transformer(latent_model_input, timestep, class_labels=class_labels).sample
                else:
                    model_output = self.transformer(latent_model_input, timestep, encoder_hidden_states=prompt_embeds).sample

            latents = self.scheduler.step(model_output, timestep, latents).prev_sample

        if output_type == "latent":
            if not return_dict:
                return (latents,)
            return DeCoPipelineOutput(images=latents)

        image = self._decode_latents(latents)
        image = (image / 2 + 0.5).clamp(0, 1)
        image = self.image_processor.postprocess(image, output_type=output_type)

        self.maybe_free_model_hooks()
        if not return_dict:
            return (image,)
        return DeCoPipelineOutput(images=image)
