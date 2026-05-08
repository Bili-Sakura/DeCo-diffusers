from __future__ import annotations

from typing import Callable, Iterable, Optional, Sequence, Union

import numpy as np
import torch

from diffusers import DiffusionPipeline
from diffusers.image_processor import VaeImageProcessor
from diffusers.pipelines.pipeline_utils import ImagePipelineOutput
from diffusers.utils.torch_utils import randn_tensor

from deco_diffusers.models.autoencoder import DeCoPixelAutoencoder
from deco_diffusers.models.transformer_deco import DeCoTransformer2DModel
from deco_diffusers.schedulers.scheduling_deco_flow_match_euler_discrete import DeCoFlowMatchEulerDiscreteScheduler


ConditioningInput = Union[int, Sequence[int], torch.Tensor]


class DeCoPipeline(DiffusionPipeline):
    model_cpu_offload_seq = "transformer"
    _callback_tensor_inputs = ["latents"]

    def __init__(
        self,
        transformer: DeCoTransformer2DModel,
        scheduler: DeCoFlowMatchEulerDiscreteScheduler,
        vae: Optional[DeCoPixelAutoencoder] = None,
    ):
        super().__init__()
        if vae is None:
            vae = getattr(transformer, "vae", None)
        self.register_modules(transformer=transformer, scheduler=scheduler, vae=vae)
        self.image_processor = VaeImageProcessor(vae_scale_factor=1)

    @staticmethod
    def _to_list(value: ConditioningInput) -> list[int]:
        if isinstance(value, torch.Tensor):
            raise TypeError("Use tensor inputs directly without list conversion.")
        if isinstance(value, str):
            raise TypeError("String prompts are not supported for class-conditioned DeCo models.")
        if isinstance(value, (int, np.integer)):
            return [int(value)]
        if isinstance(value, Iterable):
            return [int(entry) for entry in value]
        raise TypeError("Unsupported conditioning input type.")

    @staticmethod
    def _resolve_batch_size(batch_size: Optional[int], resolved: int) -> int:
        if batch_size is None:
            return resolved
        if batch_size != resolved:
            if resolved != 1:
                raise ValueError("Resolved batch size does not match provided batch_size.")
        return batch_size

    def _prepare_class_labels(
        self,
        class_labels: Optional[torch.Tensor],
        prompt: Optional[ConditioningInput],
        batch_size: Optional[int],
        device: torch.device,
    ) -> tuple[torch.Tensor, int]:
        if class_labels is None and prompt is not None:
            if isinstance(prompt, torch.Tensor):
                class_labels = prompt
            else:
                class_labels = torch.tensor(self._to_list(prompt), dtype=torch.long)
        if class_labels is None:
            raise ValueError("class_labels or prompt must be provided for class-conditioned DeCo models")

        class_labels = class_labels.to(device=device, dtype=torch.long)
        if class_labels.ndim == 0:
            class_labels = class_labels[None]
        batch_size = self._resolve_batch_size(batch_size, int(class_labels.shape[0]))
        if class_labels.shape[0] != batch_size:
            if class_labels.shape[0] == 1:
                class_labels = class_labels.repeat(batch_size)
            else:
                raise ValueError("class_labels batch size must match batch_size")
        return class_labels, batch_size

    def _prepare_prompt_embeds(
        self,
        prompt_embeds: Optional[torch.Tensor],
        prompt: Optional[ConditioningInput],
        batch_size: Optional[int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, int]:
        if prompt_embeds is None and prompt is not None:
            if isinstance(prompt, torch.Tensor):
                prompt_embeds = prompt
            else:
                raise ValueError("prompt must be a torch.Tensor of embeddings for text-conditioned DeCo models")
        if prompt_embeds is None:
            raise ValueError("prompt_embeds must be provided for text-conditioned DeCo models")
        prompt_embeds = prompt_embeds.to(device=device, dtype=dtype)
        batch_size = self._resolve_batch_size(batch_size, int(prompt_embeds.shape[0]))
        if prompt_embeds.shape[0] != batch_size:
            if prompt_embeds.shape[0] == 1:
                prompt_embeds = prompt_embeds.repeat(batch_size, 1, 1)
            else:
                raise ValueError("prompt_embeds batch size must match batch_size")
        return prompt_embeds, batch_size

    def prepare_latents(
        self,
        batch_size: int,
        num_channels: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: Optional[Union[torch.Generator, list[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if latents is None:
            return randn_tensor(
                (batch_size, num_channels, height, width),
                generator=generator,
                device=device,
                dtype=dtype,
            )
        latents = latents.to(device=device, dtype=dtype)
        if latents.shape != (batch_size, num_channels, height, width):
            raise ValueError("Provided latents do not match the expected shape.")
        return latents

    @torch.no_grad()
    def __call__(
        self,
        prompt: Optional[ConditioningInput] = None,
        batch_size: Optional[int] = None,
        height: int = 256,
        width: int = 256,
        num_inference_steps: int = 50,
        guidance_scale: float = 1.0,
        class_labels: Optional[torch.Tensor] = None,
        negative_prompt: Optional[ConditioningInput] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        generator: Optional[Union[torch.Generator, list[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, torch.Tensor, torch.Tensor], None]] = None,
        callback_steps: int = 1,
    ):
        device = self._execution_device
        dtype = next(self.transformer.parameters()).dtype

        conditioning_type = self.transformer.config.conditioning_type
        do_cfg = guidance_scale is not None and float(guidance_scale) > 1.0

        if conditioning_type == "class":
            class_labels, batch_size = self._prepare_class_labels(
                class_labels=class_labels,
                prompt=prompt,
                batch_size=batch_size,
                device=device,
            )

            if do_cfg:
                if negative_prompt is not None:
                    if isinstance(negative_prompt, torch.Tensor):
                        uncond_labels = negative_prompt
                    else:
                        uncond_labels = torch.tensor(self._to_list(negative_prompt), dtype=torch.long)
                    uncond_labels = uncond_labels.to(device=device, dtype=torch.long)
                    if uncond_labels.ndim == 0:
                        uncond_labels = uncond_labels[None]
                    if uncond_labels.shape[0] != batch_size:
                        if uncond_labels.shape[0] == 1:
                            uncond_labels = uncond_labels.repeat(batch_size)
                        else:
                            raise ValueError("negative_prompt batch size must match batch_size")
                else:
                    null_label = int(self.transformer.config.num_classes)
                    uncond_labels = torch.full((batch_size,), null_label, device=device, dtype=torch.long)
        else:
            prompt_embeds, batch_size = self._prepare_prompt_embeds(
                prompt_embeds=prompt_embeds,
                prompt=prompt,
                batch_size=batch_size,
                device=device,
                dtype=dtype,
            )

            if do_cfg:
                if negative_prompt_embeds is None and negative_prompt is not None:
                    if isinstance(negative_prompt, torch.Tensor):
                        negative_prompt_embeds = negative_prompt
                    else:
                        raise ValueError("negative_prompt must be a torch.Tensor of embeddings for text-conditioned DeCo models")
                if negative_prompt_embeds is None:
                    negative_prompt_embeds = torch.zeros_like(prompt_embeds)
                negative_prompt_embeds = negative_prompt_embeds.to(device=device, dtype=dtype)
                if negative_prompt_embeds.shape[0] != batch_size:
                    if negative_prompt_embeds.shape[0] == 1:
                        negative_prompt_embeds = negative_prompt_embeds.repeat(batch_size, 1, 1)
                    else:
                        raise ValueError("negative_prompt_embeds batch size must match batch_size")

        patch_size = int(getattr(self.transformer.config, "patch_size", 1))
        if height % patch_size != 0 or width % patch_size != 0:
            raise ValueError("height and width must be divisible by the transformer patch size")

        latents = self.prepare_latents(
            batch_size=batch_size,
            num_channels=int(self.transformer.config.in_channels),
            height=int(height),
            width=int(width),
            dtype=dtype,
            device=device,
            generator=generator,
            latents=latents,
        )

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        if callback_steps <= 0:
            raise ValueError("callback_steps must be > 0")

        for step_index, timestep in enumerate(self.progress_bar(timesteps[:-1])):
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
            if callback is not None and step_index % callback_steps == 0:
                callback(step_index, timestep, latents)

        image = latents
        if self.vae is not None:
            image = self.vae.decode(image).sample
        elif output_type != "latent":
            raise ValueError(
                f"Cannot produce output_type '{output_type}' without a VAE. Provide a VAE or set output_type='latent'."
            )

        if output_type == "latent":
            if not return_dict:
                return (image,)
            return ImagePipelineOutput(images=image)

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()

        if output_type == "pil":
            image = self.numpy_to_pil(image)
        elif output_type == "np":
            image = image
        else:
            raise ValueError("output_type must be one of {'pil', 'np', 'latent'}")

        if not return_dict:
            return (image,)
        return ImagePipelineOutput(images=image)
