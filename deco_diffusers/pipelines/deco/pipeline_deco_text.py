from __future__ import annotations

from typing import Callable, Optional, Union

import torch

from diffusers import DiffusionPipeline
from diffusers.pipelines.pipeline_utils import ImagePipelineOutput
from diffusers.utils.torch_utils import randn_tensor

from deco_diffusers.models.transformer_deco import DeCoTransformer2DModel
from deco_diffusers.schedulers.scheduling_deco_flow_match_euler_discrete import DeCoFlowMatchEulerDiscreteScheduler

PIPELINE_CLASS = "DeCoTextPipeline"


class DeCoTextPipeline(DiffusionPipeline):
    model_cpu_offload_seq = "transformer"
    _callback_tensor_inputs = ["latents"]

    def __init__(self, transformer: DeCoTransformer2DModel, scheduler: DeCoFlowMatchEulerDiscreteScheduler):
        super().__init__()
        if transformer.config.conditioning_type != "text":
            raise ValueError("DeCoTextPipeline requires a text-conditioned transformer.")
        self.register_modules(transformer=transformer, scheduler=scheduler)

    @staticmethod
    def _resolve_batch_size(batch_size: Optional[int], resolved: int) -> int:
        if batch_size is None:
            return resolved
        if batch_size != resolved:
            if resolved != 1:
                raise ValueError(f"Resolved batch size {resolved} does not match provided batch_size {batch_size}.")
        return batch_size

    def _prepare_prompt_embeds(
        self,
        prompt_embeds: Optional[torch.Tensor],
        prompt: Optional[torch.Tensor],
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
        expected_shape = (batch_size, num_channels, height, width)
        if latents.shape != expected_shape:
            raise ValueError(
                f"Provided latents have shape {tuple(latents.shape)}; expected {expected_shape} "
                "(batch_size, num_channels, height, width)."
            )
        return latents

    @torch.no_grad()
    def __call__(
        self,
        batch_size: Optional[int] = None,
        height: int = 256,
        width: int = 256,
        num_inference_steps: int = 50,
        guidance_scale: float = 1.0,
        prompt: Optional[torch.Tensor] = None,
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
        if callback is not None and callback_steps <= 0:
            raise ValueError("callback_steps must be > 0")

        prompt_embeds, batch_size = self._prepare_prompt_embeds(
            prompt_embeds=prompt_embeds,
            prompt=prompt,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )

        do_cfg = guidance_scale is not None and float(guidance_scale) > 1.0
        if do_cfg:
            if negative_prompt_embeds is None:
                negative_prompt_embeds = torch.zeros_like(prompt_embeds)
            negative_prompt_embeds = negative_prompt_embeds.to(device=device, dtype=dtype)
            if negative_prompt_embeds.shape[0] != batch_size:
                if negative_prompt_embeds.shape[0] == 1:
                    negative_prompt_embeds = negative_prompt_embeds.repeat(batch_size, 1, 1)
                else:
                    raise ValueError("negative_prompt_embeds batch size must match batch_size")

        if not hasattr(self.transformer.config, "patch_size"):
            raise ValueError(
                "Transformer config is missing required attribute patch_size. Ensure the transformer config is valid."
            )
        patch_size = int(self.transformer.config.patch_size)
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
        sampling_timesteps = timesteps[:-1]

        for step_index, timestep in enumerate(self.progress_bar(sampling_timesteps)):
            latent_model_input = self.scheduler.scale_model_input(latents, timestep)

            if do_cfg:
                latent_model_input = torch.cat([latent_model_input, latent_model_input], dim=0)
                model_output = self.transformer(
                    latent_model_input,
                    timestep,
                    encoder_hidden_states=torch.cat([negative_prompt_embeds, prompt_embeds], dim=0),
                ).sample
                model_output_uncond, model_output_text = model_output.chunk(2)
                model_output = model_output_uncond + float(guidance_scale) * (model_output_text - model_output_uncond)
            else:
                model_output = self.transformer(latent_model_input, timestep, encoder_hidden_states=prompt_embeds).sample

            latents = self.scheduler.step(model_output, timestep, latents).prev_sample
            if callback is not None and step_index % callback_steps == 0:
                callback(step_index, timestep, latents)

        image = latents
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
