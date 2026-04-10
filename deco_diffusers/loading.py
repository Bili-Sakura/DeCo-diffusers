from __future__ import annotations

from typing import Any

from deco_diffusers.models.transformer_2d import DeCoTransformer2DModel


def load_transformer_from_legacy_lightning_checkpoint(
    checkpoint_path: str,
    *,
    use_ema: bool = False,
    strict: bool = False,
    **transformer_kwargs: Any,
) -> DeCoTransformer2DModel:
    model = DeCoTransformer2DModel(**transformer_kwargs)
    model.load_legacy_checkpoint(checkpoint_path=checkpoint_path, use_ema=use_ema, strict=strict)
    return model
