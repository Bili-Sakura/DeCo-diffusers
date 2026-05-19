import pytest

torch = pytest.importorskip("torch")

from deco_diffusers.models import DeCoTransformer2DModel
from deco_diffusers.pipelines import DeCoPipeline
from deco_diffusers.schedulers import DeCoFlowMatchEulerDiscreteScheduler


def test_deco_transformer_class_forward():
    model = DeCoTransformer2DModel(
        conditioning_type="class",
        in_channels=3,
        patch_size=2,
        num_groups=12,
        hidden_size=192,
        hidden_size_x=32,
        num_blocks=6,
        num_cond_blocks=2,
        num_classes=10,
    )
    sample = torch.randn(2, 3, 32, 32)
    timesteps = torch.tensor([0.5, 0.25])
    class_labels = torch.tensor([1, 2])

    output = model(sample, timesteps, class_labels=class_labels)

    assert output.sample.shape == sample.shape


def test_scheduler_euler_step():
    scheduler = DeCoFlowMatchEulerDiscreteScheduler(shift=1.0)
    scheduler.set_timesteps(4)
    sample = torch.ones(1, 3, 8, 8)
    velocity = torch.zeros_like(sample)

    output = scheduler.step(velocity, scheduler.timesteps[0], sample)

    assert output.prev_sample.shape == sample.shape


def test_pipeline_instantiation():
    transformer = DeCoTransformer2DModel(
        conditioning_type="class",
        in_channels=3,
        patch_size=2,
        num_groups=12,
        hidden_size=192,
        hidden_size_x=32,
        num_blocks=6,
        num_cond_blocks=2,
        num_classes=10,
    )
    scheduler = DeCoFlowMatchEulerDiscreteScheduler()
    pipe = DeCoPipeline(transformer=transformer, scheduler=scheduler, vae=None)

    assert pipe.transformer.config.conditioning_type == "class"
