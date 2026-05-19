DeCo Diffusers integration
==========================

Native [Diffusers](https://github.com/huggingface/diffusers) components for DeCo, following the layout of
[NiT-diffusers](https://github.com/Bili-Sakura/NiT-diffusers).

```
src/diffusers/
├── models/
│   ├── autoencoders/autoencoder_deco.py
│   ├── layers/
│   └── transformers/
│       ├── transformer_deco.py
│       ├── transformer_deco_c2i.py
│       └── transformer_deco_t2i.py
├── pipelines/deco/pipeline_deco.py
└── schedulers/scheduling_deco_flow_match_euler_discrete.py
```

Install
-------

```bash
pip install -e .
```

Scripts add `src/` to `PYTHONPATH` automatically. You can also export:

```bash
export PYTHONPATH="${PWD}/src:${PYTHONPATH}"
```

Convert a legacy Lightning checkpoint
---------------------------------------

```bash
python scripts/convert_deco_to_diffusers.py \
  --checkpoint path/to/deco.ckpt \
  --output deco-xl-diffusers \
  --model-size deco-xl-c2i \
  --check-load
```

Sample
------

```bash
python scripts/sample_deco.py \
  --model deco-xl-diffusers \
  --class-label 207 \
  --height 256 \
  --width 256 \
  --guidance-scale 4.0
```

Train (class-conditioned)
-------------------------

```bash
python scripts/train_deco.py \
  --train-data-dir /path/to/imagenet/train \
  --output-dir ./outputs/deco-train
```

Python API
----------

```python
from diffusers import DeCoPipeline

pipe = DeCoPipeline.from_pretrained("deco-xl-diffusers")
images = pipe(class_labels=[207], batch_size=1, num_inference_steps=50, guidance_scale=4.0).images
```

Upstreaming
-----------

Copy `src/diffusers/models`, `pipelines`, and `schedulers` subtrees into the Hugging Face `diffusers` package and register classes in the lazy import tables.
