DeCo Diffusers integration
==========================

This repository follows the layout used for upstream Diffusers integration (see also
[NiT-diffusers](https://github.com/Bili-Sakura/NiT-diffusers)). Core components live under
`src/diffusers`:

- `models/transformers/transformer_deco.py` — `DeCoTransformer2DModel` (`ModelMixin` / `ConfigMixin`)
- `models/transformers/transformer_deco_c2i.py` — class-conditioned backbone
- `models/transformers/transformer_deco_t2i.py` — text-conditioned backbone
- `schedulers/scheduling_deco_flow_match_euler_discrete.py` — flow-matching Euler scheduler
- `pipelines/deco/pipeline_deco.py` — `DeCoPipeline` for sampling
- `scripts/convert_deco_to_diffusers.py` — convert legacy Lightning checkpoints

Install locally
---------------

```bash
pip install -e .
```

Or add `src` to `PYTHONPATH` when running scripts:

```bash
export PYTHONPATH="${PWD}/src:${PYTHONPATH}"
```

Convert a legacy checkpoint
---------------------------

```bash
python scripts/convert_deco_to_diffusers.py \
  --checkpoint path/to/deco.ckpt \
  --output deco-xl-diffusers \
  --model-size deco-xl-c2i \
  --check-load
```

The output directory contains `model_index.json`, `transformer/`, `scheduler/`, and `vae/`.

Sample from a converted checkpoint
----------------------------------

```bash
python scripts/sample_deco.py \
  --model deco-xl-diffusers \
  --class-label 207 \
  --height 256 \
  --width 256 \
  --guidance-scale 4.0
```

Run tests
---------

```bash
pip install -e ".[dev]"
pytest tests/test_deco_diffusers.py
```

For upstreaming to `huggingface/diffusers`, copy the files under `src/diffusers` into the
corresponding Diffusers package locations and register the classes in Diffusers' lazy import tables.
