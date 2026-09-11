# Runtime profiles

The retrieval entrypoint separates model code from the hardware runtime. Use
one environment per hardware profile; do not install every FlagOS backend into
the same runtime.

## Validated CUDA profile (current host)

The current A100 profile uses:

```text
Python 3.10
PyTorch 2.10.0+cpu plus Torch-FL CUDA boxing
transformers 4.57.6
hydra-core 1.3.2
pyarrow, Pillow, datasets, WebDataset
```

This profile runs with `train.hardware.device_type=flagos` and uses the
Torch-FL CUDA boxing path. FlagGems is installed for compatibility but remains
disabled during the baseline run.

## FlagOS/torch-fl profile (future vendor hardware)

Torch-FL's current compatibility contract pins PyTorch to the 2.10.x minor
line. Build/install the wheel and the vendor SDK in a separate environment,
following the Torch-FL platform guide. The common pieces are:

```text
PyTorch 2.10.x (matching the torch-fl wheel)
torch-fl (matching the same PyTorch minor line and vendor SDK)
transformers / Pillow / pyarrow / WebDataset
```

For multi-device training, add the FlagCX package/runtime supported by the
vendor image. Enable the portable operator path only when the platform guide
says its FlagGems/Triton backend is available:

```bash
export FLAGOS_USE_FLAGGEMS=1
```

Then launch with:

```bash
python -m flagscale.run \
  --config-path examples/retrieval/conf \
  --config-name train \
  action=test train=bge_m3 \
  train.hardware.device_type=flagos \
  train.hardware.distributed_backend=flagos \
  train.hardware.use_flaggems=true
```

`torch-fl` is not a universal pip-only dependency: the wheel may be built for
CUDA boxing, Ascend, MetaX, DCU, MUSA or another platform and must match that
platform's PyTorch/runtime ABI. Do not install the first available wheel into
the current PyTorch 2.11 CUDA environment.

## Megatron/FlagScale full-stack profile

The repository's `requirements/cuda/train.txt` is for FlagScale's Megatron
GPT examples. It adds `megatron_core`, `transformer_engine`, `tiktoken` and a
different Transformers pin. It is not needed by this native
Transformers/CLIP example. Installing `requirements/cuda/all.txt`
would also add vLLM and serving dependencies and can conflict with the
embedding environment. The native retrieval models use only the existing
FlagScale/PyTorch/Transformers dependency set; no additional embedding
framework is added.
