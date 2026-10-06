# Retrieval model training with FlagScale

This example adds a FlagScale `native` training entrypoint for the five local
checkpoints under `/flagos-search-models`:

| Config | Model | Data | Objective |
| --- | --- | --- | --- |
| `bge_m3` | BGE-M3 | T2Ranking triplet-15 | in-batch + hard-negative contrastive |
| `qwen3_embedding` | Qwen3-Embedding-0.6B | T2Ranking triplet-15 | in-batch + hard-negative contrastive |
| `qwen3_vl_embedding` | Qwen3-VL-Embedding-8B | COCO WebDataset train/test shards | symmetric image/text contrastive |
| `qwen3_reranker` | Qwen3-Reranker-8B | T2Ranking triplet-15 | binary positive/negative ranking |
| `clip_vit_large` | CLIP ViT-L/14 | COCO WebDataset train shards | symmetric image/text CLIP loss |

The runner is FlagScale: Hydra composes the YAML, the native backend launches
the built-in `flagscale/train/train_retrieval.py` entrypoint through `torchrun`,
and logs/checkpoints are placed in `/flagos-search-ckpts/<model-name>`. Users
only select the model config; they do not need to invoke a model-specific
`train.py` directly.

The implementation follows FlagScale's module boundaries:

* `flagscale/models/retrieval/` contains the registry and one adapter module
  for each of the five upstream model families. The adapter loads weights from
  the configured external `model_path`; it does not copy model assets into the
  source tree.
* `flagscale/train/datasets/retrieval.py` contains the streaming T2Ranking and
  COCO readers, deterministic split logic, and loader construction.
* `flagscale/train/losses/retrieval.py` contains the embedding, VL embedding,
  reranker, and CLIP objectives.
* `flagscale/train/train_retrieval.py` remains the small native entrypoint that
  wires FlagScale's device/DDP lifecycle, the registered model, dataset loaders,
  loss dispatch, optimizer, and epoch loop together.

See [ENVIRONMENTS.md](ENVIRONMENTS.md) for the CUDA and future Torch-FL
dependency profiles. They must be installed as separate, ABI-compatible
environments.

The training loop uses a device abstraction rather than calling CUDA APIs in
the model path. The validated default is `train.hardware.device_type=cuda`.
For a FlagOS vendor environment, use a PyTorch 2.10.x-compatible torch-fl
installation and switch the device/backend at launch time:

```bash
python -m flagscale.run \
  --config-path examples/retrieval/conf \
  --config-name train \
  action=test train=bge_m3 \
  train.hardware.device_type=flagos \
  train.hardware.distributed_backend=flagos \
  train.hardware.use_flaggems=true
```

This does not make every operator automatically portable: the selected
vendor's torch-fl wheel, runtime SDK, FlagGems/Triton backend and (for
multi-device jobs) FlagCX still have to be installed according to that
platform's guide. It does ensure that device selection, DDP initialization,
AMP selection and data movement are centralized so the model/loss code does
not need a vendor-specific fork.

## Environment

The default config uses the standard FlagScale CUDA/train dependency profile:

```bash
source /flagos-search-codes/torch-fl-cuda/.venv/bin/activate
```

It includes the Hugging Face Transformers and PyTorch packages. For
multi-GPU training, set `experiment.runner.nproc_per_node` to the number of
visible GPUs. On a multi-node run, provide a FlagScale hostfile as documented
in `docs/getting-started.md`.

## Dry run and start commands

From `/flagos-search-codes/FlagScale`:

```bash
# Validate Hydra composition and print the generated launch command.
python -m flagscale.run \
  --config-path examples/retrieval/conf \
  --config-name train \
  action=dryrun \
  train=bge_m3

# Start one model. Replace the override with any of the five config names.
# Each config runs train/validation/test for 3 epochs by default.
python -m flagscale.run \
  --config-path examples/retrieval/conf \
  --config-name train \
  action=test \
  train=bge_m3
```

Useful overrides for a small 3-epoch smoke test:

```bash
python -m flagscale.run \
  --config-path examples/retrieval/conf \
  --config-name train \
  action=test \
  train=qwen3_embedding \
  train.epochs=3 train.data.max_samples=64 train.data.eval_max_samples=16 \
  train.batch_size=1 train.gradient_accumulation_steps=1
```

The full T2Ranking and COCO streams are lazy; they are not loaded into RAM.
`train.data.max_samples` can cap a run for a smoke test; leave it `null` for
the configured full stream. T2Ranking's train-only shards are deterministically
partitioned into train/validation/test rows, while COCO uses train/validation
rows plus its supplied test shards.

## Retrieval quality evaluation

Loss alone does not show whether an encoder retrieves the right passage. Use
`evaluate_retrieval.py` to rank each query's positive against its hard
negatives and report Recall@1/5/10 and MRR:

```bash
python examples/retrieval/evaluate_retrieval.py \
  --model-type bge_m3 \
  --model-path /flagos-search-models/bge-m3/bge-m3 \
  --task embedding \
  --data-path /flagos-search-datasets/t2ranking/t2ranking/triplet-15 \
  --split validation --num-negatives 15 --batch-size 16 --max-samples 512 \
  --device flagos --output /flagos-search-ckpts/eval/bge_m3_val.json
```

Pass `--checkpoint <trained-dir>` to evaluate a fine-tuned model instead of the
base weights. For CLIP and VL embedding the script does symmetric in-batch
image/text retrieval and reports both directions. The script is single-process
and does not need torchrun; run it on `cpu` or one accelerator.

`evaluate_models.sh` wraps the same script for a base-vs-fine-tuned comparison
over several models, with identical settings on both sides:

```bash
# all five models, base and fine-tuned, on their held-out test splits
./evaluate_models.sh

# one model, validation split, small sample
MODELS=bge_m3 SPLIT=validation MAX_SAMPLES=500 ./evaluate_models.sh

# only the base models, waiting until the device is free first
TAGS=base WAIT_FOR_IDLE=1 ./evaluate_models.sh
```

Both checkpoints of a pair must use the same split and sample count; a mismatch
between 128 and 256 samples once produced a wrong conclusion, which is why the
wrapper always evaluates a pair symmetrically. Model settings are read from the
training YAML, so a reranker trained with native yes/no logits is not scored
through a random head.

## Monitoring long runs

`monitor_training.sh` appends a health verdict every interval (default 2h) so an
unattended run can be audited afterwards. Each pass reports whether a training
process is alive, whether its step is advancing, whether the loss is below the
random baseline (ln2 for the binary objective) and whether the device is busy:

```bash
MONITOR_INTERVAL_SECONDS=7200 ./monitor_training.sh /flagos-search-ckpts/monitor.log
```

## Output

Each completed run writes a Hugging Face Transformers checkpoint to
`/flagos-search-ckpts/<model-name>` plus a `training_complete.json` marker.
An intermediate snapshot is also written after **every epoch** to
`<model-name>/checkpoint-epoch-<n>/` (controlled by `train.save_every_epoch`,
default on). These 8B runs take days, so a crash between epochs would otherwise
discard the whole run; only the final directory carries `training_complete.json`,
which makes intermediates distinguishable from a finished model.

FlagScale's Hydra, host and runner logs are created under the same experiment
directory.
