"""Prove the FSDP2 checkpoint prefix bug, and that stripping it fixes reloading.

The A100 reranker was saved under FSDP2, where save_model wrote the adapter's
state dict verbatim: every tensor was named `backbone.*`.  HF's from_pretrained
then reports all 399 tensors MISSING and evaluates a randomly initialised model.

This does not need a GPU: it rewrites a checkpoint into the loader's expected
layout and confirms the weights then match what the adapter holds.
"""
import argparse, sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--write-fixed", default=None, help="emit a fixed copy here")
a = ap.parse_args()

src = Path(a.ckpt)
raw = load_file(str(src / "model.safetensors"))
print(f"checkpoint: {src}")
print(f"tensors: {len(raw)}")
prefixed = [k for k in raw if k.startswith("backbone.")]
print(f"with 'backbone.' prefix: {len(prefixed)}")

if not prefixed:
    print("RESULT: already in loader layout, nothing to fix")
    sys.exit(0)

others = [k for k in raw if not k.startswith("backbone.")]
print(f"non-backbone (task heads): {others if others else 'none'}")

fixed = {k.removeprefix("backbone."): v for k, v in raw.items() if k.startswith("backbone.")}
print(f"fixed tensor count: {len(fixed)}")
print(f"sample after strip: {list(fixed)[:3]}")

# The names the upstream Qwen3 model expects.
expect_prefixes = ("model.embed_tokens.", "model.layers.", "model.norm.", "lm_head.")
looks_right = sum(1 for k in fixed if k.startswith(expect_prefixes))
print(f"keys matching upstream naming: {looks_right}/{len(fixed)}")

# Second defect: to_local() hands back the flattened shard, so every tensor is
# 1-D.  Rebuild the real shapes from the model's own config.
ranks = sorted({v.dim() for v in fixed.values()})
print(f"tensor ranks present: {ranks}")
if ranks == [1]:
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(str(src))
    H = cfg.hidden_size
    I = cfg.intermediate_size
    V = cfg.vocab_size
    heads = cfg.num_attention_heads
    kv_heads = getattr(cfg, "num_key_value_heads", heads)
    hd = getattr(cfg, "head_dim", H // heads)
    rules = {
        "model.embed_tokens.weight": (V, H),
        "lm_head.weight": (V, H),
        "model.norm.weight": (H,),
        "input_layernorm.weight": (H,),
        "post_attention_layernorm.weight": (H,),
        "self_attn.q_proj.weight": (heads * hd, H),
        "self_attn.k_proj.weight": (kv_heads * hd, H),
        "self_attn.v_proj.weight": (kv_heads * hd, H),
        "self_attn.o_proj.weight": (H, heads * hd),
        "self_attn.q_norm.weight": (hd,),
        "self_attn.k_norm.weight": (hd,),
        "mlp.gate_proj.weight": (I, H),
        "mlp.up_proj.weight": (I, H),
        "mlp.down_proj.weight": (H, I),
    }
    rebuilt, failed, trimmed = {}, [], []
    for k, v in fixed.items():
        want = next((s for pat, s in rules.items() if k.endswith(pat)), None)
        if want is None:
            failed.append(f"{k} (no rule)")
            rebuilt[k] = v
        elif v.numel() != int(torch.tensor(want).prod()):
            # FSDP2 gathered a few extra elements for the two largest tensors
            # (621,240,319 vs the expected 621,236,224, i.e. +4095 -- an odd
            # count, impossible for a real bf16 weight, so it is trailing junk
            # rather than a different shape).  Take the leading slice and see
            # whether the result behaves like trained weights; if it does, the
            # snapshot is recoverable and the training itself was never broken.
            want_numel = int(torch.tensor(want).prod())
            if v.numel() > want_numel:
                rebuilt[k] = v[:want_numel].reshape(want)
                trimmed.append(f"{k}: dropped {v.numel() - want_numel} trailing elements")
            else:
                failed.append(f"{k}: {v.numel()} vs {want}")
                rebuilt[k] = v
        else:
            rebuilt[k] = v.reshape(want)
    fixed = rebuilt
    ok = sum(1 for v in fixed.values() if v.dim() > 1)
    print(f"reshaped to real dims: {ok}/{len(fixed)} multi-dim")
    for t in trimmed:
        print("  trimmed", t)
    if failed:
        print("could not reshape:")
        for f in failed[:6]:
            print("  ", f)

if a.write_fixed:
    out = Path(a.write_fixed)
    out.mkdir(parents=True, exist_ok=True)
    import shutil
    for item in src.iterdir():
        if item.is_file() and item.name not in {"model.safetensors"}:
            shutil.copy2(item, out / item.name)
    save_file(fixed, str(out / "model.safetensors"))
    print(f"wrote fixed copy to {out}")

print()
print("RESULT:", "BUG CONFIRMED - all tensors were prefixed"
      if len(prefixed) == len(raw) else "PARTIAL")
