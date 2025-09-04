import argparse
import json
import math
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Iterable
from datasets import load_dataset

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import util
os.environ["CUDA_VISIBLE_DEVICES"] = str(util.get_free_gpu())


@dataclass
class MLPTriplet:
    """Holds references for an MLP block's linear layers: up/gated"""
    name_prefix: str
    up_or_gate: nn.Linear
    down: nn.Linear


def find_mlp_linears(model: nn.Module) -> List[MLPTriplet]:
    """
    Find MLP 'up/gate' and 'down' Linear layers for common architectures.
    """
    name_to_mod = dict(model.named_modules())
    triplets: List[MLPTriplet] = []

    # Strategy: look for modules named like "*.mlp" that contain the needed submodules.
    mlp_prefixes = set()
    for name, module in name_to_mod.items():
        if name.endswith(".mlp"):
            mlp_prefixes.add(name)

    # LLaMA-style first
    for pref in sorted(mlp_prefixes):
        gate = name_to_mod.get(f"{pref}.gate_proj", None)
        up   = name_to_mod.get(f"{pref}.up_proj", None)
        down = name_to_mod.get(f"{pref}.down_proj", None)

        if isinstance(down, nn.Linear) and (isinstance(gate, nn.Linear) or isinstance(up, nn.Linear)):
            up_or_gate = gate if isinstance(gate, nn.Linear) else up
            triplets.append(MLPTriplet(name_prefix=pref, up_or_gate=up_or_gate, down=down))
            continue

        # GPT-2-style fallback
        c_fc  = name_to_mod.get(f"{pref}.c_fc", None)
        c_proj = name_to_mod.get(f"{pref}.c_proj", None)
        if isinstance(c_fc, nn.Linear) and isinstance(c_proj, nn.Linear):
            triplets.append(MLPTriplet(name_prefix=pref, up_or_gate=c_fc, down=c_proj))

    # If none found, attempt a generic pass: look for siblings in same parent with 'down' in name.
    if not triplets:
        # Very generic: any Linear with 'up' or 'gate' that has a sibling 'down'.
        for name, mod in name_to_mod.items():
            if isinstance(mod, nn.Linear) and (("up" in name) or ("gate" in name)):
                parent = name.rsplit(".", 1)[0] if "." in name else ""
                # find a 'down' Linear in same parent scope
                for cand, m2 in name_to_mod.items():
                    if not cand.startswith(parent):
                        continue
                    if "down" in cand and isinstance(m2, nn.Linear):
                        pref = parent if parent else name
                        triplets.append(MLPTriplet(name_prefix=pref, up_or_gate=mod, down=m2))
                        break

    return triplets


# --------------------
# Activation Collection
# --------------------
@torch.no_grad()
def tokenize_batch(tokenizer, prompts: List[str], device: torch.device, max_length: int = 512):
    toks = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    return {k: v.to(device) for k, v in toks.items()}


class ActivationBuffer:
    """
    Stores mean activations per neuron for each MLP up/gate layer.
    For memory efficiency we keep running mean across all tokens.
    """
    def __init__(self, out_features: int, device: torch.device, dtype=torch.bfloat16):
        self.count = 0
        self.mean = torch.zeros(out_features, dtype=dtype, device=device)

    def update(self, activations_2d: torch.Tensor):
        """
        activations_2d: [tokens_total, neurons]
        """
        with torch.no_grad():
            sum_ = activations_2d.double().sum(dim=0)
            n = activations_2d.shape[0]
            self.mean = (self.mean * self.count + sum_) / (self.count + n)
            self.count += n


def collect_means(model: nn.Module,
                  tokenizer,
                  prompts: List[str],
                  mlps: List[MLPTriplet],
                  device: torch.device,
                  max_length: int = 256,
                  batch_size: int = 4) -> Dict[str, torch.Tensor]:
    """
    Returns: {layer_name: mean_activation_per_neuron} for the up/gate layer.
    """
    handles = []
    buffers: Dict[str, ActivationBuffer] = {}

    def make_hook(layer_name: str):
        def hook(_module, _inp, out):
            # out shape: [batch, seq, neurons]
            if out.dim() == 3:
                B, T, N = out.shape
                flat = out.reshape(B * T, N)
            elif out.dim() == 2:
                flat = out
                N = flat.shape[-1]
            else:
                return
            if layer_name not in buffers:
                buffers[layer_name] = ActivationBuffer(N, device=out.device, dtype=torch.bfloat16)
            buffers[layer_name].update(flat.detach())
        return hook

    # Register hooks on up/gate outputs
    for trip in mlps:
        h = trip.up_or_gate.register_forward_hook(make_hook(f"{trip.name_prefix}.__up__"))
        handles.append(h)

    model.eval()
    # Simple batched forward passes
    for i in range(0, len(prompts), batch_size):
        chunk = prompts[i:i + batch_size]
        inputs = tokenize_batch(tokenizer, chunk, device, max_length=max_length)
        _ = model(**inputs)

    # Cleanup hooks
    for h in handles:
        h.remove()

    # Convert buffers to tensors
    means = {k: v.mean.float().cpu() for k, v in buffers.items()}
    return means


# --------------------
# Z-score & Selection
# --------------------
def zscore(x: torch.Tensor) -> torch.Tensor:
    mu = x.mean()
    std = x.std(unbiased=False)
    if std.item() == 0.0:
        return torch.zeros_like(x)
    return (x - mu) / std


def select_safety_neurons(
    means_harmful: Dict[str, torch.Tensor],
    means_benign: Dict[str, torch.Tensor],
    z_threshold: float
) -> Dict[str, List[int]]:
    """
    For each layer: delta = mean_benign - mean_harmful
    safety neurons = {i | zscore(delta)[i] > z_threshold}
    """
    selected: Dict[str, List[int]] = {}
    for layer_name in means_harmful.keys():
        if layer_name not in means_benign:
            continue
        delta = means_benign[layer_name] - means_harmful[layer_name]
        z = zscore(delta)
        idx = (z > z_threshold).nonzero(as_tuple=True)[0].tolist()
        selected[layer_name] = idx
    return selected


# --------------------
# Freezing via Gradient Masks
# --------------------
def _register_param_mask_hook(param: torch.nn.Parameter, mask: torch.Tensor):
    if mask.device != param.device:
        mask = mask.to(param.device)
    def _hook(grad):
        return grad * mask
    return param.register_hook(_hook)


def _freeze_linear_out_rows(linear: nn.Linear, rows: Iterable[int]):
    rows = torch.as_tensor(list(rows), dtype=torch.long, device=linear.weight.device)
    if rows.numel() == 0:
        return []
    hooks = []
    # weight rows
    wmask = torch.ones_like(linear.weight)
    wmask.index_fill_(0, rows, 0.0)
    hooks.append(_register_param_mask_hook(linear.weight, wmask))
    # bias entries
    if linear.bias is not None:
        bmask = torch.ones_like(linear.bias)
        bmask.index_fill_(0, rows, 0.0)
        hooks.append(_register_param_mask_hook(linear.bias, bmask))
    return hooks


def _freeze_linear_in_cols(linear: nn.Linear, cols: Iterable[int]):
    cols = torch.as_tensor(list(cols), dtype=torch.long, device=linear.weight.device)
    if cols.numel() == 0:
        return []
    wmask = torch.ones_like(linear.weight)
    wmask.index_fill_(1, cols, 0.0)
    return [_register_param_mask_hook(linear.weight, wmask)]


def freeze_safety_neurons(model: nn.Module,
                          mlps: List[MLPTriplet],
                          selected: Dict[str, List[int]]) -> Dict[str, List[torch.utils.hooks.RemovableHandle]]:
    """
    Register gradient masks to freeze selected neurons:
      - rows in up/gate
      - columns in down
    Returns hook handles so you can .remove() later if needed.
    """
    handles: Dict[str, List[torch.utils.hooks.RemovableHandle]] = {}
    for trip in mlps:
        key = f"{trip.name_prefix}.__up__"
        neurons = selected.get(key, [])
        if not neurons:
            continue

        h_list = []
        h_list += _freeze_linear_out_rows(trip.up_or_gate, neurons)
        h_list += _freeze_linear_in_cols(trip.down, neurons)
        handles[key] = h_list
    return handles


# --------------------
# Save / Load manifest
# --------------------
@dataclass
class FreezeManifest:
    model_name: str
    z_threshold: float
    layer_to_neurons: Dict[str, List[int]]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @staticmethod
    def from_json(js: str) -> "FreezeManifest":
        obj = json.loads(js)
        return FreezeManifest(
            model_name=obj["model_name"],
            z_threshold=float(obj["z_threshold"]),
            layer_to_neurons={k: list(map(int, v)) for k, v in obj["layer_to_neurons"].items()},
        )


# --------------------
# Main
# --------------------
def main():
    model = "google/gemma-2b-it"
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=model,
                        help="HF model id or local path (e.g., meta-llama/Llama-3.2-1B-Instruct)")
    parser.add_argument("--out_dir", type=str, default="./frozen/"+model.split("/")[-1]+"-frozen-new",
                        help="Directory to save model, tokenizer, and freeze_manifest.json")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--z_threshold", type=float, default=2.0, help="Z-score threshold for selecting safety neurons")
    parser.add_argument("--max_len", type=int, default=512, help="Max sequence length when collecting activations")
    parser.add_argument("--batch_size", type=int, default=8)

    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)

    print(f"[load] Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.to(device)

    # Find MLP triplets
    mlps = find_mlp_linears(model)
    if not mlps:
        raise RuntimeError("Could not find MLP (up/gate, down) Linear layers. "
                           "Model architecture may be unsupported by this script.")
    
    # Load harmful dataset
    harmful_ds = load_dataset("walledai/CatHarmfulQA")
    harmful = harmful_ds["en"]["prompt"]
    print(f"[data] Loaded {len(harmful)} harmful prompts")

    # Load benign dataset (same size as harmful to balance)
    benign_ds = load_dataset("facebook/natural_reasoning")
    benign = benign_ds["train"]["question"][:len(harmful)]
    print(f"[data] Loaded {len(benign)} benign prompts")
    
    print(f"[collect] Collecting mean activations on {len(benign)} benign and {len(harmful)} harmful prompts ...")
    means_benign = collect_means(model, tokenizer, benign, mlps, device, max_length=args.max_len,
                                 batch_size=args.batch_size)
    means_harmful = collect_means(model, tokenizer, harmful, mlps, device, max_length=args.max_len,
                                  batch_size=args.batch_size)

    # Select neurons via z-score on delta means
    selected = select_safety_neurons(means_harmful, means_benign, z_threshold=args.z_threshold)
    total = sum(len(v) for v in selected.values())
    print(f"[select] Selected {total} safety neurons across {len(selected)} layers (z > {args.z_threshold}).")

    # Register gradient masks to freeze the selected neurons
    handles = freeze_safety_neurons(model, mlps, selected)
    frozen_layers = sum(1 for v in handles.values() if v)
    print(f"[freeze] Registered gradient masks for {frozen_layers} layers.")

    # Save model + tokenizer
    print(f"[save] Saving model and tokenizer to: {args.out_dir}")
    tokenizer.save_pretrained(args.out_dir)
    # Note: gradient hooks aren't stored in the weights; we also save a manifest to re-apply them later.
    model.save_pretrained(args.out_dir)

    # Save manifest
    manifest = FreezeManifest(
        model_name=args.model,
        z_threshold=args.z_threshold,
        layer_to_neurons=selected
    )
    manifest_path = os.path.join(args.out_dir, "freeze_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        f.write(manifest.to_json())
    print(f"[save] Wrote freeze manifest: {manifest_path}")

    #print("\nDone. For later finetuning, load the model and call:")
    #print("  from freeze_safety_neurons import apply_freeze_from_manifest")
    #print(f"  manifest = json.load(open('{manifest_path}'))")
    #print("  apply_freeze_from_manifest(model, manifest)")


if __name__ == "__main__":
    main()
