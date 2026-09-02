"""Patch: replace block-20 activation at (doc, pos) with a cluster centroid; compare continuations."""
import argparse
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

RUN_DIR = Path("data/RUN_20260825_081438")
MODEL, TARGET_BLOCK = "Qwen/Qwen2.5-7B-Instruct", 20
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

ap = argparse.ArgumentParser()
ap.add_argument("doc", type=int)
ap.add_argument("pos", type=int, help="token position in doc, >= 1")
ap.add_argument("cluster", help="binary code, 6-11 bits, as shown in the UI")
ap.add_argument("--new-tokens", type=int, default=40)
args = ap.parse_args()

z = np.load(RUN_DIR / f"depth_{len(args.cluster):02}.npz")
c = z["centroids"][np.flatnonzero(z["codes"] == args.cluster)[0]]
mu = np.load(RUN_DIR / "mu.npy")

ids = np.load(RUN_DIR / "token_ids.npy", mmap_mode="r")[args.doc, :args.pos + 1]
ids = torch.tensor(ids, dtype=torch.long, device=DEVICE)[None]

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).eval().to(DEVICE)
MU = torch.tensor(mu, device=DEVICE)
CDIR = torch.tensor(c / np.linalg.norm(c), device=DEVICE)

patch = False
def hook(_, __, out):
    h = out[0] if isinstance(out, tuple) else out
    if patch and h.shape[1] > 1:  # prefill only; patched KV persists through decoding
        v = h[0, args.pos].float()
        h[0, args.pos] = (MU + (v - MU).norm() * CDIR).to(h.dtype)  # keep norm, swap direction

model.model.layers[TARGET_BLOCK].register_forward_hook(hook)

def continuation():
    with torch.no_grad():
        return tok.decode(model.generate(ids, max_new_tokens=args.new_tokens,
                                         do_sample=False)[0, ids.shape[1]:])

print(f"prompt: {tok.decode(ids[0])!r}")
print("\n=== baseline ===", continuation(), sep="\n")
patch = True
print("\n=== patched ===", continuation(), sep="\n")
