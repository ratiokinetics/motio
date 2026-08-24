"""tokenize → activations → recursive k-means → assign."""
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sae_lens import PretokenizeRunner, PretokenizeRunnerConfig
from sklearn.cluster import KMeans
from transformers import GPT2Model, AutoTokenizer

# --- hyperparams ---
DATASET, SIZE = "NeelNanda/c4-10k", 1_000
MODEL, CONTEXT_SIZE, LAYER = "gpt2", 512, 10
TOKENIZED, ACTS, ASSIGNED = "tokenized", "activations.npy", "assigned.jsonl"
TRAIN_SAMPLE, N_INIT, START, END = 100_000, 1, 6, 10
FWD_BATCH, ASSIGN_BATCH = 32, 256
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

out = Path(f"data/RUN_{datetime.now():%Y%m%d_%H%M%S}")
out.mkdir(parents=True)

def center_norm(x, mu):
    x = np.asarray(x, dtype=np.float32) - mu
    return x / np.linalg.norm(x, axis=-1, keepdims=True)

def fit(x, k, verbose=0):
    km = KMeans(k, n_init=N_INIT, random_state=0, verbose=verbose).fit(x) # sklearn keeps the run with lowest inertia across seeds
    return km.cluster_centers_, km.labels_, km.inertia_

def save_depth(depth, C, nodes, codes, inertia):
    sizes = np.array([len(n) for n in nodes])
    np.savez(out / f"depth_{depth:02}.npz", centroids=C, counts=sizes,
             codes=np.array([format(c, f"0{depth}b") for c in codes]), inertia=inertia)
    print(f"depth={depth} leaves={len(nodes)} inertia={inertia:.1f} min={sizes.min()} max={sizes.max()}")

# 1–2. Load + tokenize (one sequence per doc: prepend BOS, drop shorts, truncate to CONTEXT_SIZE)
cfg = PretokenizeRunnerConfig(
    dataset_path=DATASET, tokenizer_name=MODEL, split=f"train[:{SIZE}]",
    context_size=CONTEXT_SIZE, disable_concat_sequences=True,
    begin_batch_token="bos", begin_sequence_token=None, sequence_separator_token=None, # Modify for different models
    shuffle=True, seed=42, num_proc=8, save_path=str(out / TOKENIZED),
)

dataset = PretokenizeRunner(cfg).run()
ids = np.stack([np.asarray(r, dtype=np.int64) for r in dataset["input_ids"]])
print(f"Number of rows: {len(ids)}")
print(f"Length of each row: {len(ids[0])}")

# 3. Activations at LAYER
model = GPT2Model.from_pretrained(MODEL).eval().to(DEVICE)
d_model = model.config.n_embd
acts = np.lib.format.open_memmap(out / ACTS, mode="w+", dtype=np.float32, shape=(len(ids), CONTEXT_SIZE, d_model))
with torch.no_grad():
    for i in range(0, len(ids), FWD_BATCH):
        print(f"acts {i}/{len(ids)}")
        batch = torch.tensor(ids[i:i + FWD_BATCH], device=DEVICE)
        # hidden_states[0]  → token embeddings ... hidden_states[1]  → after block 0 ... hidden_states[LAYER] → after block LAYER - 1
        acts[i:i + len(batch)] = model(batch, output_hidden_states=True).hidden_states[LAYER].cpu().numpy()
acts.flush()
print("acts:", acts.shape)

# 4. Train recursive binary k-means on a sample of activations ()
flat = acts[:, 1:] # drop pos 0
n_docs, n_pos, d = flat.shape
print("flat:", flat.shape)
pick = np.sort(np.random.default_rng(0).choice(n_docs * n_pos, size=min(TRAIN_SAMPLE, n_docs * n_pos), replace=False))
print("pick:", pick.shape)
doc, pos = np.divmod(pick, n_pos)
x = flat[doc, pos]
print("x:", x.shape)
mu, xp = x.mean(0), center_norm(x, x.mean(0))
np.save(out / "mu.npy", mu)

print(f"Running flat start k-means at k={2**START} ... This may take a while...")
C, labels, inertia = fit(xp, 2**START, verbose=1)
nodes = [np.nonzero(labels == i)[0] for i in range(2**START)]
codes = list(range(2**START))
save_depth(START, C, nodes, codes, inertia)

for depth in range(START + 1, END + 1):
    child_nodes, child_codes, cents, depth_inertia = [], [], [], 0.0
    for code, ix in zip(codes, nodes):
        if len(ix) == 1:   # single-element leaf: carry through unchanged
            child_nodes.append(ix); 
            child_codes.append(code << 1); # add as "<code>0" 
            cents.append(xp[ix].mean(0, keepdims=True)); 
            continue
        C, labels, inertia = fit(xp[ix], 2)
        child_nodes += [ix[labels == 0], ix[labels == 1]]
        child_codes += [code << 1, (code << 1) | 1] # add as "<code>0" and "<code>1";
        cents.append(C); 
        depth_inertia += inertia
    nodes, codes = child_nodes, child_codes
    save_depth(depth, np.concatenate(cents), nodes, codes, depth_inertia)

# 5. Assign every token (except pos 0) to nearest leaf at the deepest level
final = np.load(out / f"depth_{END:02}.npz")
centroids, bcodes, c2 = final["centroids"], final["codes"], np.sum(final["centroids"] ** 2, axis=1)
print("centroids:", centroids.shape)
print("bcodes:", bcodes.shape)
print("c2:", c2.shape)
with open(out / ASSIGNED, "w") as f:
    for i in range(0, len(ids), ASSIGN_BATCH):
        print(f"assign {i}/{len(ids)}")
        B = len(ids[i:i + ASSIGN_BATCH])
        a = center_norm(acts[i:i + B], mu).reshape(-1, d)
        # For unit vectors: argmin ||a-c||^2 = ||a||^2 - 2a*c + ||c||^2 = 1 - 2a*c + ||c||^2
        print("a.shape:", a.shape)
        idx = np.argmin(1 - 2 * (a @ centroids.T) + c2, axis=-1).reshape(B, CONTEXT_SIZE)
        for j in range(B):
            f.write(json.dumps({"cluster_ids": [None] + [str(bcodes[idx[j, p]]) for p in range(1, CONTEXT_SIZE)]}) + "\n")
print("run:", out)
