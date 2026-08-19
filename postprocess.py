"""
Assign each token (excluding position 0) to its leaf cluster for a given run,
and write a JSONL file mirroring tokenized.jsonl but with binary cluster codes.
"""
import json
import numpy as np
from pathlib import Path
from train import preprocess

def load_leaf_clusters(run_dir: Path):
    """Return (centroids, codes_str) from the deepest depth_*.npz in run_dir."""
    npz_files = sorted(run_dir.glob("depth_*.npz"))
    d = np.load(npz_files[-1])
    return d["centroids"], d["codes"]  # codes are binary strings e.g. '0001101'

def assign_batch(acts_batch, mu, centroids):
    """Assign a batch of activations (B, T, d) to nearest centroids; returns (B, T) int indices."""
    B, T, d = acts_batch.shape
    x = preprocess(acts_batch, mu).reshape(B * T, d)            # (B*T, d), unit vecs
    # For unit vectors: argmin ||x-c||^2 = ||x||^2 - 2x*c + ||c||^2 = 1 - 2x*c + ||c||^2
    d2 = 1 - 2 * (x @ centroids.T) + np.sum(centroids**2, axis=1)   # (B*T, K)
    return np.argmin(d2, axis=-1).reshape(B, T)                      # (B, T)

def postprocess(run_dir: Path, acts_path="activations.npy", tokens_path="tokenized.jsonl",
                out_path="assigned.jsonl", batch_size=256):

    acts = np.load(acts_path, mmap_mode="r")          # (n_docs, 512, d)
    mu = np.load(run_dir / "mu.npy")
    centroids, codes = load_leaf_clusters(run_dir)

    lines = Path(tokens_path).read_text().splitlines()
    n_docs = len(lines)

    with open(out_path, "w") as fout:
        for start in range(0, n_docs, batch_size):
            print(f"Processing batch {start} to {start + batch_size}")
            batch_lines = lines[start:start + batch_size]
            batch_acts = acts[start:start + len(batch_lines)]          # (B, 512, d) — one memmap read
            indices = assign_batch(batch_acts, mu, centroids)           # (B, 512)

            for i, line in enumerate(batch_lines):
                token_ids = json.loads(line)["ids"]
                cluster_codes = [None] + [str(codes[indices[i, pos]]) for pos in range(1, len(token_ids))]
                fout.write(json.dumps({"cluster_ids": cluster_codes}) + "\n")

            if start % (batch_size * 10) == 0:
                print(f"processed {start}/{n_docs} docs")

    print(f"Written to {out_path}")

RUN_DIR = Path("data/kmeans/RUN_20260819_103855")
postprocess(RUN_DIR)
