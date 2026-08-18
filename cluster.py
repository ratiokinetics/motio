import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from transformers import GPT2LMHeadModel
import json
import torch.nn.functional as F


DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
TRAIN_SAMPLE = 1_000_000
TEST_SAMPLE = 10_000
KL_BATCH_SIZE = 64
K_LIST = [256, 512, 1024, 2048, 4096, 8192]
SEED_LIST = [0, 1, 2, 3]

def sample(acts, n, rng):
    """Draw n random rows from a (docs, pos, d) memmap -> (n, d) float32."""
    n_docs, n_pos, _ = acts.shape
    flat = np.sort(rng.choice(n_docs * n_pos, size=n, replace=False))  # sorted -> sequential-ish reads
    doc, pos = np.divmod(flat, n_pos)
    return acts[doc, pos], doc, pos

def preprocess(x, mu):
    """Center + L2-normalize on device; x: (..., d), mu: (d,) -> (unit vecs, norms)."""
    x = torch.as_tensor(x, device=DEVICE) - torch.as_tensor(mu, dtype=torch.float32, device=DEVICE)
    norm = x.norm(dim=-1, keepdim=True)
    return x / norm, norm

def preprocess_inv(xp, n, mu):
    """Inverse transform: unit vecs * norms + mean -> raw residual-stream vectors."""
    return xp * n + mu

def get_labels(x, C):
    """Nearest-centroid labels. x: unit-norm (..., d), C: (k, d) -> (...); argmax score == argmin ||x-c||^2."""
    return (x @ C.T - (C * C).sum(1) / 2).argmax(-1)

def kmeans(x, k, seed, iters=50, chunk=65_536, tol=1e-4):
    """Lloyd's k-means on unit-norm rows. x: (N, d) -> centroids (k, d)."""
    N = len(x)
    g = torch.Generator().manual_seed(seed)
    pick = lambda n: x[torch.randperm(N, generator=g)[:n].to(x.device)]
    C, prev = pick(k).clone(), float("inf")
    for it in range(iters):
        sums, counts, inertia = torch.zeros_like(C), torch.zeros(k, device=x.device), 0.0
        for i in range(0, N, chunk):  # chunked: score matrix is (chunk, k)
            xc = x[i:i + chunk]
            best, lab = (xc @ C.T - (C * C).sum(1) / 2).max(1)  # argmax score == argmin ||x-c||^2
            inertia += (1 - 2 * best).sum().item()  # ||x-c||^2 = 1 - 2*score; float64, MPS has no double
            sums.index_add_(0, lab, xc)
            counts.index_add_(0, lab, torch.ones_like(lab, dtype=counts.dtype))
        dead = counts == 0
        if dead.any():  # reseed empty clusters on random points
            sums[dead], counts[dead] = pick(int(dead.sum())), 1
        C = sums / counts[:, None]
        print(f"k={k} seed={seed} iter={it} train inertia={inertia:.1f}")
        if prev - inertia < tol * prev:
            break
        prev = inertia
    return C, inertia


def assign(acts, C, mu):
    """Nearest-centroid assignment. acts: raw (..., d) -> labels (...)."""
    x, _ = preprocess(acts, mu)
    C = torch.as_tensor(C, dtype=torch.float32, device=DEVICE)
    return get_labels(x, C)

@torch.no_grad()
def predict(ids, model):
    h = model.transformer(torch.as_tensor(ids, device=DEVICE)).last_hidden_state[:, -1]
    return model.lm_head(h).softmax(-1)

def patch(ids, C, mu, model):
    """Return centroid-patched next-token distributions."""
    mu = torch.as_tensor(mu, dtype=torch.float32, device=DEVICE)
    C = torch.as_tensor(C, dtype=torch.float32, device=DEVICE)

    def hook(_, __, out):  # snap residual stream (pos >= 1) to nearest centroid, map back via saved norm + mu
        h = out.clone()
        xp, n = preprocess(h[:, 1:], mu)
        h[:, 1:] = preprocess_inv(C[get_labels(xp, C)], n, mu)
        return h

    handle = model.transformer.h[model.config.n_layer * 2 // 3 - 1].register_forward_hook(hook)
    try:
        return predict(ids, model)
    finally:
        handle.remove()

def train():
    # 1. Preprocess the activations
    acts = np.load("activations.npy", mmap_mode="r")[:, 1:]  # drop pos 0;
    print("Loaded activations after dropping pos 0: ", acts.shape)
    x, _, _ = sample(acts, TRAIN_SAMPLE, np.random.default_rng(0))
    print("Sampled activations: ", x.shape)
    mu = x.mean(axis=0)
    xp, _ = preprocess(x, mu)
    print("Preprocessed activations: ", xp.shape) # (TRAIN_SAMPLE, d)

    # 2. Run k-means for each k and seed
    out_dir = Path(f"data/kmeans/RUN_{datetime.now():%Y%m%d_%H%M%S}")
    out_dir.mkdir(parents=True)
    np.save(out_dir / "mu.npy", mu)
    for k in K_LIST:
        for seed in SEED_LIST:
            C, inertia = kmeans(xp, k, seed)
            out = out_dir / f"kmeans_k{k}_seed{seed}.npz"
            np.savez(out, k=k, seed=seed, centroids=C.cpu().numpy(), inertia=inertia)
            print(f"k={k} seed={seed} centroids: {C.shape} inertia: {inertia:.1f} -> {out}")

def evaluate(run_dir: Path):
    assert run_dir.is_dir(), f"not a directory: {run_dir}"
    runs = [np.load(f) for f in sorted(run_dir.glob("kmeans_k*_seed*.npz"))]
    
    # 1. Plot the inertia vs k     
    ks = sorted({int(r["k"]) for r in runs})
    elbow_data = [{"k": int(r["k"]), "seed": int(r["seed"]), "inertia": float(r["inertia"])} for r in runs]
    plt.scatter([int(r["k"]) for r in runs], [float(r["inertia"]) for r in runs], alpha=0.5, label="seeds")
    plt.plot(ks, [min(float(r["inertia"]) for r in runs if int(r["k"]) == k) for k in ks], "o-", label="best seed")
    plt.xscale("log", base=2), plt.xticks(ks, ks)
    plt.xlabel("k"), plt.ylabel("inertia"), plt.title(run_dir.name), plt.legend()
    plt.savefig(run_dir / "elbow.png", dpi=150)

    # 2. Get cluster usage histogram
    acts = np.load("activations.npy", mmap_mode="r")[:, 1:]  # drop pos 0
    mu = np.load(run_dir / "mu.npy")
    x_test, doc, pos = sample(acts, TEST_SAMPLE, np.random.default_rng(42))
    seeds = sorted({int(r["seed"]) for r in runs})
    fig, axes = plt.subplots(len(ks), len(seeds), figsize=(3 * len(seeds), 2 * len(ks)), sharey="row", squeeze=False)
    for r in runs:
        k, seed = int(r["k"]), int(r["seed"])
        counts = np.bincount(assign(x_test, r["centroids"], mu).cpu().numpy(), minlength=k)
        ax = axes[ks.index(k), seeds.index(seed)]
        ax.bar(range(k), np.sort(counts)[::-1])
        ax.set_title(f"k={k} seed={seed}", fontsize=8)
    fig.tight_layout()
    fig.savefig(run_dir / "usage.png", dpi=150)

    # 3. KL measurement (+ k=1 baseline: Lloyd's k=1 fixed point is the mean of the preprocessed train sample)
    x_train, _, _ = sample(acts, TRAIN_SAMPLE, np.random.default_rng(0))
    runs.append({"k": 1, "seed": 0, "centroids": preprocess(x_train, mu)[0].mean(0, keepdim=True).cpu().numpy()})
    del x_train

    with open("tokenized.jsonl") as f:
        docs = [json.loads(line)["ids"] for line in f]

    model = GPT2LMHeadModel.from_pretrained("gpt2").eval().to(DEVICE)

    kl_sums = np.zeros(len(runs))
    for p in np.unique(pos):
        group = doc[pos == p]
        for i in range(0, len(group), KL_BATCH_SIZE):
            ids = [docs[int(d)][:int(p) + 2] for d in group[i:i + KL_BATCH_SIZE]]
            original = predict(ids, model)
            for j, r in enumerate(runs):
                patched = patch(ids, r["centroids"], mu, model)
                kl_sums[j] += F.kl_div(
                    patched.clamp_min(1e-12).log(),
                    original,
                    reduction="none",
                ).sum().item()

    kls = kl_sums / len(doc)
    kl_data = [{"k": int(r["k"]), "seed": int(r["seed"]), "kl": float(v)} for r, v in zip(runs, kls)]
    (run_dir / "eval.json").write_text(json.dumps({"elbow": elbow_data, "kl": kl_data}, indent=2))

    plt.figure()
    ks = sorted({int(r["k"]) for r in runs})
    plt.scatter([int(r["k"]) for r in runs], kls, alpha=0.5, label="seeds")
    plt.plot(ks, [min(kl for r, kl in zip(runs, kls) if int(r["k"]) == k) for k in ks], "o-", label="best seed")
    plt.xscale("log", base=2), plt.xticks(ks, ks)
    plt.xlabel("k"), plt.ylabel("KL"), plt.title(run_dir.name), plt.legend()
    plt.savefig(run_dir / "kl.png", dpi=150)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="mode", required=True)
    sub.add_parser("train")
    sub.add_parser("eval").add_argument("run_dir", type=Path, help="data folder from a train run, e.g. data/kmeans/RUN_...")
    args = p.parse_args()
    train() if args.mode == "train" else evaluate(args.run_dir)