import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
N_SAMPLE = 1_000_000
K_LIST = [8, 16, 32]
SEED_LIST = [0, 1, 2, 3]

def sample(acts, n, rng):
    """Draw n random rows from a (docs, pos, d) memmap -> (n, d) float32."""
    n_docs, n_pos, _ = acts.shape
    flat = np.sort(rng.choice(n_docs * n_pos, size=n, replace=False))  # sorted -> sequential-ish reads
    doc, pos = np.divmod(flat, n_pos)
    return acts[doc, pos]

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

def patch_demo(ids, C, mu, top=10):
    """Two forward passes over one sequence of token ids: normal vs centroid-patched at layer l.
    Prints top next-token probs for both; returns the two full distributions."""
    model = GPT2LMHeadModel.from_pretrained("gpt2").eval().to(DEVICE)
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    N_LAYERS = model.config.n_layer  # 12 for gpt2
    LAYER = N_LAYERS * 2 // 3
    mu = torch.as_tensor(mu, dtype=torch.float32, device=DEVICE)
    C = torch.as_tensor(C, dtype=torch.float32, device=DEVICE)
    ids = torch.as_tensor(ids, device=DEVICE)[None]  # (1, T)

    def hook(_, __, out):  # snap residual stream (pos >= 1) to nearest centroid, map back via saved norm + mu
        h = out  # residual stream (B, T, d) 
        xp, n = preprocess(h[:, 1:], mu)
        h[:, 1:] = preprocess_inv(C[get_labels(xp, C)], n, mu)

    with torch.no_grad():
        p = model(ids).logits[0, -1].softmax(-1)
        model.transformer.h[LAYER - 1].register_forward_hook(hook)  # h[LAYER-1] output == hidden_states[LAYER]
        q = model(ids).logits[0, -1].softmax(-1)
    for name, d in (("normal ", p), ("patched", q)):
        print(name, ", ".join(f"{tok.decode([int(i)])!r} {v:.3f}" for v, i in zip(*d.topk(top))))
    return p, q

def train():
    # 1. Preprocess the activations
    acts = np.load("activations.npy", mmap_mode="r")[:, 1:]  # drop pos 0;
    print("Loaded activations after dropping pos 0: ", acts.shape)
    x = sample(acts, N_SAMPLE, np.random.default_rng(0))
    print("Sampled activations: ", x.shape)
    mu = x.mean(axis=0)
    xp, _ = preprocess(x, mu)
    print("Preprocessed activations: ", xp.shape) # (N_SAMPLE, d)

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
    plt.scatter([int(r["k"]) for r in runs], [float(r["inertia"]) for r in runs], alpha=0.5, label="seeds")
    plt.plot(ks, [min(float(r["inertia"]) for r in runs if int(r["k"]) == k) for k in ks], "o-", label="best seed")
    plt.xscale("log", base=2), plt.xticks(ks, ks)
    plt.xlabel("k"), plt.ylabel("inertia"), plt.title(run_dir.name), plt.legend()
    plt.savefig(run_dir / "elbow.png", dpi=150)

    # 2. Get cluster usage histogram
    acts = np.load("activations.npy", mmap_mode="r")[:, 1:]  # drop pos 0
    mu = np.load(run_dir / "mu.npy")
    x_test = sample(acts, 100_000, np.random.default_rng(42))
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

    # 3. KL measurement
    ids = [2025,998,1042,318,257,1964,8876,290,3356,326,318,17988,286,477,655,6637,329,4934,290,12932,284,35531,262,6712,340,3667,5529,13114,32000,290,18911,11,6032,1390,3277,12,27219,11,290,12129,13,32229,1042,11009,329,262,9014,286,262,1181,351,1181,1203,14515,290,16171,1479,15814,13,1081,257,15074,1364,12,5469,3356,11,428,3555,286,41661,318,4624,319,262,15189,3634,1364,286,262,1964,10958,11,3221,3417,355,262,19466,8539,286,262,15889,3356,357,33203,14012,19803,737,198,198,32661,504,423,5615,287,14515,1231,8766,28398,444,890,878,262,9323,286,2585,11,35423,11,393,44982,13,2080,262,4485,286,20325,38958,5920,11,30186,11965,3812,4934,635,8278,13,4900,20675,286,26177,4213,389,1043,477,3690,2106,11,3660,41661,9349,422,262,39057,13,5856,262,6846,2063,286,262,678,400,290,262,717,4647,286,262,1160,400,4289,11,262,26177,3356,45671,287,749,3354,286,262,995,290,550,257,2383,2597,287,3259,6,12766,329,48936,13,26386,26177,4266,286,1807,7042,1141,428,2278,13,32229,1023,423,2077,636,287,1811,37888,11,749,14660,287,262,6342,1520,1726,11,262,3394,7511,1810,290,262,7897,7511,1810,11,3025,886,7498,262,886,286,262,15993,6980,286,41661,13,554,262,938,4647,286,262,1160,400,290,656,262,2310,301,4289,11,262,26177,3356,468,587,33316,298,1752,517,11,3957,287,11533,290,4588,1626,3098,12,49970,11,3098,12,5767,290,3098,12,20541,5612,8650,13,198,198,2025,998,1023,1873,10084,10581,11,543,743,307,4143,9086,656,12253,290,16673,10064,26,612,318,2383,21721,1022,262,734,13,15815,560,5050,1949,284,29308,644,281,26177,3592,1244,307,588,11,475,12253,10815,11,543,423,15074,2077,257,6590,1210,11,4031,284,25525,4934,290,262,1181,13,4650,44497,286,1692,14355,423,587,12824,416,26177,4583,11,19976,11,290,7201,87,271,13,198,198,36,43408,11,29191,11,290,6770,220,198,198,464,304,774,76,2770,8159,286,41661,318,422,262,13406,8312,281,668,71,544,11,3616,366,19419,257,22740,1600,13160,286,262,21231,281,12,5855,19419,4943,290,262,1573,610,14636,418,5855,27940,1,393,366,81,18173,11074,383,35488,532,1042,43397,262,15735,1459,326,2090,4662,42856,13,32229,1042,3568,287,3594,422,1467,3682,355,41661,68,290,42856,422,1315,2670,26,1903,3594,514,1095,6661,1417,257,2565,286,8967,13,26386,18783,1626,262,4141,9303,30538,511,7691,355,29676,11,3584,1178,884,5371,4888,867,5009,351,1568,29676,13,4650,44347,286,262,678,400,4289,884,355,3977,1793,5404,357,1558,3980,1906,1507,2623,8]
    patch_demo(ids, runs[0]["centroids"], mu)

    
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="mode", required=True)
    sub.add_parser("train")
    sub.add_parser("eval").add_argument("run_dir", type=Path, help="data folder from a train run, e.g. data/kmeans/RUN_...")
    args = p.parse_args()
    train() if args.mode == "train" else evaluate(args.run_dir)