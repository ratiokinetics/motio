import numpy as np
from datetime import datetime
from pathlib import Path
from sklearn.cluster import KMeans

TRAIN_SAMPLE = 1_000_000
N_INIT = 9  # k-means++ restarts per fit, best kept by inertia
START_LEVEL_DEPTH = 6
END_LEVEL_DEPTH = 12

def sample(acts, n, rng):
    """Draw n random rows from a (docs, pos, d) memmap -> (n, d) float32."""
    n_docs, n_pos, _ = acts.shape
    flat = np.sort(rng.choice(n_docs * n_pos, size=n, replace=False))  # sorted -> sequential-ish reads
    doc, pos = np.divmod(flat, n_pos)
    return acts[doc, pos]

def preprocess(x, mu):
    """Center + L2-normalize; x: (..., d), mu: (d,) -> unit vecs."""
    x = np.asarray(x, dtype=np.float32) - mu
    return x / np.linalg.norm(x, axis=-1, keepdims=True)

def fit(x, k):
    """Best-of-N_INIT k-means++. x: (n, d) -> (centroids, labels, inertia)"""
    km = KMeans(k, n_init=N_INIT, random_state=0).fit(x)
    return km.cluster_centers_, km.labels_, km.inertia_

def save_depth(out_dir, depth, C, nodes, inertia):
    sizes = np.array([len(n) for n in nodes])
    np.savez(out_dir / f"depth_{depth:02}.npz", centroids=C, counts=sizes, inertia=inertia)
    print(
        f"depth={depth} leaves={len(nodes)} inertia={inertia:.1f} "
        f"min_size={sizes.min()} median_size={np.median(sizes):.0f} max_size={sizes.max()}"
    )

def train():
    # 1. Preprocess the activations
    acts = np.load("activations.npy", mmap_mode="r")[:, 1:]  # drop pos 0;
    print("Loaded activations after dropping pos 0: ", acts.shape)
    x = sample(acts, TRAIN_SAMPLE, np.random.default_rng(0))
    print("Sampled activations: ", x.shape)
    mu = x.mean(axis=0)
    xp = preprocess(x, mu)
    print("Preprocessed activations: ", xp.shape)  # (TRAIN_SAMPLE, d)

    # 2. Run recursive binary k-means
    out_dir = Path(f"data/kmeans/RUN_{datetime.now():%Y%m%d_%H%M%S}")
    out_dir.mkdir(parents=True)
    np.save(out_dir / "mu.npy", mu)

    # --- flat start at k=2**START_LEVEL_DEPTH ---
    C, labels, inertia = fit(xp, 2**START_LEVEL_DEPTH)
    nodes = [np.flatnonzero(labels == i) for i in range(2**START_LEVEL_DEPTH)]
    save_depth(out_dir, START_LEVEL_DEPTH, C, nodes, inertia)

    # --- binary recursion: each leaf -> 2 children ---
    for depth in range(START_LEVEL_DEPTH + 1, END_LEVEL_DEPTH + 1):
        children, depth_centroids, depth_inertia = [], [], 0.0
        for idxs in nodes:
            if len(idxs) < 2:  # unsplittable leaf: carry through unchanged (singleton inertia = 0)
                print(f"depth={depth} unsplittable leaf at index {idxs} carrying {len(idxs)} elements")
                children.append(idxs)
                depth_centroids.append(xp[idxs].mean(0, keepdims=True))
                continue
            C, labels, inertia = fit(xp[idxs], 2)  # always k=2
            children += [idxs[labels == 0], idxs[labels == 1]]
            depth_centroids.append(C)
            depth_inertia += inertia
        nodes = children
        save_depth(out_dir, depth, np.concatenate(depth_centroids), nodes, depth_inertia)

if __name__ == "__main__":
    train()
