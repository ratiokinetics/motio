# SPEC

## Setup 

**Dataset**
- Dataset: wikimedia/wikipedia/20231101.en
- Preprocess: keep docs with $\ge 512$ tokens and truncate them to the first 512 tokens to obtain a dataset of (n_docs, 512). `tokenized.jsonl`

**Activations**
- Model: GPT-2 (start small, $d=768$); residual stream
- Activations at layer $\ell$: (n_docs, 512, d) `activations.npy`

## Cluster 

**Preprocess** 
- Drop token at position 0
- Center on the mean of the remaining vectors. Store $\mu$ to keep the transform invertible

**Fit**
- k-means (torch/MPS, ~40 lines, no new deps) on a 1-2M subsample, random over docs *and* positions
- Sweep $k \in \{256, 1024, 4096\}$
- Assign all 6M in one batched pass -> `labels.npy` `(11810, 512)` int16, ~12 MB

**Eval**

- Precision loss (primary): patch centroids back at layer $\ell$ (positions $\ge 1$ only), measure $\mathrm{KL}$ vs original output distribution + loss delta. Baselines: $k=1$ (global mean) = ceiling, random centroid = floor
- Stability: rerun with different random seeds or data samples. Similar activations should remain grouped.
- Cluster usage: size histogram; dead clusters + effective code count (usage perplexity)
- Read the clusters: top tokens + contexts for 20 random clusters. Grab-bags = good KL but useless visualization

## Inference

- Interface should take a prompt as input -> token IDs -> motion visualization 
