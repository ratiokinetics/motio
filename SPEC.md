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
- L2 Normalization

**Fit**
- k-means (torch/MPS, ~40 lines, no new deps) on a 1-2M subsample, random over docs *and* positions
- Sweep $k$ and seeds

**Eval**

- Plot inertia for different k, seeds and preprocessing combinations
- Cluster usage: size histogram;
- Precision loss (primary): patch centroids back at layer $\ell$ (positions $\ge 1$ only), measure $\mathrm{KL}$ vs original output distribution. Consider as baselines: a. no patch (KL=0) and b. global mean (k=1)

## Lang

- (inference) Interface should take a prompt as input -> token IDs -> motion visualization in a 2D grid.
- 1st Eval: build the k × k matrix heatmap of how often cluster i is followed by cluster j.
- 2nd Eval: simple semantic or synactic combos
