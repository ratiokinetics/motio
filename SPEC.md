# SPEC

## Setup 

**Dataset**
- Dataset: wikimedia/wikipedia/20231101.en
- Preprocess: keep docs with $\ge 512$ tokens and truncate them to the first 512 tokens to obtain a dataset of (n_docs, 512). 

**Activations**
- Model: GPT-2 (start small, $d=768$); residual stream
- Activations at layer $\ell$: (n_docs, 512, d)

## Train 

- Cluster

## Inference

- Motion across cluster, visualize 2D, symbolic interpretaiton

### Eval 

- Precision loss