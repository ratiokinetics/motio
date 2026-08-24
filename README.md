# motio

```
pip install -r requirements.txt
```

### Preprocess

```
python preprocess.py
```

1. Download HF Dataset: wikimedia/wikipedia/20231101.en
2. Preprocess it: keep docs with $\ge 512$ tokens and truncate them to the first 512 tokens to obtain a dataset of (n_docs, 512). `tokenized.jsonl`
3. Pass each sequence through GPT-2 ($d=768$) and extract activations at layer $\ell = N_LAYERS * 2 // 3$ to obtain `activations.npy` (n_docs, 512, d) 

### Train

```
python train.py
```

1. Load activations from the previous step (drop token at position 0)
2. Sample #TRAIN_SAMPLE of them 
3. Preprocess data -> subtract the mean and L2-normalize to unit vectors
4. Calculate k-means at k=2^START_LEVEL_DEPTH then recursively bisect until 2^END_LEVEL_DEPTH leaves 
5. Save per-depth centroids, leaf sizes, and total inertia to data/kmeans/RUN_TIMESTAMP/depth_XX.npz

### Postprocess

```
python postprocess.py
```

### Interface

```
python server.py
```

Boots run artifacts into RAM and serves http://localhost:8000/interface.html. Enter a sequence of cluster codes (6–10 bits each) by typing them space-separated. Shorter codes prefix-match deeper clusters; matching docs appear on the right.

### TODO

- Scale: bigger model and SAE dataset
- Account for dead branches in interface
- Attach Activation verbalizer
- Add matrix heatmap to show interesting and recurring patterns
- Graphic step to display a sentence
- Score to measure how close to cluster some sequence is
- Allow to user enter to enter a sentence 
- min 2 codes to be shown
- better colours
- Add interesting codes that I discovered 
- Autointerpretability. 