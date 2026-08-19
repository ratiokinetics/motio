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

### Explore

```
python -c "from transformers import GPT2TokenizerFast; import json; json.dump(GPT2TokenizerFast.from_pretrained('gpt2').get_vocab(), open('vocab.json','w'))"
python -m http.server
```

Open http://localhost:8000/interface.html and enter a sequence of cluster codes (6–10 bits each, space separated). Shorter codes prefix-match deeper clusters; matching tokens are highlighted in the decoded text.