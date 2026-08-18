# motio

```
pip install -r requirements.txt
```

### Dataset

```
python setup.py
```

1. Download HF Dataset: wikimedia/wikipedia/20231101.en
2. Preprocess it: keep docs with $\ge 512$ tokens and truncate them to the first 512 tokens to obtain a dataset of (n_docs, 512). `tokenized.jsonl`
3. Pass each sequence through GPT-2 ($d=768$) and extract activations at layer $\ell = N_LAYERS * 2 // 3$ to obtain `activations.npy` (n_docs, 512, d) 


