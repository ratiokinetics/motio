# motio

```
pip install -r requirements.txt
```

### Train

```
python train.py
```

### Interface

```
python frontend/server.py
```

Boots run artifacts into RAM and serves http://localhost:8000

### Eval

Requires:
- Train run and server booted for the same run
- Fetch SAE dataset `qwen2.5-7b-it/20-matryoshka-65k` via `aws s3 sync s3://neuronpedia-datasets/v1/qwen2.5-7b-it/20-matryoshka-65k/ ./data/qwen2.5-7b-it/20-matryoshka-65k --no-sign-request`
- `OPENROUTER_API_KEY` environment variable

```
export OPENROUTER_API_KEY=...
python eval.py
```

### Patch

```
python patch.py <doc> <pos> <cluster_id>
```

### Diagram

```
python diagram.py mock/diagram.json
python patch_diagram.py mock/patch_diagram.json
```