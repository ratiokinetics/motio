"""Packs run artifacts into data.js so interface.html runs fully static (file://, no server)."""
import base64, gzip, json
from pathlib import Path
import numpy as np
from datasets import load_from_disk
from transformers import AutoTokenizer

RUN_DIR = Path("data/RUN_20260825_081438")
TOKENIZER_NAME = "Qwen/Qwen2.5-7B-Instruct"

def _cached(cache, build):
    cache = Path(cache)
    if cache.exists():
        return np.load(cache)
    arr = build()
    np.save(cache, arr)
    return arr

def _token_ids():
    ds = load_from_disk(RUN_DIR / "tokenized")
    return np.stack([np.asarray(x, dtype=np.int32) for x in ds["input_ids"]])

def _leaf_codes():
    rows = []
    with open(RUN_DIR / "assigned.jsonl") as f:
        for line in f:
            ids = json.loads(line)["cluster_ids"]
            rows.append([int(c, 2) for c in ids[1:]])
    return np.array(rows, dtype=np.uint16)

def _scores():
    rows = []
    with open(RUN_DIR / "assigned.jsonl") as f:
        for line in f:
            rows.append(json.loads(line)["scores"][1:])
    return np.rint(np.array(rows, dtype=np.float32) * 255).astype(np.uint8)

TOKEN_IDS = _cached(RUN_DIR / "token_ids.npy", _token_ids)    # (N, 512)
LEAF_CODES = _cached(RUN_DIR / "leaf_codes.npy", _leaf_codes) # (N, 511); col j <-> token pos j+1
SCORES = _cached(RUN_DIR / "scores.npy", _scores)             # (N, 511, 6); depths 6..11, uint8 = round(score*255)

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
vocab, inv = np.unique(TOKEN_IDS, return_inverse=True)        # compact vocab of used tokens
texts = tokenizer.batch_decode([[int(i)] for i in vocab])
vocab_b = json.dumps(texts, ensure_ascii=False).encode()
vocab_b += b" " * (-len(vocab_b) % 4)                         # 4-byte align the arrays that follow
payload = (np.array([len(vocab_b), len(TOKEN_IDS)], "<u4").tobytes() + vocab_b +
           inv.astype("<u4").tobytes() + LEAF_CODES.astype("<u2").tobytes() + SCORES.tobytes())
b64 = base64.b64encode(gzip.compress(payload, 9)).decode()
with open("data.js", "w") as f:
    f.write("window.MOTIO_B64=" + json.dumps(b64))
print(f"data.js {len(b64)/1e6:.1f} MB (raw payload {len(payload)/1e6:.1f} MB)")
