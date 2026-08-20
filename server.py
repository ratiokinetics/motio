"""motio server — boots run artifacts into RAM, serves /tree, /search, /docs/{id} + static files."""
import json
import re
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
from transformers import GPT2Model, GPT2TokenizerFast

RUN_DIR = Path("data/kmeans/RUN_20260819_103855")
START, END, SEQ = 6, 10, 512
MAX_HITS, CTX = 200, 25
PREFIX_RE = re.compile(rf"[01]{{{START},{END}}}")

# ---------- boot ----------

def _cached(cache, build):
    cache = Path(cache)
    if cache.exists():
        return np.load(cache)
    arr = build()
    np.save(cache, arr)
    return arr

def _rows(path, field, f):
    return np.array([f(json.loads(l)[field]) for l in Path(path).read_text().splitlines()], dtype=np.uint16)

print("boot: token_ids + leaf_codes")
TOKEN_IDS = _cached("data/token_ids.npy", lambda: _rows("tokenized.jsonl", "ids", lambda r: r))  # (N, 512)
# col j <-> token pos j+1 (pos 0 is never assigned)
LEAF_CODES = _cached("data/leaf_codes.npy", lambda: _rows("assigned.jsonl", "cluster_ids", lambda r: [int(c, 2) for c in r[1:]]))  # (N, 511)

print("boot: tokenizer + gpt-2")
TOKENIZER = GPT2TokenizerFast.from_pretrained("gpt2")
MODEL = GPT2Model.from_pretrained("gpt2").eval()  # not used by current endpoints; kept hot per SPECS
TOK_TEXT = TOKENIZER.batch_decode([[i] for i in range(len(TOKENIZER))])  # per-token text lookup

print("boot: kmeans artifacts")
MU = np.load(RUN_DIR / "mu.npy")
LEAF_CENTROIDS = np.load(RUN_DIR / f"depth_{END}.npz")["centroids"]
DEPTH_CODES = {d: set(np.load(RUN_DIR / f"depth_{d:02}.npz")["codes"].tolist()) for d in range(START, END + 1)}

def build_tree():
    """Every node at depths START..END; descendants of dead nodes (carried chains) are not real splits."""
    nodes = []
    def walk(code, parent):
        d = len(code)
        kids = [code + "0", code + "1"]
        alive = d < END and all(k in DEPTH_CODES[d + 1] for k in kids)
        nodes.append({"code": code, "depth": d, "parent": parent,
                      "children": kids if alive else [], "dead": d < END and not alive})
        if alive:
            for k in kids:
                walk(k, code)
    for c in sorted(DEPTH_CODES[START]):
        walk(c, None)
    return {"start": START, "end": END, "nodes": nodes}

TREE = build_tree()

# ---------- endpoints ----------

def search(prefixes):
    W = LEAF_CODES.shape[1]
    m = len(prefixes)
    if W - m + 1 <= 0:
        return {"total": 0, "hits": []}
    hit = np.ones((len(LEAF_CODES), W - m + 1), dtype=bool)
    for j, p in enumerate(prefixes):
        mask = (LEAF_CODES >> (END - len(p))) == int(p, 2)
        hit &= mask[:, j:W - m + 1 + j]
    docs, cols = np.nonzero(hit)

    hits = []
    for d, c in zip(docs[:MAX_HITS].tolist(), cols[:MAX_HITS].tolist()):
        start, end = c + 1, c + m
        ws, we = max(1, start - CTX), min(SEQ - 1, end + CTX)  # ±CTX tokens of context, pos 0 excluded
        toks = [{"text": TOK_TEXT[TOKEN_IDS[d, p]],
                 "code": format(int(LEAF_CODES[d, p - 1]), f"0{END}b"),
                 "match": start <= p <= end} for p in range(ws, we + 1)]
        hits.append({"doc_id": d, "start": start, "end": end, "ctx_start": ws, "tokens": toks})
    return {"total": int(len(docs)), "hits": hits}

def doc(i):
    return {"doc_id": i, "tokens": [
        {"text": TOK_TEXT[TOKEN_IDS[i, p]],
         "code": format(int(LEAF_CODES[i, p - 1]), f"0{END}b") if p else None}
        for p in range(SEQ)]}

# ---------- http ----------

class Handler(SimpleHTTPRequestHandler):
    def _json(self, obj, status=200):
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/tree":
            return self._json(TREE)
        if m := re.fullmatch(r"/docs/(\d+)", self.path):
            i = int(m[1])
            return self._json(doc(i)) if i < len(TOKEN_IDS) else self._json({"error": "no such doc"}, 404)
        if self.path == "/":
            self.path = "/interface.html"
        super().do_GET()

    def do_POST(self):
        if self.path != "/search":
            return self._json({"error": "not found"}, 404)
        try:
            prefixes = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)))["prefixes"]
            assert prefixes and all(isinstance(p, str) and PREFIX_RE.fullmatch(p) for p in prefixes)
        except Exception:
            return self._json({"error": f"prefixes must be a non-empty list of {START}-{END} bit binary strings"}, 400)
        self._json(search(prefixes))

if __name__ == "__main__":
    print(f"{len(TOKEN_IDS)} docs · {len(TREE['nodes'])} tree nodes · http://localhost:8000/interface.html")
    ThreadingHTTPServer(("", 8000), partial(Handler, directory=str(Path(__file__).parent))).serve_forever()
