"""Boot run artifacts into RAM; serve /tree, /search, /docs/{id} + static UI."""
import json, re
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import numpy as np

HERE, ROOT = Path(__file__).resolve().parent, Path(__file__).resolve().parent.parent
RUN_DIR = ROOT / "data" / "RUN_20260825_081438"
TOKENIZER_NAME = "Qwen/Qwen2.5-7B-Instruct"
START, END, SEQ, MAX_HITS, CTX, DEPS = 6, 11, 512, 200, 25, 6
PREFIX_RE = re.compile(rf"[01]{{{START},{END}}}")


def _cached(path, build):
    path = Path(path)
    if path.exists():
        return np.load(path)
    arr = build()
    np.save(path, arr)
    return arr


def _token_ids():
    from datasets import load_from_disk
    ds = load_from_disk(RUN_DIR / "tokenized")
    return np.stack([np.asarray(x, dtype=np.int32) for x in ds["input_ids"]])


def _assigned():
    lc, sc = RUN_DIR / "leaf_codes.npy", RUN_DIR / "scores.npy"
    if lc.exists() and sc.exists():
        return np.load(lc), np.load(sc)
    codes, scores = [], []
    with open(RUN_DIR / "assigned.jsonl") as f:
        for line in f:
            r = json.loads(line)
            codes.append([int(c, 2) for c in r["cluster_ids"][1:]])
            scores.append(r["scores"][1:])
    codes = np.array(codes, np.uint16)
    scores = np.rint(np.array(scores, np.float32) * 255).astype(np.uint8)
    np.save(lc, codes); np.save(sc, scores)
    return codes, scores


print("boot: arrays")
TOKEN_IDS = _cached(RUN_DIR / "token_ids.npy", _token_ids)          # (N, 512)
LEAF_CODES, SCORES = _assigned()                                    # (N, 511), (N, 511, 6)
uniq, inv = np.unique(TOKEN_IDS, return_inverse=True)
vpath = RUN_DIR / "vocab.json"
if vpath.exists():
    VOCAB = json.loads(vpath.read_text())
else:
    print("boot: tokenizer")
    from transformers import AutoTokenizer
    VOCAB = AutoTokenizer.from_pretrained(TOKENIZER_NAME).batch_decode([[int(i)] for i in uniq])
    vpath.write_text(json.dumps(VOCAB, ensure_ascii=False))
TOKEN_IDS = inv.astype(np.uint32).reshape(TOKEN_IDS.shape)          # compact ids into VOCAB


def toks(d, ws, we):
    return [{"text": VOCAB[TOKEN_IDS[d, p]],
             "code": format(int(LEAF_CODES[d, p - 1]), f"0{END}b") if p else None}
            for p in range(ws, we + 1)]


def build_tree():
    seen = np.zeros(1 << END, np.uint8)
    seen[LEAF_CODES.ravel()] = 1
    depth = {d: {int(c) >> (END - d) for c in np.flatnonzero(seen)} for d in range(START, END + 1)}
    nodes = []
    def walk(code):
        d = len(code)
        kids = [code + "0", code + "1"]
        alive = d < END and all(int(k, 2) in depth[d + 1] for k in kids)
        nodes.append({"code": code, "depth": d, "dead": d < END and not alive})
        if alive:
            for k in kids:
                walk(k)
    for c in sorted(depth[START]):
        walk(format(c, f"0{START}b"))
    return {"nodes": nodes}


TREE = build_tree()


def search(prefixes):
    W, m = LEAF_CODES.shape[1], len(prefixes)
    hit = np.ones((len(LEAF_CODES), W - m + 1), bool)
    for j, p in enumerate(prefixes):
        hit &= ((LEAF_CODES >> (END - len(p))) == int(p, 2))[:, j:W - m + 1 + j]
    docs, cols = np.nonzero(hit)
    nxt, ok = cols + m, cols + m < W
    follow = np.bincount(LEAF_CODES[docs[ok], nxt[ok]], minlength=1 << END).tolist()
    di = np.array([len(p) - START for p in prefixes])
    j = np.arange(m)
    with np.errstate(divide="ignore"):
        scores = np.exp(np.log(SCORES[docs[:, None], cols[:, None] + j, di] / 255.0).mean(1))
    order = np.argsort(-scores)[:MAX_HITS]
    hits = []
    for d, c, s in zip(docs[order].tolist(), cols[order].tolist(), scores[order].tolist()):
        start, end = c + 1, c + m
        ws, we = max(1, start - CTX), min(SEQ - 1, end + CTX)
        hits.append({"doc_id": d, "start": start, "end": end, "score": float(s),
                     "ctx_start": ws, "tokens": toks(d, ws, we)})
    return {"total": int(len(docs)), "hits": hits, "follow": follow}


def doc(i):
    return {"doc_id": i, "tokens": toks(i, 0, SEQ - 1)}


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
            return self._json(doc(i) if i < len(TOKEN_IDS) else {"error": "no such doc"},
                              200 if i < len(TOKEN_IDS) else 404)
        if self.path == "/":
            self.path = "/index.html"
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
    print(f"{len(TOKEN_IDS)} docs · {len(TREE['nodes'])} nodes · http://localhost:8000")
    ThreadingHTTPServer(("", 8000), partial(Handler, directory=str(HERE))).serve_forever()
