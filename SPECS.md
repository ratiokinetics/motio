# motio interface

GPT-2 tokens clustered by residual activations (layer `n_layer * 2 // 3`). Codes are binary strings of length 6–10. Leaves are 10-bit. Pos 0 is never assigned.

Run artifacts: `tokenized.jsonl`, `assigned.jsonl`, `data/kmeans/RUN_*/{mu.npy,depth_XX.npz}`.

## Match

Token leaf `c` matches prefix `p` iff `c.startswith(p)`.

A pattern `p0, p1, …` hits a doc at token index `i` iff tokens `i, i+1, …` match `p0, p1, …` in order. Pos 0 never hits.

## Layout

The map is a square. Each bit of a code cuts the current rectangle in half, in order:

- even index (0, 2, 4, …): split X — `0` left, `1` right
- odd index (1, 3, 5, …): split Y — `0` bottom, `1` top

A child occupies half of its parent. Dead nodes do not split.

## Server

Boot once into RAM: `token_ids (N,512)`, `leaf_codes (N,511)` as uint16, tokenizer, GPT-2, `mu`, leaf centroids, tree from `depth_*.npz`.

Dead node: parent `p` at depth `d` whose `p0` or `p1` is missing at `d+1`.

### `GET /tree`

```
GET /tree

→ {
  "start": 6,
  "end": 10,
  "nodes": [
    { "code": "000000", "depth": 6, "parent": null,     "children": ["0000000", "0000001"], "dead": false },
    { "code": "0000000", "depth": 7, "parent": "000000", "children": ["00000000", "00000001"], "dead": false },
    { "code": "111000", "depth": 6, "parent": null,     "children": [], "dead": true }
  ]
}
```

(`nodes` lists every node at depths 6–10; shown abbreviated.)

### `POST /search`

```
POST /search
{ "prefixes": ["0000001", "000001"] }

→ {
  "hits": [
    {
      "doc_id": 42,
      "start": 17,
      "end": 18,
      "ctx_start": 1,
      "tokens": [
        { "text": "The",  "code": "0100110010", "match": false },
        { "text": " sat", "code": "0000001101", "match": true },
        { "text": " on",  "code": "0000010010", "match": true },
        { "text": " mat", "code": "0110010010", "match": false }
      ]
    }
  ]
}
```

`start`/`end` are inclusive match indices in the 512-token doc. `tokens` covers the match ±25 tokens of context (clamped to 1–511), starting at doc position `ctx_start`; `match` marks the queried span. `code` is always 10-bit; the UI paints `code[:len(prefix)]` vs the rest. Mixed prefix lengths allowed.

### `GET /docs/{id}`

```
GET /docs/42

→ {
  "doc_id": 42,
  "tokens": [
    { "text": "The",  "code": null },
    { "text": " cat", "code": "0100110010" },
    { "text": " sat", "code": "0000001101" }
  ]
}
```

Full 512 tokens (`tokens` abbreviated). Same object as search, no `match`.

## Frontend 

Two panes (LHS 60% / RHS 40%)

**LHS — map** This is the entry point. The user has enter by manually writing it in the search bar. The map renders it graphically. The user can choose different zoom level/granulaties of the map

**RHS** lists hits: doc id, matched codes (prefix highlighted), and the context snippet with the match marked. Each hit expands inline to the full 512-token doc (`GET /docs/{id}`). Hovering a token shows its motio code.
