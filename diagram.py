"""Paper figure: scaled-down motio map (LHS) + example matches (RHS)."""
import argparse, json, re, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]

FILL, STROKE, GRID, HL = "#f5f5f5", "#111", "#ccc", "#ffe600"
START = 6


def show(code):
    n = min(START, len(code))
    head, rest = str(int(code[:n], 2)), code[n:]
    return f"{head}.{rest}" if rest else head


def rect_of(code):
    x = y = 0.0
    w = h = 1.0
    for i, b in enumerate(code):
        if i % 2 == 0:
            w /= 2
            if b == "1":
                x += w
        else:
            h /= 2
            if b == "1":
                y += h
    return x, y, w, h


def px_of(code, gs):
    x, y, w, h = rect_of(code)
    return x * gs, (1 - y - h) * gs, w * gs, h * gs


def lines(label):
    if isinstance(label, list):
        return [str(x) for x in label]
    s = str(label)
    if "\n" in s:
        return s.split("\n")
    parts = s.split()
    return [" ".join(parts[:-1]), parts[-1]] if len(parts) >= 2 else [s]


def spans(s):
    out = []
    for p in re.split(r"(<<.*?>>)", s):
        if p.startswith("<<") and p.endswith(">>"):
            out.append((p[2:-2], True))
        elif p:
            out.append((p, False))
    return out or [("", False)]


def example(ax, cx, cy, s, fs=10):
    r = ax.figure.canvas.get_renderer()
    inv = ax.transData.inverted()
    segs = spans(s)

    def width(txt, hl):
        t = ax.text(0, 0, txt, fontsize=fs, fontweight="bold" if hl else "normal", alpha=0)
        w = t.get_window_extent(renderer=r).transformed(inv).width
        t.remove()
        return w

    ws = [width(t, hl) for t, hl in segs]
    x = cx - sum(ws) / 2
    for (t, hl), w in zip(segs, ws):
        kw = dict(ha="left", va="center", fontsize=fs, color="black", zorder=6)
        if hl:
            kw.update(fontweight="bold",
                      bbox=dict(boxstyle="square,pad=0.12", facecolor=HL, edgecolor="none"))
        ax.text(x, cy, t, **kw)
        x += w + (3 if hl else 0)


def draw_arrow(ax, a, b, color, lw=2.4, head=12):
    dx, dy = b[0] - a[0], b[1] - a[1]
    L = (dx * dx + dy * dy) ** 0.5
    if L < 1:
        return
    ux, uy = dx / L, dy / L
    s = min(head, L * 0.6)
    x1, y1 = b[0] - s * ux, b[1] - s * uy
    ax.plot([a[0], x1], [a[1], y1], color=color, lw=lw, solid_capstyle="butt", zorder=3)
    hx, hy = -uy, ux
    ax.add_patch(Polygon(
        [(b[0], b[1]), (x1 + 0.5 * s * hx, y1 + 0.5 * s * hy),
         (x1 - 0.5 * s * hx, y1 - 0.5 * s * hy)],
        closed=True, facecolor=color, edgecolor=color, lw=0, zorder=3,
    ))


def load(path):
    d = json.loads(path.read_text())
    pixels, seqs = d.get("pixels") or [], d.get("sequences") or []
    if not pixels:
        sys.exit("json needs a non-empty 'pixels' list")
    codes = []
    for p in pixels:
        c, lab = p.get("code", ""), str(p.get("label") or "").strip()
        if not re.fullmatch(r"[01]+", c):
            sys.exit(f"bad pixel code {c!r}")
        if not lab:
            sys.exit(f"pixel {c} needs a label")
        codes.append(c)
    if len(set(codes)) != len(codes):
        sys.exit("pixel codes must be unique")
    known = set(codes)
    for i, s in enumerate(seqs):
        cs, ex = s.get("codes") or [], s.get("examples") or []
        if not cs or not all(re.fullmatch(r"[01]+", c) for c in cs):
            sys.exit(f"sequence {i}: codes must be non-empty binary strings")
        missing = [c for c in cs if c not in known]
        if missing:
            sys.exit(f"sequence {i}: unknown code {missing[0]} (add it to pixels)")
        if not 1 <= len(ex) <= 6:
            sys.exit(f"sequence {i}: expected 1–6 examples, got {len(ex)}")
    return pixels, seqs, bool(d.get("arrows", True))


def main():
    ap = argparse.ArgumentParser(description="Render a motio paper diagram from JSON.")
    ap.add_argument("json", type=Path)
    args = ap.parse_args()
    pixels, seqs, arrows = load(args.json)
    out = args.json.with_suffix(".png")
    labels = {p["code"]: p["label"] for p in pixels}
    depth = START

    pad, gs, gap, rhs_w = 24, 720, 28, 420
    W, H = pad + gs + (gap + rhs_w if seqs else 0) + pad, pad + gs + pad
    fig, ax = plt.subplots(figsize=(W / 100, H / 100), dpi=100)
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    ox, oy = pad, pad
    ax.add_patch(Rectangle((ox, oy), gs, gs, facecolor="#fff", edgecolor="#000", lw=2, zorder=1))

    for i in range(1 << depth):
        x, y, w, h = px_of(format(i, f"0{depth}b"), gs)
        ax.add_patch(Rectangle((ox + x, oy + y), w, h, facecolor="none",
                               edgecolor=GRID, lw=0.7, zorder=2))

    def box(code):
        x, y, w, h = px_of(code, gs)
        return ox + x, oy + y, w, h

    def ctr(code):
        x, y, w, h = box(code)
        return x + w / 2, y + h / 2

    colors = [s.get("color") or STROKE for s in seqs]
    if arrows:
        for i, s in enumerate(seqs):
            if not s.get("color"):
                continue
            for a, b in zip(s["codes"], s["codes"][1:]):
                if a != b:
                    draw_arrow(ax, ctr(a), ctr(b), s["color"], lw=3.2, head=14)

    def callout(cx, cy, code, lab, dx=None, dy=None):
        if dx is None:
            vx, vy = cx - mx, cy - my
            n = (vx * vx + vy * vy) ** 0.5
            if n < 4:
                vx, vy, n = 1.0, -1.0, 2 ** 0.5
            dx, dy = vx / n * 82, vy / n * 82
        lx = min(max(cx + dx, ox + 52), ox + gs - 52)
        ly = min(max(cy + dy, oy + 30), oy + gs - 30)
        ax.plot([cx, lx], [cy, ly], color=STROKE, lw=0.8, zorder=5)
        ax.text(lx, ly - 11, show(code), ha="center", va="center",
                fontsize=10, color="#444", fontfamily="monospace", zorder=7,
                bbox=dict(boxstyle="square,pad=0.2", facecolor="white", edgecolor="none"))
        ax.text(lx, ly + 11, lab, ha="center", va="center",
                fontsize=13, color="black", zorder=7,
                bbox=dict(boxstyle="square,pad=0.2", facecolor="white", edgecolor="none"))

    pts = [ctr(c) for c in labels]
    mx = sum(p[0] for p in pts) / len(pts)
    my = sum(p[1] for p in pts) / len(pts)
    pix = {p["code"]: p for p in pixels}

    for code, lab in labels.items():
        x, y, w, h = box(code)
        s = min(w, h)
        outside = s < 50
        ax.add_patch(Rectangle((x, y), w, h, facecolor=FILL, edgecolor=STROKE,
                               lw=1.0 if outside else 1.6, zorder=4))
        ls = lines(lab)
        if outside:
            off = pix[code].get("offset")
            callout(x + w / 2, y + h / 2, code, " ".join(ls),
                    *(off if off else (None, None)))
        else:
            ax.text(x + w / 2, y + h * 0.22, show(code), ha="center", va="center",
                    fontsize=max(7, min(11, s * 0.18)), color="#444",
                    fontfamily="monospace", zorder=6)
            fs = min(16 if len(ls) == 1 else 13, max(9, s * 0.26))
            for k, t in enumerate(ls):
                ax.text(x + w / 2, y + h * (0.52 + k * 0.24), t,
                        ha="center", va="center", fontsize=fs, color="black", zorder=6)

    if seqs:
        fig.canvas.draw()
        m, pgap = len(seqs), 12
        ph = (gs - pgap * (m - 1)) / m
        rx = ox + gs + gap
        for i, s in enumerate(seqs):
            y = oy + i * (ph + pgap)
            fill, edge, lw = (to_rgba(colors[i], 0.12), colors[i], 1.6) if s.get("color") else (FILL, STROKE, 1.4)
            ax.add_patch(FancyBboxPatch(
                (rx, y), rhs_w, ph, boxstyle="round,pad=0,rounding_size=7",
                facecolor=fill, edgecolor=edge, lw=lw, zorder=4,
            ))
            ax.text(rx + rhs_w / 2, y + ph * 0.18, "  ".join(show(c) for c in s["codes"]),
                    ha="center", va="center", fontsize=7.5, color="#444",
                    fontfamily="monospace", zorder=6)
            exs = s["examples"]
            two = len(exs) > 3
            for k, ex in enumerate(exs):
                col, row = (k // 3, k % 3) if two else (0, k)
                cx = rx + rhs_w * (0.25 + col * 0.5) if two else rx + rhs_w / 2
                example(ax, cx, y + ph * (0.42 + row * 0.2), ex)

    fig.savefig(out, dpi=200, facecolor="white")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
