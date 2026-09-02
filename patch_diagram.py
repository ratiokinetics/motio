"""Paper figure: activation patching — prompt on top, untouched vs patched continuations below."""
import argparse, json, re, sys
from pathlib import Path
from diagram import FILL, STROKE, draw_arrow, example, show
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.patches import FancyBboxPatch


def wrap(s, w=46):
    lines = []
    for para in str(s).split("\n"):
        out = [""]
        for t in re.findall(r"<<.*?>>|\S+", para):
            cand = (out[-1] + " " + t).strip()
            if not out[-1] or len(re.sub(r"<<|>>", "", cand)) <= w:
                out[-1] = cand
            else:
                out.append(t)
        lines.extend(out if out != [""] else [""])
    return lines


def main():
    ap = argparse.ArgumentParser(description="Render an activation-patching figure from JSON.")
    ap.add_argument("json", type=Path)
    args = ap.parse_args()
    d = json.loads(args.json.read_text())
    try:
        pr, un, pa, cl = d["prompt"], d["untouched"], d["patched"], d["cluster"]
        code, label = cl["code"], cl["label"]
    except KeyError as e:
        sys.exit(f"json needs key {e}")
    if not re.fullmatch(r"[01]+", code):
        sys.exit(f"bad cluster code {code!r}")
    color = d.get("color") or "#c0392b"

    pad, gap, pw = 24, 28, 420
    pr_l, un_l, pa_l = wrap(pr, 96), wrap(un), wrap(pa)
    height = lambda n: 34 + 22 * n + 14
    pr_h, qh, az = height(len(pr_l)), height(max(len(un_l), len(pa_l))), 84
    W, H = pad + 2 * pw + gap + pad, pad + pr_h + az + qh + pad

    fig, ax = plt.subplots(figsize=(W / 100, H / 100), dpi=100)
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    fig.canvas.draw()

    def panel(x, y, w, h, header, ls, edge=STROKE, lw=1.4, fill=FILL):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=7",
                                    facecolor=fill, edgecolor=edge, lw=lw, zorder=4))
        ax.text(x + w / 2, y + 18, header, ha="center", va="center",
                fontsize=7.5, color="#444", fontfamily="monospace", zorder=6)
        for k, t in enumerate(ls):
            example(ax, x + w / 2, y + 45 + 22 * k, t)

    qy = pad + pr_h + az
    panel(pad, pad, 2 * pw + gap, pr_h, "prompt", pr_l)
    panel(pad, qy, pw, qh, "untouched", un_l)
    panel(pad + pw + gap, qy, pw, qh, "patched", pa_l,
          edge=color, lw=1.6, fill=to_rgba(color, 0.12))

    lx, rx = pad + pw / 2, pad + pw + gap + pw / 2
    draw_arrow(ax, (lx, pad + pr_h), (lx, qy), STROKE, lw=3.2, head=14)
    draw_arrow(ax, (rx, pad + pr_h), (rx, qy), color, lw=3.2, head=14)
    my = pad + pr_h + az / 2
    ax.text(rx + 14, my - 11, show(code), ha="left", va="center", fontsize=10,
            color="#444", fontfamily="monospace", zorder=7,
            bbox=dict(boxstyle="square,pad=0.2", facecolor="white", edgecolor="none"))
    ax.text(rx + 14, my + 11, label, ha="left", va="center", fontsize=13,
            color="black", zorder=7,
            bbox=dict(boxstyle="square,pad=0.2", facecolor="white", edgecolor="none"))

    out = args.json.with_suffix(".png")
    fig.savefig(out, dpi=200, facecolor="white")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
