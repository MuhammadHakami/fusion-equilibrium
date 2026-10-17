"""Architecture figure of the F18K entry (repro/fusion/model.png), drawn as the paper's Figure 1 (paper/alloc/fig_arch.py): feature
maps as slabs with their tensor sizes, fully connected layers as bars, fields as images, the register as a gate-level circuit, one coloured
band per module (stem, backbone, read-out, neck, head) and one italic line on its role.
  row A  backbone: the shot map (frames x tokens) through two VisionHOPE-T blocks (inset: the block's five inductive biases)
  row B  neck + head: the frame's MLP trunk, the zero-init gate, the input-dependent 3 x 10-atom Rydberg register, linear head + skip
  row C  the F18K ensemble and the psi assembly (psi trunks on a PCA basis, shape-head ensemble, psi edits toward the heads)

Every size printed on the figure is read from the trained PyTorch checkpoints when they are present (runs/strat/fusion/*.pt,
runs/sophelio_sota/*.pt), and the script stops if a drawn number disagrees with a tensor shape:

    .venv/bin/python repro/fusion/fig_model.py            -> repro/fusion/model.png (+ model.pdf)
"""
from __future__ import annotations

import math
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Arc, Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "repro/fusion/model.png"
NAVY, RUST, TEAL, PLUM, GREY = "#1f4e79", "#a0522d", "#2e7d5b", "#5b4a8a", "#555555"
C_IN, C_VH, C_TR, C_GATE, C_QPU, C_QPU2, C_HEAD, C_ENS, C_PSI = "#4a7fb5", "#3f8f9a", "#8c8c8c", "#7a5aa8", "#b5541f", "#d0803f", "#2e7d5b", "#1f4e79", "#5b4a8a"
BLUE, DBLUE, ORANGE, GREEN, PURPLE = "#dbe8f5", "#9ec1e3", "#fbe3cf", "#d9efe6", "#e6e1f2"
C_GATE2, C_OUT = "#8a7fbf", "#555555"

# drawn sizes (checked against the checkpoints in sizes())
S = dict(fin=599, extra=2, tokens=8, c=128, blocks=2, heads=8, hd=16, ffn=512, width=2048, depth=4, out=71, q_in=64, regs=3, atoms=10,
         layers=4, params=42.6, base=427, psi_k=64, psi_w=2048, psi_d=2, xp_out=4, qb_in=456)


def sizes() -> dict:
    """Read the V60 / psi / head checkpoints and assert every drawn number; returns the sizes used for the labels."""
    import torch
    s, chk = dict(S), []
    p = ROOT / "runs/strat/fusion/head_boundary_v8_V60_s1.pt"
    if p.exists():
        ck = torch.load(p, map_location="cpu", weights_only=False); sd, qf = ck["net"], ck["qf"]
        nb = len({k.split(".")[2] for k in sd if k.startswith("bb.blocks.")})
        got = dict(fin=len(ck["mu"]), out=len(ck["ymu"]), tokens=qf["tokens"], c=qf["bb_width"], blocks=nb, width=sd["proj.weight"].shape[0],
                   depth=len({k.split(".")[1] for k in sd if k.startswith("blocks.")}), q_in=sd["q_in.weight"].shape[0],
                   regs=len({k.split(".")[2] for k in sd if k.startswith("reg.regs.")}), atoms=sd["reg.regs.0.pos"].shape[0],
                   layers=sd["reg.regs.0.base"].shape[0], ffn=sd["bb.blocks.0.ffn.fc1.weight"].shape[0],
                   heads=sd["bb.blocks.0.operator.d_skip"].shape[0], hd=sd["bb.blocks.0.operator.d_skip"].shape[1],
                   params=round(sum(v.numel() for k, v in sd.items() if not k.endswith(("zdiag", "cz_mask", "bitsf"))) / 1e6, 1))
        assert sd["bb.tok.weight"].shape == (got["tokens"] * got["c"], got["fin"] + S["extra"]), sd["bb.tok.weight"].shape
        assert sd["skip.weight"].shape == (got["out"], got["fin"]) and sd["q_out.weight"].shape == (got["out"], got["regs"] * got["atoms"])
        assert sd["reg.regs.0.cond.weight"].shape == (got["layers"] * got["atoms"] * 2, got["q_in"])
        assert sd["gate.weight"].shape == (got["width"], got["width"]) and sd["inp.0.weight"].shape == (got["width"], got["fin"])
        chk.append(f"V60 ({got['params']} M parameters)")
        s.update(got)
    p = ROOT / "runs/sophelio_sota/PSIFA_s1.pt"
    if p.exists():
        ck = torch.load(p, map_location="cpu", weights_only=False); sd = next(ck[k] for k in ("net", "state", "model") if k in ck)
        got = dict(base=sd["inp.0.weight"].shape[1], psi_w=sd["inp.0.weight"].shape[0], psi_k=sd["psi.weight"].shape[0],
                   psi_d=len({k.split(".")[1] for k in sd if k.startswith("blocks.")}))
        chk.append("PSIFA"); s.update(got)
    p = ROOT / "runs/strat/fusion/head_xp_XPACS_s1.pt"
    if p.exists():
        s["xp_out"] = len(torch.load(p, map_location="cpu", weights_only=False)["ymu"]); chk.append("XPACS")
    p = ROOT / "runs/strat/fusion/head_qb_QHAR2Hu_s1.pt"
    if p.exists():
        s["qb_in"] = len(torch.load(p, map_location="cpu", weights_only=False)["mu"]); chk.append("QHAR2Hu")
    bad = {k: (S[k], s[k]) for k in S if s[k] != S[k]}
    if bad:
        raise SystemExit(f"figure labels disagree with the checkpoints (drawn, actual): {bad}")
    print("sizes checked against: " + (", ".join(chk) if chk else "nothing (no checkpoints found; drawn defaults)"))
    return s


# ------------------------------------------------------------------------------------------------- primitives (paper Fig. 1 style)
def band(ax, x0, x1, y, title, color, h=1.35, fs=4.3, tc="white", ha="center"):
    ax.add_patch(FancyBboxPatch((x0, y), x1 - x0, h, boxstyle="round,pad=0.05,rounding_size=0.45", fc=color, ec="none"))
    ax.text((x0 + x1) / 2 if ha == "center" else x0 + 0.8, y + h / 2, title, ha=ha, va="center", fontsize=fs, color=tc, weight="bold")


def desc(ax, x, y, text, fs=4.1, color="0.25", ha="center"):
    ax.text(x, y, text, ha=ha, va="top", fontsize=fs, color=color, style="italic", linespacing=1.2)


def lab(ax, x, y, text, fs=4.0, ha="center", va="top", color="0.1", **kw):
    ax.text(x, y, text, ha=ha, va=va, fontsize=fs, color=color, linespacing=1.15, **kw)


def arrow(ax, a, b, ls="-", color="0.25", rad=0.0, lw=0.5, style=None):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=5.5, lw=lw, color=color, ls=ls,
                                 connectionstyle=style or f"arc3,rad={rad}", shrinkA=0.3, shrinkB=0.3))


def vec(ax, x, y, h, fc="#bbbbbb", w=0.6, label=None, fs=3.9):
    ax.add_patch(Rectangle((x, y), w, h, fc=fc, ec="0.3", lw=0.4))
    if label:
        lab(ax, x + w / 2, y - 0.3, label, fs=fs)


def mlp(ax, x, y, w, h, widths, fc="#8a7fbf", fs=3.9, labels=None):
    """Fully connected layers: one bar per layer, height ~ log width, fan connections, widths below. Returns (centres, yc)."""
    n = len(widths); top = max(math.log2(v + 1) for v in widths); cx = [x + w * (i + 0.5) / n for i in range(n)]
    hs = [h * (0.28 + 0.72 * math.log2(v + 1) / top) for v in widths]; yc = y + h / 2 + 0.5
    for i in range(n - 1):
        for fa in (-0.45, 0.0, 0.45):
            for fb in (-0.45, 0.45):
                ax.plot([cx[i] + 0.3, cx[i + 1] - 0.3], [yc + fa * hs[i], yc + fb * hs[i + 1]], color="0.6", lw=0.25)
    for i, (c, hh, v) in enumerate(zip(cx, hs, widths)):
        ax.add_patch(Rectangle((c - 0.3, yc - hh / 2), 0.6, hh, fc=fc, ec="0.3", lw=0.4))
        ax.text(c, y + 0.15, labels[i] if labels else str(v), ha="center", va="bottom", fontsize=fs)
    return cx, yc


def fmap(ax, x, y, w, h, rows=8, cols=8, fc="#9ec1e3", layers=3, hl=None, seed=0):
    """A feature map as a stack of channel sheets (front sheet gridded: rows = frames, columns = tokens); hl = highlighted row."""
    d = 0.32
    for i in range(layers - 1, 0, -1):
        ax.add_patch(Rectangle((x + i * d, y + i * d), w, h, fc=fc, ec="0.35", lw=0.35, alpha=0.55))
    r = np.random.default_rng(seed); f = np.cumsum(r.normal(size=(rows, cols)), 0)
    ax.imshow(f, extent=(x, x + w, y, y + h), cmap="Blues", aspect="auto", origin="upper", zorder=2, alpha=0.75)
    for i in range(rows + 1):
        ax.plot([x, x + w], [y + i * h / rows] * 2, color="white", lw=0.25, zorder=3)
    for j in range(cols + 1):
        ax.plot([x + j * w / cols] * 2, [y, y + h], color="white", lw=0.25, zorder=3)
    ax.add_patch(Rectangle((x, y), w, h, fc="none", ec="0.3", lw=0.45, zorder=4))
    if hl is not None:
        ax.add_patch(Rectangle((x, y + h - (hl + 1) * h / rows), w, h / rows, fc="none", ec="#e8a33d", lw=0.9, zorder=5))
    return x + w + (layers - 1) * d, y + h / 2


def kernel(ax, x, y, k, s, fc="#a9d3d7"):
    """A k x k convolution kernel drawn as a small grid of side s."""
    for i in range(k):
        for j in range(k):
            ax.add_patch(Rectangle((x + j * s / k, y + i * s / k), s / k, s / k, fc=fc, ec="white", lw=0.2))
    ax.add_patch(Rectangle((x, y), s, s, fc="none", ec="0.3", lw=0.4))


def scans(ax, x, y, s, n=4):
    """Four-direction scans over the 2-D map (SRNL): a grid with a sweep arrow along each direction."""
    for i in range(n):
        for j in range(n):
            ax.add_patch(Rectangle((x + j * s / n, y + i * s / n), s / n, s / n, fc="#cfe6e8", ec="white", lw=0.2))
    ax.add_patch(Rectangle((x, y), s, s, fc="none", ec="0.3", lw=0.4))
    m = s * 0.12; c = "#1f6f78"
    arrow(ax, (x + m, y + s - m * 1.5), (x + s - m, y + s - m * 1.5), color=c, lw=0.45)
    arrow(ax, (x + s - m, y + m * 1.5), (x + m, y + m * 1.5), color=c, lw=0.45)
    arrow(ax, (x + m * 1.5, y + s - m), (x + m * 1.5, y + m), color=c, lw=0.45)
    arrow(ax, (x + s - m * 1.5, y + m), (x + s - m * 1.5, y + s - m), color=c, lw=0.45)


def node(ax, x, y, sym, r=0.85, fc="white", ec="0.3", fs=4.6):
    ax.add_patch(Circle((x, y), r, fc=fc, ec=ec, lw=0.5, zorder=5))
    ax.text(x, y, sym, ha="center", va="center", fontsize=fs, zorder=6)


def circuit(ax, x, y, w, h, fs=3.9, layers=4, atoms=10, nw=4):
    """One register: |0> wires, [R_z(delta) R_x(Omega g)] per atom, the blockade phase over all atoms, x layers, measurement."""
    wy = [y + h - 0.6 - i * (h - 1.2) / (nw - 1) for i in range(nw)]
    lb_ = ["1", "2", "$\\vdots$", str(atoms)] if nw == 4 else ["1", "$\\vdots$", str(atoms)]
    for yy, lb in zip(wy, lb_):
        if lb != "$\\vdots$":
            ax.plot([x + 1.7, x + w - 0.2], [yy, yy], color="0.35", lw=0.4)
            ax.text(x + 0.05, yy, f"$|0\\rangle_{{{lb}}}$", fontsize=fs, va="center")
        else:
            ax.text(x + 0.9, yy, lb, fontsize=fs, va="center", ha="center")
    gx = x + 3.2
    for yy, lb in zip(wy, lb_):
        if lb != "$\\vdots$":
            ax.add_patch(Rectangle((gx, yy - 0.55), 2.7, 1.1, fc="#f6c89f", ec="0.3", lw=0.4))
            ax.text(gx + 1.35, yy, "$R_z\\!R_x$", fontsize=fs - 0.4, ha="center", va="center")
    ix = gx + 3.1
    ax.add_patch(Rectangle((ix, wy[-1] - 0.6), 2.2, wy[0] - wy[-1] + 1.2, fc="#e9a46b", ec="0.3", lw=0.4))
    ax.text(ix + 1.1, (wy[0] + wy[-1]) / 2, "$e^{-i\\tau V}$", fontsize=fs, ha="center", va="center", rotation=90)
    ax.add_patch(Rectangle((gx - 0.4, wy[-1] - 1.0), ix + 2.6 - gx, wy[0] - wy[-1] + 2.0, fc="none", ec=RUST, lw=0.45, ls=(0, (2, 1.2))))
    ax.text(ix + 2.7, wy[0], f"$\\times {layers}$", fontsize=fs + 0.4, color=RUST, va="center", bbox=dict(fc=ORANGE, ec="none", pad=0.3))
    mx = x + w - 2.3
    for yy, lb in zip(wy, lb_):
        if lb != "$\\vdots$":
            ax.add_patch(Rectangle((mx, yy - 0.55), 1.8, 1.1, fc="white", ec="0.3", lw=0.4))
            ax.add_patch(Arc((mx + 0.9, yy - 0.3), 1.2, 1.0, theta1=0, theta2=180, lw=0.35))
            ax.plot([mx + 0.9, mx + 1.4], [yy - 0.3, yy + 0.35], color="0.2", lw=0.35)


def atoms(ax, x, y, w, h, seed=5):
    """One 10-atom register: trained 2-D positions (orange) with blockade radii; drive / detuning per input (shading)."""
    r = np.random.default_rng(seed)
    for i in range(10):
        gx, gy = x + 0.7 + (i % 5) * (w - 1.4) / 4, y + 0.8 + (i // 5) * (h - 1.6)
        dx, dy = r.uniform(-0.45, 0.45), r.uniform(-0.35, 0.35)
        ax.add_patch(Circle((gx + dx, gy + dy), 0.3, fc=matplotlib.colormaps["Oranges"](r.uniform(0.35, 0.9)), ec="0.3", lw=0.3))
        if i in (2, 6):
            ax.add_patch(Circle((gx + dx, gy + dy), 1.0, fc="none", ec=C_QPU, lw=0.3, ls=(0, (1.5, 1))))


def lcfs(ax, x, y, s, n=16, color=TEAL, ls="-", rays=True, lw=0.7):
    th = np.linspace(0, 2 * np.pi, 200); R = x + s * (0.42 * np.cos(th + 0.35 * np.sin(th))); Z = y + s * 0.62 * np.sin(th)
    ax.plot(R, Z, color=color, lw=lw, ls=ls, zorder=6)
    if rays:
        for t in np.linspace(0, 2 * np.pi, n, endpoint=False):
            k = np.argmin(np.abs((th - t + np.pi) % (2 * np.pi) - np.pi))
            ax.plot([x, R[k]], [y, Z[k]], color="0.55", lw=0.25, zorder=5)
        ax.add_patch(Circle((x, y), 0.25, fc=RUST, ec="none", zorder=6))


def field(ax, x, y, w, h, shift=0.0, label=None):
    yy, xx = np.mgrid[-1:1:33j, -1:1:33j]; f = -np.exp(-(((xx - shift) / 0.55) ** 2 + (yy / 0.8) ** 2)) + 0.1 * xx
    ax.imshow(f, extent=(x, x + w, y, y + h), cmap="RdBu_r", aspect="auto", origin="lower", zorder=2)
    ax.contour(np.linspace(x, x + w, 33), np.linspace(y, y + h, 33), f, levels=6, colors="k", linewidths=0.22, zorder=3)
    ax.add_patch(Rectangle((x, y), w, h, fc="none", ec="0.3", lw=0.45, zorder=4))
    if label:
        lab(ax, x + w / 2, y - 0.3, label)


def modes(ax, x, y, w, h):
    """PCA basis of psi as small images (coarse to fine)."""
    yy, xx = np.mgrid[-1:1:20j, -1:1:20j]; tiles = [(0, 0), (1, 0), (0, 1), (2, 1), (1, 2), (3, 3)]
    tw, th = w / 3 - 0.25, h / 2 - 0.25
    for i, (kx, ky) in enumerate(tiles):
        f = np.cos(np.pi * kx * (xx + 1) / 2) * np.cos(np.pi * ky * (yy + 1) / 2) * np.exp(-(xx ** 2 + yy ** 2))
        cx, cy = x + (i % 3) * (tw + 0.25), y + (1 - i // 3) * (th + 0.25)
        ax.imshow(f, extent=(cx, cx + tw, cy, cy + th), cmap="PuOr", aspect="auto", origin="lower", zorder=2)
        ax.add_patch(Rectangle((cx, cy), tw, th, fc="none", ec="0.35", lw=0.3, zorder=3))


def glyph(ax, x, y, kind, w=4.2, h=2.6):
    """Tiny member icons for the ensemble: mlp (bars), reg (bars + circuit), vh (map + circuit)."""
    if kind in ("mlp", "reg"):
        for i in range(4):
            ax.add_patch(Rectangle((x + 0.25 + i * 0.55, y + 0.3), 0.32, h - 0.6, fc="#9a9a9a", ec="0.3", lw=0.3))
    if kind == "vh":
        fmap(ax, x + 0.1, y + 0.35, 1.9, h - 0.9, rows=5, cols=4, layers=2, seed=3)
    if kind in ("reg", "vh"):
        cx = x + 2.55
        for k in range(3):
            ax.plot([cx, cx + 1.5], [y + 0.55 + k * 0.75] * 2, color="0.35", lw=0.3)
        ax.add_patch(Rectangle((cx + 0.45, y + 0.35), 0.55, 1.95, fc="#e9a46b", ec="0.3", lw=0.3))
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.3", fc="none", ec="0.55", lw=0.35))


def tokamak(ax, x, y, s):
    """DIII-D poloidal cross-section: vessel, plasma flux surfaces, 18 F-coils, magnetic probes, Thomson chord."""
    th = np.linspace(0, 2 * np.pi, 200)
    for k, f in enumerate((0.9, 0.7, 0.5, 0.3)):
        R = x + f * s * 0.30 * np.cos(th + 0.35 * np.sin(th)); Z = y + f * s * 0.48 * np.sin(th)
        ax.fill(R, Z, color=matplotlib.colormaps["RdBu_r"](0.62 + 0.09 * k), zorder=2, lw=0)
    R = x + s * 0.30 * np.cos(th + 0.35 * np.sin(th)); Z = y + s * 0.48 * np.sin(th); ax.plot(R, Z, color=TEAL, lw=0.6, zorder=3)
    ax.plot(x + s * 0.40 * np.cos(th + 0.3 * np.sin(th)), y + s * 0.62 * np.sin(th), color="0.35", lw=1.0, zorder=3)
    for t in np.linspace(0, 2 * np.pi, 14, endpoint=False):
        ax.add_patch(Circle((x + s * 0.40 * np.cos(t + 0.3 * np.sin(t)), y + s * 0.62 * np.sin(t)), 0.16, fc="#e8a33d", ec="none", zorder=4))
    cs = [(-0.52, z) for z in (-0.45, -0.22, 0.0, 0.22, 0.45)] + [(0.55, z) for z in (-0.4, -0.2, 0.0, 0.2, 0.4)] + \
         [(r, 0.75) for r in (-0.3, -0.1, 0.1, 0.3)] + [(r, -0.75) for r in (-0.3, -0.1, 0.1, 0.3)]
    for r, z in cs:
        ax.add_patch(Rectangle((x + r * s - 0.35, y + z * s - 0.3), 0.7, 0.6, fc="#7a96b8", ec="0.3", lw=0.3, zorder=4))
    ax.plot([x + 0.02 * s] * 2, [y - 0.55 * s, y + 0.55 * s], color="#2e7d5b", lw=0.5, ls=(0, (2, 1)), zorder=5)
    for z in np.linspace(-0.4, 0.4, 7):
        ax.add_patch(Circle((x + 0.02 * s, y + z * s), 0.14, fc="#2e7d5b", ec="none", zorder=6))


def traces(ax, x, y, w, h):
    """The shot's input signals over time (Ip, coil currents, Thomson Te), the 32-frame window and frame t."""
    t = np.linspace(0, 1, 200); hh = h / 3
    sig = [np.clip(np.minimum(t / 0.15, 1) * np.minimum((1 - t) / 0.12, 1), 0, 1),
           [0.5 + 0.35 * np.sin(2 * np.pi * (t * f + p)) * np.minimum(t / 0.2, 1) for f, p in ((1.3, 0.1), (0.8, 0.5), (2.0, 0.3))],
           0.25 + 0.6 * np.clip(np.minimum(t / 0.2, 1) * np.minimum((1 - t) / 0.15, 1), 0, 1) * (1 + 0.15 * np.sin(40 * t))]
    cols = ["#1f4e79", ["#4a7fb5", "#6b9bc9", "#9ec1e3"], "#2e7d5b"]
    for i, (sg, c) in enumerate(zip(sig, cols)):
        y0 = y + h - (i + 1) * hh
        ax.add_patch(Rectangle((x, y0 + 0.15), w, hh - 0.3, fc="white", ec="0.6", lw=0.3))
        for ss, cc in (zip(sg, c) if isinstance(c, list) else [(sg, c)]):
            ax.plot(x + t * w, y0 + 0.3 + (hh - 0.6) * np.asarray(ss) * 0.9, color=cc, lw=0.5)
    ax.add_patch(Rectangle((x + 0.48 * w, y), 0.16 * w, h, fc="#e8a33d", alpha=0.18, ec="none"))
    ax.plot([x + 0.56 * w] * 2, [y, y + h], color="#e8a33d", lw=0.7)
    for i, nm in enumerate(("Ip", "coil I", "Te, ne")):
        lab(ax, x - 0.3, y + h - (i + 0.5) * hh, nm, fs=3.6, ha="right", va="center")
    lab(ax, x + 0.56 * w, y - 0.3, "t", fs=3.8, color="#b5541f")


def main() -> None:
    s = sizes(); W = s["width"]; T = s["tokens"] * s["c"]; R = s["regs"] * s["atoms"]
    fig, ax = plt.subplots(figsize=(7.2, 6.1)); ax.set_xlim(0, 100); ax.set_ylim(-1.5, 81.0); ax.axis("off")

    # ======================= input panel (top left)
    band(ax, 0.5, 20.6, 79.3, "Input: one DIII-D shot", C_IN, h=1.7, fs=5.0)
    traces(ax, 4.2, 64.6, 15.0, 11.2)
    tokamak(ax, 10.0, 52.0, 9.0); lab(ax, 10.0, 44.3, "sensors at frame $t$", fs=3.6, color="0.35")

    # ======================= row 1: backbone
    band(ax, 21.4, 99.5, 79.3, f"Backbone: VisionHOPE-T over the whole shot  ({s['params']} M parameters with neck and head)", C_VH, h=1.7, fs=5.0)
    by, yc = 76.3, 70.0
    band(ax, 21.4, 29.3, by, "embedding", GREY); band(ax, 30.0, 52.3, by, f"VisionHOPE-T x {s['blocks']} (SRNL blocks)", C_VH)
    band(ax, 53.0, 62.4, by, "read-out", NAVY)
    arrow(ax, (19.6, yc), (22.4, yc)); lab(ax, 21.0, yc + 0.5, "$X$", fs=4.0, va="bottom")
    mlp(ax, 22.6, yc - 3.4, 6.0, 5.8, [s["fin"] + s["extra"], T], fc="#b0b0b0")
    e0 = fmap(ax, 30.2, yc - 2.5, 3.0, 5.0, seed=1); lab(ax, 31.7, yc - 2.8, f"{s['c']} x L x {s['tokens']}")
    e1 = fmap(ax, 39.2, yc - 2.5, 3.0, 5.0, seed=2); lab(ax, 40.7, yc - 2.8, f"{s['c']} x L x {s['tokens']}")
    e2 = fmap(ax, 48.2, yc - 2.5, 3.0, 5.0, seed=4, hl=3); lab(ax, 49.7, yc - 2.8, f"{s['c']} x L x {s['tokens']}")
    for k, (x0, nxt) in enumerate(((e0[0], 39.1), (e1[0], 48.1))):        # the two VisionHOPE-T blocks (detail: the framed inset)
        bx = x0 + 0.5
        ax.add_patch(FancyBboxPatch((bx, yc - 1.2), 3.6, 2.4, boxstyle="round,pad=0.05,rounding_size=0.4", fc=C_VH, ec="none"))
        ax.text(bx + 1.8, yc, f"SRNL\nblock {k + 1}", ha="center", va="center", fontsize=3.4, color="white", weight="bold", linespacing=1.05)
        arrow(ax, (x0 + 0.05, yc), (bx, yc)); arrow(ax, (bx + 3.65, yc), (nxt, yc))
        ax.plot([bx + 1.8, bx + 1.8], [yc + 1.25, 75.6], color=C_VH, lw=0.5, ls=(0, (2, 1)))
    ax.plot([e0[0] + 2.3, 64.0], [75.6, 75.6], color=C_VH, lw=0.5, ls=(0, (2, 1)))
    ax.add_patch(Rectangle((64.0, 63.4), 35.8, 12.2, fc="none", ec=C_VH, lw=0.6, ls=(0, (2, 1))))
    arrow(ax, (28.3, yc), (30.1, yc))
    ax.add_patch(Polygon([(e2[0] + 0.1, yc + 0.85), (54.0, yc + 3.4), (54.0, yc - 3.4), (e2[0] + 0.1, yc + 0.25)], fc="#fbe3c0", ec="0.5", lw=0.3))
    ax.add_patch(Rectangle((54.1, yc - 3.4), 0.7, 6.8, fc="#f3c98b", ec="0.3", lw=0.4)); lab(ax, 54.45, yc - 3.7, f"{T}")
    mlp(ax, 55.3, yc - 3.4, 5.0, 5.8, [T, W], fc="#7a96b8", labels=["", str(W)])
    vec(ax, 61.0, yc - 3.4, 6.8, fc="#9ec1e3", w=0.7); lab(ax, 61.35, yc - 3.7, "$c$")
    arrow(ax, (60.4, yc), (60.9, yc))
    # inset: one VisionHOPE-T block
    band(ax, 64.0, 99.5, by, "VisionHOPE-T block (detail of blocks 1 and 2)", C_VH)
    ty = yc
    arrow(ax, (64.2, ty), (65.1, ty), lw=0.4)
    kernel(ax, 65.2, ty - 1.2, 3, 2.4); lab(ax, 66.4, ty - 1.5, "pos. conv\n3x3", fs=3.5)
    node(ax, 69.2, ty, "LN", r=0.8, fs=3.5)
    ax.add_patch(FancyBboxPatch((70.8, ty + 0.7), 8.2, 4.3, boxstyle="round,pad=0.1,rounding_size=0.4", fc="#e3f1f2", ec=C_VH, lw=0.8))
    scans(ax, 71.2, ty + 1.15, 3.3)
    ax.text(76.9, ty + 3.6, "SRNL", ha="center", va="center", fontsize=5.0, color=C_VH, weight="bold")
    ax.text(76.9, ty + 2.0, "5 memories", ha="center", va="center", fontsize=3.2, color="0.25")
    kernel(ax, 71.6, ty - 4.4, 7, 2.4, fc="#b9dfe2"); node(ax, 75.3, ty - 3.2, "$\\sigma$", r=0.7, fs=4.0)
    lab(ax, 72.8, ty - 4.7, "gated 7x7", fs=3.5)
    node(ax, 80.6, ty, "+", r=0.75)
    arrow(ax, (79.15, ty + 2.6), (80.2, ty + 0.65), lw=0.8, color=C_VH)
    kernel(ax, 82.0, ty - 1.1, 5, 2.2); lab(ax, 83.1, ty - 1.4, "pos. conv\n5x5", fs=3.5)
    node(ax, 85.6, ty, "+", r=0.7); node(ax, 87.4, ty, "LN", r=0.8, fs=3.5)
    mlp(ax, 88.4, ty - 3.4, 8.4, 5.8, [s["c"], s["ffn"], s["c"]], fc="#7bbac1"); lab(ax, 92.6, ty - 3.7, "MLP (GRN)", fs=3.5)
    node(ax, 99.0, ty, "+", r=0.6)
    for a_, b_ in (((67.65, ty), (68.35, ty)), ((69.95, ty + 0.35), (70.75, ty + 1.3)), ((69.8, ty - 0.55), (71.5, ty - 2.5)),
                   ((74.05, ty - 3.2), (74.55, ty - 3.2)), ((76.0, ty - 2.9), (80.0, ty - 0.45)), ((81.4, ty), (81.9, ty)),
                   ((84.25, ty), (84.85, ty)), ((86.3, ty), (86.55, ty)), ((96.5, ty + 0.1), (98.35, ty))):
        arrow(ax, a_, b_, lw=0.4)
    ax.plot([64.6, 64.6, 85.6, 85.6], [ty, ty - 5.9, ty - 5.9, ty - 0.7], color="0.5", lw=0.35, ls=(0, (2, 1)))
    ax.plot([86.6, 86.6, 99.0, 99.0], [ty - 0.5, ty - 5.9, ty - 5.9, ty - 0.6], color="0.5", lw=0.35, ls=(0, (2, 1)))

    # ======================= row 2: neck + head
    band(ax, 21.4, 46.6, 60.6, "Neck", C_GATE, h=1.7, fs=5.0); band(ax, 47.3, 99.5, 60.6, "Head: input-dependent Rydberg register + linear read-out", C_QPU, h=1.7, fs=5.0)
    by, yc = 57.6, 50.8
    band(ax, 21.4, 34.6, by, "frame encoder (MLP)", GREY); band(ax, 35.3, 46.6, by, "fusion gate", C_GATE2, ha="left")
    band(ax, 47.3, 53.4, by, "encoding", C_QPU2); band(ax, 54.1, 74.3, by, f"Rydberg register {s['regs']} x {s['atoms']}", C_QPU)
    band(ax, 75.0, 87.4, by, "linear read-out", C_HEAD); band(ax, 88.1, 99.5, by, "shape output", C_OUT)
    arrow(ax, (17.0, yc), (21.6, yc)); lab(ax, 19.3, yc + 0.5, f"$x_t$: {s['fin']}", fs=3.7, va="bottom")
    mlp(ax, 21.6, yc - 3.6, 12.2, 6.6, [s["fin"], W, W, W], fc="#b0b0b0", labels=[str(s["fin"]), str(W), f"x{s['depth']}", str(W)])
    vec(ax, 34.1, yc - 3.4, 6.8, fc="#d9d9d9", w=0.6); lab(ax, 34.4, yc - 3.7, "$z$")
    mlp(ax, 35.3, yc - 1.8, 3.4, 4.4, [W, W], fc=C_GATE2, labels=["", ""]); lab(ax, 37.0, yc - 1.7, "$G$", fs=4.0)
    node(ax, 39.8, yc + 0.7, "tanh", r=0.95, fs=3.5); node(ax, 42.6, yc + 0.7, "$\\odot$"); node(ax, 44.6, yc - 1.8, "+")
    vec(ax, 45.6, yc - 3.4, 6.8, fc="#cdbfe6", w=0.6); lab(ax, 45.9, yc - 3.7, "$h$")
    arrow(ax, (34.75, yc + 0.2), (35.6, yc + 0.2)); arrow(ax, (38.4, yc + 0.7), (38.85, yc + 0.7)); arrow(ax, (40.75, yc + 0.7), (41.75, yc + 0.7))
    arrow(ax, (34.75, yc - 2.6), (43.75, yc - 2.0), rad=0.08); arrow(ax, (43.3, yc + 0.1), (44.1, yc - 1.15)); arrow(ax, (45.45, yc - 1.8), (45.55, yc - 1.8))
    ax.plot([61.35, 61.35, 42.6], [yc + 15.8, 63.0, 63.0], color=C_VH, lw=0.55)
    arrow(ax, (42.6, 63.0), (42.6, yc + 1.6), color=C_VH); lab(ax, 43.0, 63.3, f"$c$: {W}", fs=3.8, color=C_VH, ha="left", va="bottom")
    mlp(ax, 47.3, yc - 1.4, 4.6, 5.0, [W, s["q_in"]], fc=C_QPU2); lab(ax, 49.6, yc - 1.7, "$u$, tanh", fs=3.6)
    atoms(ax, 53.9, yc - 3.6, 5.2, 3.4)
    circuit(ax, 59.4, yc - 3.4, 13.6, 7.4, layers=s["layers"], atoms=s["atoms"])
    vec(ax, 73.6, yc - 2.4, 4.6, fc="#f3c98b", w=0.6); lab(ax, 73.9, yc - 2.7, f"$\\langle Z\\rangle$\n{s['regs']} x {s['atoms']}")
    arrow(ax, (46.25, yc), (47.6, yc + 0.6)); arrow(ax, (51.6, yc + 1.0), (53.8, yc + 1.0)); arrow(ax, (59.0, yc + 1.0), (59.3, yc + 1.0))
    arrow(ax, (73.0, yc - 0.1), (73.55, yc - 0.1))
    mlp(ax, 75.2, yc + 0.3, 4.0, 4.0, [R, s["out"]], fc="#5fa98a"); lab(ax, 77.2, yc + 0.0, "$\\alpha W_q$", fs=3.9)
    mlp(ax, 75.2, yc - 7.8, 4.0, 4.0, [W, s["out"]], fc="#5fa98a"); lab(ax, 77.2, yc - 8.1, "$W_o$", fs=3.9)
    mlp(ax, 80.4, yc - 7.8, 4.0, 4.0, [s["fin"], s["out"]], fc="#5fa98a"); lab(ax, 82.4, yc - 8.1, "$W_s$", fs=3.9)
    lab(ax, 80.2, yc - 5.3, "$x_t$", fs=3.9, ha="right", va="center")
    node(ax, 86.2, yc - 1.0, "+")
    arrow(ax, (74.25, yc - 0.1), (75.4, yc + 2.4)); arrow(ax, (79.2, yc + 2.4), (85.5, yc - 0.5))
    arrow(ax, (45.9, yc - 3.5), (75.4, yc - 5.3), style="angle,angleA=-90,angleB=180,rad=0", color="0.45")
    arrow(ax, (79.2, yc - 5.3), (85.7, yc - 1.7), color="0.45"); arrow(ax, (84.4, yc - 5.3), (86.0, yc - 1.8), color="0.45")
    arrow(ax, (87.05, yc - 1.0), (89.0, yc - 1.0))
    lcfs(ax, 92.6, yc - 0.6, 5.2)
    lab(ax, 93.6, yc - 4.5, f"$y$: {s['out']} = 7 shape scalars\n+ 64 boundary rays", fs=3.6)

    # ======================= row 3: ensemble and psi reconstruction
    band(ax, 0.5, 99.5, 38.4, "Ensemble and psi reconstruction", C_ENS, h=1.7, fs=5.0, ha="left")
    by, yc = 35.4, 27.4
    band(ax, 0.8, 17.2, by, "psi networks (MLP) x 6", C_PSI); band(ax, 17.9, 29.0, by, "PCA basis", C_PSI)
    band(ax, 29.7, 58.4, by, "shape networks", C_HEAD, ha="left"); band(ax, 59.1, 87.0, by, "psi correction to the predicted shape", GREY)
    band(ax, 87.7, 99.5, by, "output", C_OUT)
    lab(ax, 1.5, yc + 4.0, f"$x_t$: {s['base']}", fs=3.7, ha="left", va="bottom")
    vec(ax, 1.2, yc - 3.4, 6.4, fc="#cfd8e3", w=0.7)
    mlp(ax, 2.6, yc - 3.6, 11.0, 6.2, [s["base"], s["psi_w"], s["psi_w"], s["psi_w"], s["psi_k"]], fc="#9d8fc9",
        labels=[str(s["base"]), "", f"{s['psi_w']} x{s['psi_d']}", "", str(s["psi_k"])])
    vec(ax, 14.6, yc - 3.0, 5.6, fc="#cdbfe6", w=0.7); lab(ax, 14.95, yc - 3.3, f"{s['psi_k']}")
    modes(ax, 18.3, yc - 3.0, 7.6, 6.2); node(ax, 27.6, yc + 0.1, "$\\Sigma$")
    arrow(ax, (13.9, yc), (14.55, yc)); arrow(ax, (15.35, yc), (18.2, yc)); arrow(ax, (26.0, yc + 0.1), (26.7, yc + 0.1))
    gl = [("vh", "VisionHOPE + register x 2", "0.25"), ("mlp", "MLP x 12", "0.45"), ("reg", "MLP + register x 5", "0.30")]
    for i, (k, t_, w_) in enumerate(gl):
        gy = yc + 4.2 - i * 3.0; glyph(ax, 46.6, gy - 1.2, k, w=4.0, h=2.4)
        lab(ax, 30.4, gy, w_, fs=3.7, ha="left", va="center", color=C_HEAD); lab(ax, 33.2, gy, t_, fs=3.6, ha="left", va="center")
        arrow(ax, (50.7, gy), (53.5, yc + 1.4), color="0.5")
    ax.plot([93.6, 93.6, 48.6], [44.1, 41.0, 41.0], color="0.45", lw=0.55)
    arrow(ax, (48.6, 41.0), (48.6, yc + 5.45), color="0.45"); lab(ax, 49.1, 41.4, "$y$", fs=3.9, ha="left", va="bottom")
    node(ax, 54.4, yc + 1.4, "$\\Sigma$"); lab(ax, 54.4, yc + 0.3, f"{s['out']}", fs=3.6)
    glyph(ax, 30.6, yc - 8.2, "mlp", w=4.0, h=2.4); lab(ax, 32.6, yc - 8.5, f"q95, betaN\n$x_t$: {s['qb_in']}", fs=3.5)
    glyph(ax, 38.4, yc - 8.2, "mlp", w=4.0, h=2.4); lab(ax, 40.4, yc - 8.5, f"X-points\n$x_t$: {s['fin']}", fs=3.5)
    arrow(ax, (42.5, yc - 7.0), (73.0, yc - 0.2), style="angle,angleA=0,angleB=90,rad=0", color="0.45")
    ax.plot([30.5, 29.7, 29.7, 97.0], [yc - 7.0, yc - 7.0, yc - 11.2, yc - 11.2], color="0.45", lw=0.5)
    arrow(ax, (97.0, yc - 11.2), (97.0, yc - 1.4), color="0.45")
    field(ax, 60.2, yc - 3.6, 7.2, 7.2); lcfs(ax, 63.8, yc - 0.0, 4.6, color="#e8a33d", ls=(0, (2, 1)), rays=False, lw=0.6)
    lab(ax, 63.8, yc - 3.9, "$\\psi_0$", fs=4.0)
    arrow(ax, (67.6, yc), (78.4, yc)); field(ax, 78.6, yc - 3.6, 7.2, 7.2, shift=0.06)
    lcfs(ax, 82.2, yc + 0.0, 4.6, color=TEAL, rays=False, lw=0.7)
    arrow(ax, (27.6, yc - 0.8), (60.1, yc - 3.3), style="angle,angleA=-90,angleB=180,rad=0", color="0.45")
    arrow(ax, (55.25, yc + 1.4), (63.6, yc + 3.7), rad=-0.3, color=TEAL)
    arrow(ax, (85.9, yc), (88.6, yc))
    field(ax, 88.8, yc - 2.8, 5.6, 5.6, shift=0.06); lab(ax, 91.6, yc - 3.1, "$\\psi$: 65 x 65", fs=3.8)
    lab(ax, 97.4, yc + 0.0, "q95\nbetaN", fs=4.0, va="center")

    # ======================= row 4: loss and register mechanism
    band(ax, 0.5, 99.5, 10.6, "Loss and the input-dependent Rydberg register", PLUM, h=1.7, fs=5.0)
    lab(ax, 24.0, 8.3, r"$\mathcal{L}=\frac{1}{\sum m_{ik}}\sum_{i} w_i \sum_{k=1}^{71}\lambda_k\, m_{ik}\,(\hat y_{ik}-y_{ik})^2,\quad \lambda_{\rm shape}=9.14,\ \lambda_{\rm ray}=1$", fs=4.5)
    lab(ax, 24.0, 2.6, r"$S = 0.55\,R^2_{\psi} + 0.15\,R^2_{q_{95},\beta_N} + 0.10\,(1-D_{\rm LCFS}) + 0.20\,\overline{R^2}_{\rm shape}$", fs=4.4)
    lab(ax, 74.0, 8.3, r"$u=\tanh(W_{\rm in}h),\quad \delta_{lk}(u)=b_{lk}+\pi\tanh(W u)_{lk},\quad g_k(u)=\sigma(W_g u)_k$", fs=4.4)
    lab(ax, 74.0, 4.8, r"$|\psi(u)\rangle=\prod_{l=1}^{4}\left[e^{-i\tau_l V}\bigotimes_{k=1}^{10}R_z(\delta_{lk})\,R_x(\Omega_l\,g_k)\right]|0\rangle^{\otimes 10},"
        r"\quad V=\sum_{i<j}\frac{C_6}{|\mathbf{r}_i-\mathbf{r}_j|^6}\,n_i n_j$", fs=4.4)
    lab(ax, 74.0, 0.5, r"$y = W_o h + W_s x_t + \alpha\, W_q\,\langle Z\rangle,\qquad h = z+\tanh(Gz)\odot c$", fs=4.4)
    fig.savefig(OUT, dpi=300, bbox_inches="tight", pad_inches=0.03, facecolor="white")
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.03)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
