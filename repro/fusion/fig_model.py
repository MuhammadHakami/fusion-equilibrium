"""Architecture figure of the F18K entry (repro/fusion/model.png): the VisionHOPE + input-dependent Rydberg-register shape head in
layer detail, the per-frame inputs it reads, and the ensemble / psi assembly around it. Drawing style of the paper's Figure 1
(paper/alloc/fig_arch.py): coloured title band per module, one italic line on its role, fully connected layers as bars, the register
as a gate-level circuit.

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
def band(ax, x0, x1, y, title, color, h=1.5, fs=5.2, tc="white", ha="center"):
    ax.add_patch(FancyBboxPatch((x0, y), x1 - x0, h, boxstyle="round,pad=0.05,rounding_size=0.45", fc=color, ec="none"))
    ax.text((x0 + x1) / 2 if ha == "center" else x0 + 1.0, y + h / 2, title, ha=ha, va="center", fontsize=fs, color=tc, weight="bold")


def desc(ax, x, y, text, fs=4.3, color="0.25", ha="center"):
    ax.text(x, y, text, ha=ha, va="top", fontsize=fs, color=color, style="italic", linespacing=1.2)


def lab(ax, x, y, text, fs=4.2, ha="center", va="top", color="0.1", **kw):
    ax.text(x, y, text, ha=ha, va=va, fontsize=fs, color=color, linespacing=1.15, **kw)


def arrow(ax, a, b, ls="-", color="0.25", rad=0.0, lw=0.7):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=6, lw=lw, color=color, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=0.5, shrinkB=0.5))


def box(ax, x, y, w, h, text, fc, fs=4.4, ec="0.35", lw=0.5, ls="-", tc="0.1"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1,rounding_size=0.4", fc=fc, ec=ec, lw=lw, ls=ls))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, linespacing=1.2, color=tc)


def mlp(ax, x, y, w, h, widths, fc="#8a7fbf", fs=4.0, labels=None):
    """Fully connected layers: one bar per layer, height ~ log width, fan connections, widths below."""
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


def circuit(ax, x, y, w, h, fs=4.0, layers=4, atoms=10):
    """One 10-atom register: |0> wires, [R_z(delta) R_x(Omega g)] per atom, the blockade phase over all atoms, x layers, <Z>."""
    nw = 4; wy = [y + h - 0.6 - i * (h - 1.2) / (nw - 1) for i in range(nw)]
    lb_ = ["1", "2", "$\\vdots$", str(atoms)]
    for yy, lb in zip(wy, lb_):
        if lb != "$\\vdots$":
            ax.plot([x + 1.7, x + w - 0.2], [yy, yy], color="0.35", lw=0.4)
            ax.text(x + 0.05, yy, f"$|0\\rangle_{{{lb}}}$", fontsize=fs, va="center")
        else:
            ax.text(x + 0.9, yy, lb, fontsize=fs, va="center", ha="center")
    gx = x + 3.3
    for yy, lb in zip(wy, lb_):
        if lb != "$\\vdots$":
            ax.add_patch(Rectangle((gx, yy - 0.55), 2.8, 1.1, fc="#f6c89f", ec="0.3", lw=0.4))
            ax.text(gx + 1.4, yy, "$R_z\\!R_x$", fontsize=fs - 0.4, ha="center", va="center")
    ix = gx + 3.2
    ax.add_patch(Rectangle((ix, wy[-1] - 0.6), 2.3, wy[0] - wy[-1] + 1.2, fc="#e9a46b", ec="0.3", lw=0.4))
    ax.text(ix + 1.15, (wy[0] + wy[-1]) / 2, "$e^{-i\\tau V}$", fontsize=fs, ha="center", va="center", rotation=90)
    ax.add_patch(Rectangle((gx - 0.4, wy[-1] - 1.0), ix + 2.7 - gx, wy[0] - wy[-1] + 2.0, fc="none", ec=RUST, lw=0.45, ls=(0, (2, 1.2))))
    ax.text(ix + 2.8, wy[0], f"$\\times {layers}$", fontsize=fs + 0.4, color=RUST, va="center", bbox=dict(fc=ORANGE, ec="none", pad=0.3))
    mx = x + w - 2.4
    for yy, lb in zip(wy, lb_):
        if lb != "$\\vdots$":
            ax.add_patch(Rectangle((mx, yy - 0.55), 1.8, 1.1, fc="white", ec="0.3", lw=0.4))
            ax.add_patch(Arc((mx + 0.9, yy - 0.3), 1.2, 1.0, theta1=0, theta2=180, lw=0.35))
            ax.plot([mx + 0.9, mx + 1.4], [yy - 0.3, yy + 0.35], color="0.2", lw=0.35)


def shotmap(ax, x, y, w, h, rows=9, cols=8, seed=2):
    """The per-shot 2-D map: rows = frames of the shot (time), columns = learned tokens; scan arrows in four directions."""
    r = np.random.default_rng(seed); f = np.cumsum(r.normal(size=(rows, cols)), 0) + np.sin(np.arange(cols))[None]
    ax.imshow(f, extent=(x, x + w, y, y + h), cmap="viridis", aspect="auto", origin="upper", zorder=2, alpha=0.85)
    for i in range(rows + 1):
        ax.plot([x, x + w], [y + i * h / rows] * 2, color="white", lw=0.25, zorder=3)
    for j in range(cols + 1):
        ax.plot([x + j * w / cols] * 2, [y, y + h], color="white", lw=0.25, zorder=3)
    ax.add_patch(Rectangle((x, y), w, h, fc="none", ec="0.3", lw=0.5, zorder=4))
    yr = y + h * (1 - 4.5 / rows)
    ax.add_patch(Rectangle((x, yr - h / rows / 2), w, h / rows, fc="none", ec="#ffd54f", lw=0.8, zorder=5))


def lcfs(ax, x, y, s, n=16):
    """A D-shaped last closed flux surface with rays from the axis (the 64-ray boundary target)."""
    th = np.linspace(0, 2 * np.pi, 200); R = x + s * (0.42 * np.cos(th + 0.35 * np.sin(th))); Z = y + s * 0.62 * np.sin(th)
    ax.plot(R, Z, color=TEAL, lw=0.7)
    for t in np.linspace(0, 2 * np.pi, n, endpoint=False):
        k = np.argmin(np.abs((th - t + np.pi) % (2 * np.pi) - np.pi))
        ax.plot([x, R[k]], [y, Z[k]], color="0.55", lw=0.25)
    ax.add_patch(Circle((x, y), 0.25, fc=RUST, ec="none"))


def field(ax, x, y, w, h, seed=0):
    yy, xx = np.mgrid[-1:1:33j, -1:1:33j]; f = -np.exp(-((xx / 0.55) ** 2 + (yy / 0.8) ** 2)) + 0.1 * xx
    ax.imshow(f, extent=(x, x + w, y, y + h), cmap="RdBu_r", aspect="auto", origin="lower", zorder=2)
    ax.contour(np.linspace(x, x + w, 33), np.linspace(y, y + h, 33), f, levels=6, colors="k", linewidths=0.25, zorder=3)
    ax.add_patch(Rectangle((x, y), w, h, fc="none", ec="0.3", lw=0.5, zorder=4))


def shares(ax, x, y, w, h, items):
    """Horizontal stacked bar of ensemble weights."""
    tot = sum(v for _, v, _ in items); cx = x
    for name, v, col in items:
        ww = w * v / tot; ax.add_patch(Rectangle((cx, y), ww, h, fc=col, ec="white", lw=0.5))
        ax.text(cx + ww / 2, y + h / 2, f"{name}\n{v / tot:.2f}", ha="center", va="center", fontsize=3.9, color="white", weight="bold", linespacing=1.1)
        cx += ww


# ------------------------------------------------------------------------------------------------- figure
def main() -> None:
    s = sizes()
    fig, ax = plt.subplots(figsize=(9.8, 7.6)); ax.set_xlim(0, 100); ax.set_ylim(0, 78); ax.axis("off")

    # ======================= row A: per-frame inputs (challenge inputs only)
    band(ax, 0.5, 99.5, 76.0, f"Inputs per EFIT frame t (challenge inputs only): {s['fin']} features, Ip-sign canonical", C_IN, h=1.7, fs=5.6)
    segs = [("magnetics 294: 21 channels x 14 times\n(18 F-coil currents, E-coil, B-coil, Ip;\nt - 200 ms ... t + 20 ms), corrected", 294, "#4a7fb5"),
            ("Thomson 132: core / edge\nTe, ne, pe (stats + profiles)", 132, "#6b9bc9"), ("time\n1", 1, "#8fb3d9"),
            ("Thomson on fixed\nR / Z grid 105", 105, "#3f8f9a"), ("full-rate current\nhistory 24", 24, "#5aa5ae"),
            ("li proxies\n(Te^1.5) 13", 13, "#7bbac1"), ("physics scalings,\ncoil / |Ip| 29", 29, "#2e7d5b"), ("sign\nIp 1", 1, "#5b4a8a")]
    assert sum(v for _, v, _ in segs) == s["fin"] and 294 + 132 + 1 == s["base"]
    wv = [max(math.sqrt(v), 4.2) for _, v, _ in segs]; x0, W_, y0 = 1.0, 98.0, 71.5; cx = x0
    for (name, v, col), w_ in zip(segs, wv):
        ww = W_ * w_ / sum(wv); ax.add_patch(Rectangle((cx, y0), ww, 2.4, fc=col, ec="white", lw=0.6))
        ax.text(cx + ww / 2, y0 - 0.35, name, ha="center", va="top", fontsize=4.0, linespacing=1.1); cx += ww
    desc(ax, 50, 66.9, f"psi trunks read the base {s['base']}; shape and X-point heads all {s['fin']}; q95 / betaN heads the base + physics scalings "
         f"+ the 19 coil currents / |Ip| ({s['qb_in']}). Reversed-Ip shots: currents x sign(Ip) (canonical inputs).", fs=4.3)

    # ======================= row B: VisionHOPE + register shape head (V60)
    band(ax, 0.5, 99.5, 62.6, f"New in F18K: VisionHOPE + input-dependent Rydberg-register shape head "
         f"(V60: {s['params']} M parameters; 2 seeds x 60 epochs)", C_VH, h=1.7, fs=5.6)
    # B1 shot map
    band(ax, 0.8, 15.4, 59.9, "shot map", C_IN, h=1.25, fs=4.9)
    desc(ax, 8.1, 59.5, "every frame of the shot:\ninputs + time + valid flag")
    shotmap(ax, 2.2, 44.6, 11.8, 10.4)
    lab(ax, 1.8, 49.8, "frames (time)", fs=3.9, ha="right", va="center", rotation=90)
    lab(ax, 8.1, 55.3, f"{s['tokens']} tokens", fs=3.9, va="bottom")
    lab(ax, 8.1, 44.1, f"{s['c']} ch x L frames x {s['tokens']} tokens\nL: 32-frame crops (training),\n32-frame tiles, stride 16,\n"
        "overlaps averaged (inference)", fs=3.9)
    lab(ax, 8.1, 38.6, f"tokenizer: Linear {s['fin'] + s['extra']}\n-> {s['tokens']} x {s['c']}, GELU, LN", fs=3.8, color="0.3")
    # B2 VisionHOPE block
    band(ax, 16.6, 41.4, 59.9, f"VisionHOPE-T block x {s['blocks']}: five inductive biases", C_VH, h=1.25, fs=4.9)
    desc(ax, 29.0, 59.5, "scans along time carry the shot history,\nscans across tokens mix the signals")
    bx = [("conditional positional conv 3x3, depthwise", "#cfe6e8"),
          (f"SRNL scans in 4 directions, {s['heads']} heads x {s['hd']}  [SSM]", "#a9d3d7"),
          ("5 self-modifying memories k, v, m, eta, alpha\n[test-time training, nested learning]", "#a9d3d7"),
          ("gated 7x7 depthwise local mixing  [CNN]", "#cfe6e8"),
          ("LePE 5x5 positional conv  [global context]", "#cfe6e8"),
          (f"GRN feed-forward {s['c']} -> {s['ffn']} (dw 3x3) -> {s['c']}", "#e6f2f3")]
    yb = 55.4
    for t_, col in bx:
        hh = 2.5 if "\n" in t_ else 1.6; box(ax, 17.4, yb - hh, 23.2, hh, t_, col, fs=4.0); yb -= hh + 0.42
    lab(ax, 29.0, yb - 0.15, f"row of frame t: {s['tokens']} x {s['c']} = {s['tokens'] * s['c']}, LN\nLinear {s['tokens'] * s['c']} -> {s['width']}  =  c",
        fs=4.0, color="0.2")
    # B3 trunk + gate
    band(ax, 42.6, 61.4, 59.9, "frame trunk + zero-init gate (neck)", C_GATE, h=1.25, fs=4.9)
    desc(ax, 52.0, 59.5, "G = 0 at init: training starts\nfrom the per-frame model")
    mlp(ax, 44.0, 49.6, 16.0, 7.0, [s["fin"], s["width"], s["width"], s["width"]], fc="#b0b0b0",
        labels=[str(s["fin"]), str(s["width"]), f"x{s['depth']}", str(s["width"])])
    lab(ax, 51.0, 49.3, f"z = MLP(x_t): {s['fin']} -> {s['width']}, {s['depth']} residual blocks\n[LN, Linear, GELU, Linear, dropout 0.1], LN",
        fs=3.9)
    box(ax, 46.2, 40.0, 11.6, 3.4, "$h = z + \\tanh(G z) \\odot c$", "white", fs=4.8, ec=C_GATE, lw=0.8)
    lab(ax, 52.0, 39.5, f"G: Linear {s['width']} -> {s['width']}, zero-init", fs=3.9, color="0.3")
    # B4 register
    band(ax, 62.6, 87.4, 59.9, f"Rydberg register: {s['regs']} x {s['atoms']} atoms, input-dependent", C_QPU, h=1.25, fs=4.9)
    desc(ax, 75.0, 59.5, "per input: every atom's detuning and drive;\ntrained atom positions set the blockade graph")
    mlp(ax, 62.6, 50.2, 5.0, 5.6, [s["width"], s["q_in"]], fc=C_QPU2)
    lab(ax, 65.1, 49.9, "Linear, tanh", fs=3.8)
    circuit(ax, 68.4, 49.2, 18.6, 8.0, layers=s["layers"], atoms=s["atoms"])
    lab(ax, 77.7, 48.4, f"x {s['regs']} registers; each 2^{s['atoms']} amplitudes, exact statevector", fs=3.9)
    lab(ax, 75.0, 46.6, "$\\delta_{lk} = b_{lk} + \\pi \\tanh(W h)_{lk}$,     drive $\\Omega_l\\, \\sigma(W_g h)_k$", fs=4.1)
    lab(ax, 75.0, 44.5, "$V = \\sum_{i<j} C_6 / |r_i - r_j|^6\\, n_i n_j$,   r trained (2-D)", fs=4.1)
    lab(ax, 75.0, 42.4, f"W: {s['q_in']} -> {s['layers']} x {s['atoms']},  W_g: {s['q_in']} -> {s['atoms']} per register;  "
        f"read-out <Z_i>, {s['regs'] * s['atoms']} values", fs=3.8, color="0.3")
    # B5 head + outputs
    band(ax, 88.6, 99.5, 59.9, "head", C_HEAD, h=1.25, fs=4.9)
    desc(ax, 94.3, 59.5, "linear + skip\n+ alpha x register")
    lab(ax, 94.3, 56.6, "$y = W_o h + W_s x_t$\n$+\\ \\alpha\\, W_q \\langle Z \\rangle$", fs=4.6)
    lab(ax, 94.3, 52.7, f"W_o {s['width']} -> {s['out']}\nW_s {s['fin']} -> {s['out']}\nW_q {s['regs'] * s['atoms']} -> {s['out']}\nalpha learned (init 0.1)",
        fs=3.8, color="0.3")
    lcfs(ax, 94.3, 43.9, 5.0)
    lab(ax, 94.3, 40.4, f"{s['out']} outputs / frame:\n7 shape scalars + 64\nboundary rays", fs=3.9)
    # arrows row B
    arrow(ax, (14.1, 49.8), (17.2, 49.8))
    arrow(ax, (37.0, 38.6), (46.0, 41.2), rad=0.15)
    arrow(ax, (52.0, 47.4), (52.0, 43.6))
    arrow(ax, (57.9, 42.6), (63.0, 51.6), rad=-0.25)
    arrow(ax, (86.8, 54.4), (89.0, 55.6))
    ax.add_patch(FancyArrowPatch((57.9, 40.6), (89.0, 53.4), arrowstyle="-|>", mutation_scale=6, lw=0.7, color="0.45",
                                 connectionstyle="angle,angleA=0,angleB=90,rad=0", shrinkA=0.5, shrinkB=0.5))

    # ======================= row C: ensemble and psi assembly
    band(ax, 0.5, 99.5, 34.6, "F18K ensemble and psi assembly", C_ENS, h=1.7, fs=5.6, ha="left")
    band(ax, 0.8, 22.0, 31.6, "psi trunks x 6", C_PSI, h=1.25, fs=4.9)
    desc(ax, 11.4, 31.2, "PCA coefficients of the 65 x 65 flux")
    mlp(ax, 0.8, 22.2, 12.6, 7.4, [s["base"], s["psi_w"], s["psi_w"], s["psi_w"], s["psi_k"]], fc="#9d8fc9",
        labels=[f"{s['base']}", "2048", "x2", "2048", str(s["psi_k"])])
    field(ax, 14.8, 21.4, 6.4, 6.4)
    lab(ax, 11.4, 20.6, f"MLP {s['psi_w']} x {s['psi_d']} residual + skip -> {s['psi_k']} PCA\ncoefficients; psi = mean + sum_k c_k phi_k;\n"
        "3 trunks also read the physics scalings", fs=3.9)

    band(ax, 23.4, 58.0, 31.6, f"shape-head ensemble ({s['out']} outputs)", C_HEAD, h=1.25, fs=4.9)
    desc(ax, 40.7, 31.2, "Ip-sign routed; weights fixed on held-out shots")
    shares(ax, 24.0, 25.6, 33.4, 3.6, [("MLP (8 + 4 twins)", 24, "#6c757d"), ("MLP + register x5", 16, C_QPU), ("VisionHOPE + reg. x2", 13.5, C_VH)])
    lab(ax, 40.7, 25.1, "MLP 4096 x 4 (8 seeds; reversed Ip: 4 canonical twins);\nregister heads: MLP + 30-atom register; li blended\n"
        "with a CatBoost li tree (w 0.2)", fs=3.85)
    box(ax, 24.0, 14.8, 16.2, 4.0, f"q95 / betaN heads (Huber): 3 seeds\n+ 2 canonical twins (reversed Ip)\nMLP 4096 x 4, {s['qb_in']} in; 0.8 + 0.2 NAS", "#eef5f1", fs=3.9)
    box(ax, 41.2, 14.8, 16.2, 4.0, f"X-point heads x 3\nMLP 4096 x 4 -> {s['xp_out']}\n(lower / upper X-point R, Z)", "#eef5f1", fs=3.9)

    band(ax, 59.4, 99.5, 31.6, "psi edits toward the heads (no training)", GREY, h=1.25, fs=4.9)
    desc(ax, 79.5, 31.2, "the predicted flux is moved onto the predicted shape")
    steps = ["axis\nshift", "boundary\nwarp", "X-point\nplacement", "kappa\nwarp", "X-point\nwarp", "Gauss-\nNewton fit", "li warp\nx 2",
             "kappa + li\npass 2"]
    sx = 59.8
    for i, t_ in enumerate(steps):
        box(ax, sx, 24.4, 4.4, 4.6, t_, "#f2f2f2", fs=3.7); sx += 5.0
        if i < len(steps) - 1:
            arrow(ax, (sx - 0.55, 26.7), (sx - 0.05, 26.7), lw=0.5)
    box(ax, 63.0, 11.0, 10.0, 4.0, "q95, betaN\n(clamped to the\nlabelled range)", "#eef5f1", fs=4.0)
    field(ax, 76.0, 11.4, 6.6, 6.6)
    lab(ax, 79.3, 10.9, "psi(R, Z) 65 x 65", fs=4.3)
    box(ax, 85.6, 12.6, 13.6, 3.6, "scored: S = 0.55 R2 psi + 0.15 R2 q/b\n+ 0.10 (1 - D_LCFS) + 0.20 shape R2", "white", fs=3.7, ls=(0, (2, 1)))
    ang = dict(arrowstyle="-|>", mutation_scale=6, lw=0.7, shrinkA=0.5, shrinkB=0.5)
    ax.add_patch(FancyArrowPatch((21.2, 21.6), (62.0, 24.3), color="0.4", connectionstyle="angle,angleA=0,angleB=90,rad=0", **ang))
    ax.add_patch(FancyArrowPatch((57.6, 18.4), (71.9, 24.3), color="0.4", connectionstyle="angle,angleA=0,angleB=90,rad=0", **ang))
    ax.add_patch(FancyArrowPatch((32.1, 14.7), (62.9, 13.0), color="0.4", connectionstyle="angle,angleA=-90,angleB=180,rad=0", **ang))
    arrow(ax, (57.6, 27.4), (59.7, 26.9))
    arrow(ax, (94.4, 24.3), (82.8, 17.6), rad=-0.2)
    ax.add_patch(FancyArrowPatch((94.3, 38.0), (54.0, 29.3), color=C_VH, ls=(0, (3, 1.5)),
                                 connectionstyle="angle,angleA=180,angleB=90,rad=0", **ang))

    # ======================= footer: how it was chosen
    band(ax, 0.5, 99.5, 7.6, "How the configuration was chosen (shot-level stratified CV on the training shots; test folds untouched)", PLUM, h=1.5, fs=5.2)
    foot = ["Optuna TPE: head learning rate, width, depth, register read-in and layers.  NAS: the q95 / betaN entry trunks.  "
            "Running now: an 8-fold Optuna search of the VisionHOPE depth / width / tokens / crop with the tethered x, y, z register.",
            "Lean 4 + Mathlib (formal/): a head joins the blend only if <e_M, e_X> < |e_M|^2 (CompetitionBlend); the zero-init gate is "
            "never worse than the trunk; a label-noise floor caps the reachable score (ScoreCeiling).",
            "Adoption: paired gain on 150 held-out shots with the CI above zero.  VisionHOPE + register head: "
            "S +0.000084 [+0.000036, +0.000134]."]
    for i, t_ in enumerate(foot):
        lab(ax, 50, 5.6 - i * 1.9, t_, fs=4.1, va="top")
    fig.savefig(OUT, dpi=300, bbox_inches="tight", facecolor="white"); fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
