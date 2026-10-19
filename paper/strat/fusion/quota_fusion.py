"""Doc 35 §47: QUOTA-Fusion - the paper's hybrid layout (backbone -> necks -> Rydberg head, Fig. 1) for the Fusion shape heads.

User (Oct 17): classical + quantum as in the paper's model diagram, backbone VisionHOPE and / or our CNextU-net, head = the
Rydberg register with the paper's necks, tuned by NAS / Optuna, weak spots analysed with Lean; separate weights per competition.

Every frame is predicted from a window of 2W + 1 frames of its own shot (offsets -W..W x stride, clamped at the shot ends; each
frame carries its time offset and an inside-the-shot flag), all inputs-only (the J1QHCS input blocks, Ip-sign canonical).
  backbone  tcnext  1-D CNextU-net over the window (ConvNeXt blocks: depthwise k7 + LN + 4x MLP, two /2 stages, skip-concat decoder)
            hope    VisionHOPE SRNL blocks (self-referential nested learning, five self-modifying memories; forward + backward
                    scans over the window), baseline/VisionHOPE CUDA kernels
            mlp     no temporal backbone (control = the deployed J1QHCS recipe)
  necks     stem: one per-frame MLP embedding shared by all frames; gate (paper's zero-init insertion): h = z + tanh(G z) * P c,
            z = the instantaneous trunk (deployed MLP width x depth on the centre frame), c = the backbone's centre output,
            G zero-init so the model starts as the instantaneous one; "pure": h = P c (the backbone replaces the trunk)
  head      linear read-out + linear skip of the centre frame + alpha * W <Z> of the 30-atom input-dependent Rydberg register
            (paper/qlayer.py MultiRegister, 3 x 10), alpha engaged at 0.1 as the deployed register heads
Targets / loss = hpo_qpikan (v8 labels: 7 consistency scalars + 64 rays, scalar weight 9.14, edge weights, clip 1.0).

    QW_QFAST=1 <J1QHCS env> .venv/bin/python paper/strat/fusion/quota_fusion.py cv <arm> '<json params>' <fold k> [K]
        -> runs/strat/fusion/quota/<arm>_f<k>.npz (OOF rows + predictions) + one QF_CV line
"""
from __future__ import annotations

import json
import os
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("STRAT_CUDA", "0")
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import fx  # noqa: E402,I001

VH = fx.ROOT / "baseline/vh_pkg"                                             # visionhope -> ../VisionHOPE (relative symlink, MIT source)
OUT = fx.OUT / "quota"
BASE = {"backbone": "mlp", "neck": "gate", "win": 8, "stride": 1, "bb_width": 256, "bb_depth": 2, "width": 2048, "depth": 4,
        "drop": 0.1, "q_layers": 4, "q_in": 64, "link": "independent", "lr": 0.0012505271791067153, "alpha_init": 0.1}


class CNBlock(nn.Module):
    """ConvNeXt block in 1-D (channels first): depthwise conv, LayerNorm, 4x MLP, layer scale."""

    def __init__(self, c, k=7):
        super().__init__()
        self.dw = nn.Conv1d(c, c, k, padding=k // 2, groups=c); self.ln = nn.LayerNorm(c)
        self.mlp = nn.Sequential(nn.Linear(c, 4 * c), nn.GELU(), nn.Linear(4 * c, c)); self.g = nn.Parameter(torch.full((c,), 1e-2))

    def forward(self, x):
        return x + (self.g * self.mlp(self.ln(self.dw(x).transpose(1, 2)))).transpose(1, 2)


def _even(x):
    return F.pad(x, (0, x.shape[-1] % 2), mode="replicate")


class TCNextUNet(nn.Module):
    """CNextU-net over the frame window: stem (per frame) -> [C] -> /2 [2C] -> /2 [4C] -> up + skip [2C] -> up + skip [C]."""

    def __init__(self, fin, c, depth):
        super().__init__()
        st = lambda ch: nn.Sequential(*[CNBlock(ch) for _ in range(depth)])  # noqa: E731
        self.stem = nn.Sequential(nn.Linear(fin, c), nn.GELU(), nn.Linear(c, c))
        self.e0, self.d0 = st(c), nn.Conv1d(c, 2 * c, 2, 2)
        self.e1, self.d1 = st(2 * c), nn.Conv1d(2 * c, 4 * c, 2, 2)
        self.mid = st(4 * c)
        self.u1, self.f1, self.g1 = nn.Conv1d(4 * c, 2 * c, 1), nn.Conv1d(4 * c, 2 * c, 1), st(2 * c)
        self.u0, self.f0, self.g0 = nn.Conv1d(2 * c, c, 1), nn.Conv1d(2 * c, c, 1), st(c)
        self.out_dim = c

    def forward(self, xw, centre):
        e0 = self.e0(self.stem(xw).transpose(1, 2)); e1 = self.e1(self.d0(_even(e0))); m = self.mid(self.d1(_even(e1)))
        g1 = self.g1(self.f1(torch.cat([F.interpolate(self.u1(m), size=e1.shape[-1]), e1], 1)))
        g0 = self.g0(self.f0(torch.cat([F.interpolate(self.u0(g1), size=e0.shape[-1]), e0], 1)))
        return g0[:, :, centre]


class HopeBlock(nn.Module):
    """VisionHOPE's SRNL memory as a sequence mixer over the window (forward + backward scans), then a 4x MLP; pre-norm residual."""

    def __init__(self, c, head_dim=16):
        super().__init__()
        if str(VH) not in sys.path:
            sys.path.insert(0, str(VH))
        from visionhope.models.srnl import SRNL
        self.ln1, self.ln2 = nn.LayerNorm(c), nn.LayerNorm(c)
        self.fw, self.bw = SRNL(c, head_dim=head_dim), SRNL(c, head_dim=head_dim)
        self.mlp = nn.Sequential(nn.Linear(c, 4 * c), nn.GELU(), nn.Linear(4 * c, c))

    def forward(self, x):
        with torch.autocast("cuda", enabled=False):                          # SRNL keeps an FP32 recurrent state
            y = self.ln1(x.float()); x = x.float() + self.fw(y) + self.bw(y.flip(1)).flip(1)
        return x + self.mlp(self.ln2(x))


class HopeSeq(nn.Module):
    def __init__(self, fin, c, depth):
        super().__init__()
        self.stem = nn.Sequential(nn.Linear(fin, c), nn.GELU(), nn.Linear(c, c))
        self.blocks = nn.ModuleList([HopeBlock(c) for _ in range(depth)]); self.norm = nn.LayerNorm(c); self.out_dim = c

    def forward(self, xw, centre):
        h = self.stem(xw)
        for b in self.blocks:
            h = b(h)
        return self.norm(h[:, centre])


class VHopeMap(nn.Module):
    """The full VisionHOPE block (baseline/VisionHOPE models/block.py: conditional positional conv, gated 7x7 depthwise local mixing,
    four-direction SRNL scans with five self-modifying memories, output LePE, GRN feed-forward with depthwise conv, layer scale) on a
    2-D map per window: rows = the 2W + 1 frames, columns = G learned tokens per frame (a linear tokenizer of the frame's inputs).
    Scans along rows carry the shot's history, along columns the cross-signal mixing. Read-out: the centre row, flattened."""

    def __init__(self, fin, c, depth, tokens=8, mixer=None):
        super().__init__()
        if str(VH) not in sys.path:
            sys.path.insert(0, str(VH))
        from visionhope.models.block import VisionHOPEBlock
        self.G, self.c = tokens, c
        self.tok = nn.Linear(fin, c * tokens); self.ln = nn.LayerNorm(c)
        self.blocks = nn.ModuleList([VisionHOPEBlock(c, head_dim=16, mixer_dim=mixer) for _ in range(depth)])
        self.norm = nn.LayerNorm(c * tokens); self.out_dim = c * tokens

    def dense(self, xw):
        """(B, L, fin) -> (B, L, c * G): every row of the map read out (whole-shot sequences: one pass per shot)."""
        B, L, _ = xw.shape
        h = self.ln(F.gelu(self.tok(xw)).view(B, L, self.G, self.c)).permute(0, 3, 1, 2).contiguous()     # (B, c, L, G)
        for b in self.blocks:
            h = b(h)
        return self.norm(h.permute(0, 2, 1, 3).flatten(2))

    def forward(self, xw, centre):
        return self.dense(xw)[:, centre]


class QuotaFusion(nn.Module):
    def __init__(self, fin, out, p):
        super().__init__()
        w, self.bb_kind, self.neck = p["width"], p["backbone"], p["neck"]
        if not (self.bb_kind != "mlp" and self.neck == "pure"):              # instantaneous trunk (the deployed classical part)
            self.inp = nn.Sequential(nn.Linear(fin, w), nn.GELU())
            self.blocks = nn.ModuleList([nn.Sequential(nn.LayerNorm(w), nn.Linear(w, w), nn.GELU(), nn.Linear(w, w), nn.Dropout(p["drop"]))
                                         for _ in range(p["depth"])])
            self.norm = nn.LayerNorm(w)
        if self.bb_kind != "mlp":
            if self.bb_kind == "vhope":
                self.bb = VHopeMap(fin + 2, p["bb_width"], p["bb_depth"], p.get("tokens", 8), p.get("mixer"))
            else:
                self.bb = {"tcnext": TCNextUNet, "hope": HopeSeq}[self.bb_kind](fin + 2, p["bb_width"], p["bb_depth"])
            self.proj = nn.Linear(self.bb.out_dim, w)
            if self.neck == "gate":
                self.gate = nn.Linear(w, w); nn.init.zeros_(self.gate.weight); nn.init.zeros_(self.gate.bias)
        self.out = nn.Linear(w, out); self.skip = nn.Linear(fin, out)
        from qlayer import MultiRegister, TetheredGeometry3D
        self.q_in = nn.Linear(w, p["q_in"])
        if p.get("reg", "multi") == "tether3d":                              # stage 2: per-input 3-D positions + tethers (doc 35 §23)
            self.reg = TetheredGeometry3D(p["q_in"], 10, 3, layers=p["q_layers"])
        else:
            self.reg = MultiRegister(p["q_in"], 10, 3, layers=p["q_layers"], ansatz="rydberg", link=p["link"], gate=True)
        self.q_out = nn.Linear(30, out); self.alpha = nn.Parameter(torch.full((1,), float(p["alpha_init"])))
        from qlayer import ConditionedGeometryBlock
        for m_ in self.reg.modules():                                        # §48c: physical noise while training (eval stays exact)
            if isinstance(m_, ConditionedGeometryBlock):
                m_.noise = float(p.get("q_noise", 0.0))
        self.fin = fin

    def trunk(self, x):
        h = self.inp(x)
        for b in self.blocks:
            h = h + b(h)
        return self.norm(h)

    def forward(self, xw, centre):
        x = xw[:, centre, :self.fin]
        if self.bb_kind == "mlp":
            h = self.trunk(x)
        else:
            c = self.proj(self.bb(xw, centre))
            h = c if self.neck == "pure" else (lambda z: z + torch.tanh(self.gate(z)) * c)(self.trunk(x))
        return self._head(h, x)

    def _head(self, h, x):
        y = self.out(h) + self.skip(x)
        q = self.reg(torch.tanh(self.q_in(h)).float())
        return y + self.alpha * self.q_out(q.to(y.dtype))

    def forward_seq(self, xs, valid):
        """Whole shots (S, L, fin + 2) with a valid mask (S, L) -> predictions for the valid frames, row-major (Nv, out)."""
        x = xs[valid][:, :self.fin]; c = self.proj(self.bb.dense(xs)[valid])
        h = c if self.neck == "pure" else (lambda z: z + torch.tanh(self.gate(z)) * c)(self.trunk(x))
        return self._head(h, x)


VH_T = os.path.expanduser("~/vh_spark/weights/visionhope_tiny.pth")              # VisionHOPE-T ImageNet-1K (source checkpoint)


def warm_start(net, p, fold=None, log=""):
    """Never from scratch (user, Oct 17): (1) p["warm"] = a saved model (quota/<arm>_f<k>.pt in CV, a deployed head .pt in production)
    -> every parameter with the same name and shape (instantaneous trunk, read-out, skip, register); the new VisionHOPE branch and the
    zero-init gate keep their init, so the model starts exactly at the warm model. (2) p["vh_pre"] = "stage<s>" -> the VisionHOPE
    blocks from VisionHOPE-T stage s (stage 1: width 128, mixer 64, 4 blocks)."""
    if p.get("warm"):
        path = fx.OUT / p["warm"].format(f=fold) if not os.path.isabs(p["warm"]) else p["warm"]
        sd = torch.load(path, map_location="cpu", weights_only=False)["net"]; own = net.state_dict()
        ok = {k: v for k, v in sd.items() if k in own and own[k].shape == v.shape}; net.load_state_dict(ok, strict=False)
        print(f"[qf]{log} warm start {path.name if hasattr(path, 'name') else path}: {len(ok)} / {len(own)} tensors", flush=True)
    if p.get("vh_pre"):
        st_ = int(p["vh_pre"].replace("stage", "")); src = torch.load(VH_T, map_location="cpu", weights_only=False)
        src = src.get("state_dict", src.get("model", src)); n = 0
        for i, b in enumerate(net.bb.blocks):
            pre = f"stages.{st_}.{i}."; sub = {k[len(pre):]: v for k, v in src.items() if k.startswith(pre)}
            miss, unexp = b.load_state_dict(sub, strict=False); assert not unexp, unexp[:5]; n += len(sub)
        print(f"[qf]{log} VisionHOPE-T stage {st_}: {n} tensors into {len(net.bb.blocks)} blocks", flush=True)


class Shots:
    """Standardised inputs of every frame of the given shots on the GPU (+ one zero pad row); a batch = whole shots padded to the
    longest, with time since the shot's first frame and a valid flag as two extra channels."""

    TSCALE = 5000.0                                                           # frame times are in ms; shots last ~2-8 s

    def __init__(self, d, fm, edge, shots, mu, sd):
        rows = np.nonzero(np.isin(d.shot, shots))[0]; dev = torch.device("cuda"); self.N = len(rows)
        X = torch.tensor(np.clip((d.inputs(rows, fm, edge) - mu) / sd, -10, 10), dtype=torch.float32, device=dev)
        self.X = torch.cat([X, torch.zeros(1, X.shape[1], device=dev)]); del X
        t = np.asarray(d.X[rows, d.fn.index("t")], np.float64); sh = d.shot[rows]
        self.t = torch.tensor(np.r_[t, 0.0], dtype=torch.float32, device=dev)
        self.st = np.r_[0, np.nonzero(np.diff(sh))[0] + 1]; self.n = np.diff(np.r_[self.st, len(rows)]); self.sid = sh[self.st]
        pos = np.full(len(d.shot), -1, np.int64); pos[rows] = np.arange(len(rows)); self.pos = pos

    def batch(self, k):
        """k: shot indices (into self.st) -> xs (S, L, fin + 2), valid (S, L), idx (S, L) frame positions (pad -> N)."""
        st, n = torch.tensor(self.st[k], device="cuda"), torch.tensor(self.n[k], device="cuda"); ar = torch.arange(int(n.max()), device="cuda")
        valid = ar[None] < n[:, None]; idx = torch.where(valid, st[:, None] + ar[None], torch.full_like(st[:, None] + ar[None], self.N))
        rel = ((self.t[idx] - self.t[st][:, None]) / self.TSCALE * valid)[..., None]
        return torch.cat([self.X[idx], rel, valid[..., None].float()], -1), valid, idx

    def crops(self, k, L, rng):
        """One random L-frame crop per shot (shorter shots whole): xs (S, L, fin + 2), valid, idx."""
        n = self.n[k]; s0 = (rng.random(len(k)) * np.maximum(n - L + 1, 1)).astype(np.int64)
        st = torch.tensor(self.st[k] + s0, device="cuda"); m = torch.tensor(np.minimum(n - s0, L), device="cuda")
        ar = torch.arange(L, device="cuda"); valid = ar[None] < m[:, None]
        idx = torch.where(valid, st[:, None] + ar[None], torch.full_like(st[:, None] + ar[None], self.N))
        t0 = self.t[torch.tensor(self.st[k], device="cuda")]
        rel = ((self.t[idx] - t0[:, None]) / self.TSCALE * valid)[..., None]
        return torch.cat([self.X[idx], rel, valid[..., None].float()], -1), valid, idx

    def tiles(self, k, L):
        """Inference crops covering every frame of the shots k: starts 0, L/2, L, ... (overlaps averaged by the caller)."""
        out = []
        for j in k:
            n = int(self.n[j]); starts = list(range(0, max(n - L, 0) + 1, max(L // 2, 1)))
            if starts[-1] + L < n:
                starts.append(n - L)
            out += [(j, s_) for s_ in starts]
        return out

    def tile_batch(self, tl, L):
        k = np.array([j for j, _ in tl]); s0 = np.array([s_ for _, s_ in tl]); n = self.n[k]
        st = torch.tensor(self.st[k] + s0, device="cuda"); m = torch.tensor(np.minimum(n - s0, L), device="cuda")
        ar = torch.arange(L, device="cuda"); valid = ar[None] < m[:, None]
        idx = torch.where(valid, st[:, None] + ar[None], torch.full_like(st[:, None] + ar[None], self.N))
        t0 = self.t[torch.tensor(self.st[k], device="cuda")]
        rel = ((self.t[idx] - t0[:, None]) / self.TSCALE * valid)[..., None]
        return torch.cat([self.X[idx], rel, valid[..., None].float()], -1), valid, idx

    def buckets(self, k, per, rng):
        """Shots sorted by length (little padding), cut into batches of `per`, batch order shuffled."""
        k = np.asarray(k)[np.argsort(self.n[k] + rng.random(len(k)))]
        return [k[i:i + per] for i in rng.permutation(np.arange(0, len(k), per))]


def train_seq(p, d, sota, c, fm, edge, tr, Ytr, ev_rows, epochs, seed, sw=9.142857, clip=1.0, log=""):
    """§47 V arms: whole-shot batches (one backbone pass per shot), the same targets / weights / loss as train_eval."""
    import hpo_qpikan as H
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    S = _setup(p, d, sota, c, fm, edge, tr, Ytr); del S.X
    torch.cuda.empty_cache()
    Sh = Shots(d, fm, edge, np.unique(np.r_[d.shot[tr], d.shot[ev_rows]]), S.mu, S.sd); pt = torch.tensor(Sh.pos[tr], device="cuda")
    Y = torch.zeros(Sh.N + 1, S.Y.shape[1], device="cuda"); M = torch.zeros_like(Y); Wt = torch.zeros(Sh.N + 1, device="cuda")
    Y[pt], M[pt], Wt[pt] = S.Y, S.M, S.W; del S.Y, S.M, S.W
    trs = np.nonzero(np.isin(Sh.sid, np.unique(d.shot[tr])))[0]; evs = np.nonzero(np.isin(Sh.sid, np.unique(d.shot[ev_rows])))[0]
    OW = _ow(p, S, sw, Y.shape[1]); net = QuotaFusion(Sh.X.shape[1], Y.shape[1], p).cuda()
    warm_start(net, p, p.get("_fold"), log)
    if int(p.get("jepa", 0)) > 0:                                             # §47b: JEPA pretraining of the VisionHOPE encoder (inputs only)
        from vhope_jepa import pretrain
        pretrain(net.bb, Sh, trs, int(p["jepa"]), crop=int(p.get("crop", 32)), per=int(p.get("shots", 128)), seed=seed, log=log)
    per = int(p.get("shots", 16)); crop = int(p.get("crop", 0))                 # crop > 0: random crop-frame windows of `per` shots / step
    nb = max(1, len(tr) // (per * crop)) if crop else len(Sh.buckets(trs, per, rng)); steps = max(1, epochs * nb)
    rest = [q for n_, q in net.named_parameters() if n_ != "alpha"]
    opt = torch.optim.AdamW([{"params": rest}, {"params": [net.alpha], "weight_decay": 0.0}], lr=p["lr"], weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=p["lr"], total_steps=steps, pct_start=0.05); t0 = time.time(); step = 0
    inz, hub = float(p.get("in_noise", 0.0)), float(p.get("huber", 0.0))     # §48c: input-noise injection (standardised), Huber delta
    print(f"[qf]{log} {json.dumps(p)} fin {Sh.X.shape[1]} train shots {len(trs)} batches/epoch {nb} steps {steps} "
          f"params {sum(q.numel() for q in net.parameters()) / 1e6:.1f}M", flush=True)
    net.train()
    for _ in range(epochs):
        for k in ([rng.choice(trs, per, replace=False) for _ in range(nb)] if crop else Sh.buckets(trs, per, rng)):
            xs, valid, idx = Sh.crops(k, crop, rng) if crop else Sh.batch(k); g = idx[valid]
            if inz > 0:                                                       # sensor noise on the inputs, not on time / valid channels
                xs = torch.cat([xs[..., :-2] + inz * torch.randn_like(xs[..., :-2]) * valid[..., None], xs[..., -2:]], -1)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = net.forward_seq(xs, valid)
            r = out.float() - Y[g]
            e = (r.square() if hub <= 0 else 2 * F.huber_loss(r, torch.zeros_like(r), reduction="none", delta=hub)) * M[g] * OW
            loss = (e.sum(1) * Wt[g]).sum() / M[g].sum().clamp(min=1)
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), clip); opt.step(); sch.step(); step += 1
            if step % max(1, steps // 10) == 0:
                print(f"[qf]{log} step {step}/{steps} loss {loss.item():.4f} alpha {net.alpha.item():+.4f} {time.time() - t0:.0f}s", flush=True)
    net.eval(); P = torch.zeros(Sh.N + 1, Y.shape[1], device="cuda"); cnt = torch.zeros(Sh.N + 1, 1, device="cuda")
    with torch.no_grad():
        if crop:                                                              # overlapping crops (stride crop / 2), predictions averaged
            tl = Sh.tiles(evs, crop)
            for j in range(0, len(tl), per):
                xs, valid, idx = Sh.tile_batch(tl[j:j + per], crop); g = idx[valid]
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    P.index_add_(0, g, net.forward_seq(xs, valid).float())
                cnt.index_add_(0, g, torch.ones(len(g), 1, device="cuda"))
            P = P / cnt.clamp(min=1)
        else:
            for k in Sh.buckets(evs, per, np.random.default_rng(0)):
                xs, valid, idx = Sh.batch(k)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    P[idx[valid]] = net.forward_seq(xs, valid).float()
    return net, S, P[torch.tensor(Sh.pos[ev_rows], device="cuda")].cpu().numpy() * S.ysd + S.ymu


class Windows:
    """Standardised inputs of every frame of the given shots on the GPU + per-frame window indices (offsets clamped to the shot)."""

    def __init__(self, d, fm, edge, shots, mu, sd, win, stride):
        rows = np.nonzero(np.isin(d.shot, shots))[0]; dev = torch.device("cuda")
        self.X = torch.tensor(np.clip((d.inputs(rows, fm, edge) - mu) / sd, -10, 10), dtype=torch.float32, device=dev)
        t = np.asarray(d.X[rows, d.fn.index("t")], np.float64); sh = d.shot[rows]
        st = np.r_[0, np.nonzero(np.diff(sh))[0] + 1]; n = np.diff(np.r_[st, len(rows)])
        self.t = torch.tensor(t, dtype=torch.float32, device=dev)
        self.lo = torch.tensor(np.repeat(st, n), device=dev); self.hi = torch.tensor(np.repeat(st + n - 1, n), device=dev)
        pos = np.full(len(d.shot), -1, np.int64); pos[rows] = np.arange(len(rows)); self.pos = torch.tensor(pos, device=dev)
        self.off = torch.arange(-win, win + 1, device=dev) * stride; self.centre = win
        dt = np.diff(t)[np.diff(sh) == 0]; self.tscale = float(max(np.median(dt) * max(win * stride, 1), 1e-6))

    def batch(self, grow):
        """grow: global row ids (numpy or cuda tensor) -> (B, 2W+1, fin + 2)."""
        p = self.pos[torch.as_tensor(grow, device=self.X.device)]
        raw = p[:, None] + self.off; idx = torch.maximum(torch.minimum(raw, self.hi[p][:, None]), self.lo[p][:, None])
        inside = (raw == idx).float()[..., None]; dt = ((self.t[idx] - self.t[p][:, None]) / self.tscale)[..., None]
        return torch.cat([self.X[idx], dt, inside], -1)


def train_eval(p, d, sota, c, fm, edge, tr, Ytr, ev_rows, epochs, seed, batch=4096, sw=9.142857, clip=1.0, log=""):
    import hpo_qpikan as H
    torch.manual_seed(seed); np.random.seed(seed)
    S = _setup(p, d, sota, c, fm, edge, tr, Ytr)                              # as the deployed heads (§48h: psi coefficients)
    shots = np.unique(np.r_[d.shot[tr], d.shot[ev_rows]]); Wn = Windows(d, fm, edge, shots, S.mu, S.sd, p["win"], p["stride"])
    torch.cuda.empty_cache()
    if S.X is not None:
        del S.X
    trg = torch.tensor(tr, device="cuda"); net = QuotaFusion(Wn.X.shape[1], S.Y.shape[1], p).cuda(); warm_start(net, p, p.get("_fold"), log)
    OW = _ow(p, S, sw, S.Y.shape[1])
    steps = max(1, epochs * (len(tr) // batch)); rest = [q for n_, q in net.named_parameters() if n_ != "alpha"]
    opt = torch.optim.AdamW([{"params": rest}, {"params": [net.alpha], "weight_decay": 0.0}], lr=p["lr"], weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=p["lr"], total_steps=steps, pct_start=0.05); t0 = time.time()
    print(f"[qf]{log} {json.dumps(p)} fin {Wn.X.shape[1]} rows {len(tr)} steps {steps} params {sum(q.numel() for q in net.parameters()) / 1e6:.1f}M"
          f" tscale {Wn.tscale:.4f}s", flush=True)
    net.train()
    for step in range(steps):
        i = torch.randint(0, len(tr), (batch,), device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = net(Wn.batch(trg[i]), Wn.centre)
        e = (out.float() - S.Y[i]).square() * S.M[i] * OW; loss = (e.sum(1) * S.W[i]).sum() / S.M[i].sum().clamp(min=1)
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), clip); opt.step(); sch.step()
        if (step + 1) % max(1, steps // 10) == 0:
            print(f"[qf]{log} step {step + 1}/{steps} loss {loss.item():.4f} alpha {net.alpha.item():+.4f} {time.time() - t0:.0f}s", flush=True)
    net.eval(); P = []
    with torch.no_grad():
        for j in range(0, len(ev_rows), 8192):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                P.append(net(Wn.batch(ev_rows[j:j + 8192]), Wn.centre).float())
    return net, S, torch.cat(P).cpu().numpy() * S.ysd + S.ymu


def predict_rows(ck, dd, d, rows, per=128):
    """gpu_build.heads_gpu hook (§47): a saved QUOTA-Fusion head on `rows` (whole shots, contiguous, time-ordered within a shot).
    dd = the head's input view (canonical / thz / ... wrappers), d = the base data (frame times, shot ids). Overlapping crops
    (stride crop / 2) tile every shot; predictions are averaged; returns de-standardised outputs (len(rows), n_main)."""
    p = ck["qf"]; fin = len(ck["mu"]); net = QuotaFusion(fin, len(ck["ymu"]), p).cuda(); net.load_state_dict(ck["net"]); net.eval()
    X = torch.tensor(np.clip((dd.inputs(rows, ck["fm"], ck["edge"]) - ck["mu"]) / ck["sd"], -10, 10), dtype=torch.float32, device="cuda")
    Sh = Shots.__new__(Shots); Sh.N = len(rows); Sh.X = torch.cat([X, torch.zeros(1, fin, device="cuda")]); del X
    t = np.asarray(d.X[rows, d.fn.index("t")], np.float64); sh = np.asarray(d.shot)[rows]
    Sh.t = torch.tensor(np.r_[t, 0.0], dtype=torch.float32, device="cuda")
    Sh.st = np.r_[0, np.nonzero(np.diff(sh))[0] + 1]; Sh.n = np.diff(np.r_[Sh.st, len(rows)]); Sh.sid = sh[Sh.st]
    crop = int(p["crop"]); P = torch.zeros(Sh.N + 1, len(ck["ymu"]), device="cuda"); cnt = torch.zeros(Sh.N + 1, 1, device="cuda")
    tl = Sh.tiles(np.arange(len(Sh.st)), crop)
    with torch.no_grad():
        for j in range(0, len(tl), per):
            xs, valid, idx = Sh.tile_batch(tl[j:j + per], crop); g = idx[valid]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                P.index_add_(0, g, net.forward_seq(xs, valid).float())
            cnt.index_add_(0, g, torch.ones(len(g), 1, device="cuda"))
    P = (P / cnt.clamp(min=1))[:Sh.N].cpu().numpy(); nm = ck.get("n_main", P.shape[1])
    return P[:, :nm] * ck["ysd"][:nm] + ck["ymu"][:nm]


class QbSetup:
    """hpo_qpikan.Setup for the q95 / betaN targets (no ray-shape calibration, no GPU input copy): mu / sd, standardised targets,
    label mask, edge weights - the same definitions as the shape setup."""

    def __init__(self, d, sota, c, fm, edge, rows, Y):
        T = lambda z: torch.tensor(np.asarray(z), dtype=torch.float32, device="cuda")  # noqa: E731
        self.mu, self.sd = sota.moments64(d.inputs(rows, fm, edge)); self.ymu, self.ysd = np.nanmean(Y, 0), np.nanstd(Y, 0) + 1e-9
        Yn = (Y - self.ymu) / self.ysd; self.M = T(np.isfinite(Yn)); self.Y = T(np.nan_to_num(Yn))
        pos = d.edge_feats()[rows, 1]; W = 1 + c.get("edge_w", 0.0) * ((pos < 0.1) | (pos > 0.9)); self.W = T(W / W.mean()); self.X = None


class PsiSetup(QbSetup):
    """§48h: the 64 psi PCA coefficients (orthonormal basis: coefficient SSE = pixel SSE). One global scale for every coefficient and the
    PSI-F component weights (sigma_k / sigma_0)^-gamma, gamma 0.3152 as sota_train (applied through OW)."""

    def __init__(self, d, sota, c, fm, edge, rows, Y):
        super().__init__(d, sota, c, fm, edge, rows, Y)
        T = lambda z: torch.tensor(np.asarray(z), dtype=torch.float32, device="cuda")  # noqa: E731
        var = np.nanvar(Y, 0); self.ysd = np.full(Y.shape[1], np.sqrt(var.mean())); Yn = (Y - self.ymu) / self.ysd
        self.M = T(np.isfinite(Yn)); self.Y = T(np.nan_to_num(Yn))
        cw = np.sqrt(var / var[0]) ** -0.3152355297039971; self.cw = cw / cw.mean()


def _setup(p, d, sota, c, fm, edge, tr, Ytr):
    import hpo_qpikan as H
    if p.get("target") == "psi":
        return PsiSetup(d, sota, c, fm, edge, tr, Ytr)
    if p.get("target") == "xp":                                               # §48j: the 4 X-point coordinates (NaN where absent)
        return QbSetup(d, sota, c, fm, edge, tr, Ytr)
    return QbSetup(d, sota, c, fm, edge, tr, Ytr) if Ytr.shape[1] == 2 else H.Setup(d, sota, c, fm, edge, tr, Ytr, None, None)


def _ow(p, S, sw, n):
    """Per-output loss weights for n outputs (train_seq has already moved S.Y to the GPU and deleted it)."""
    if p.get("target") == "psi":
        return torch.tensor(S.cw, dtype=torch.float32, device="cuda")
    if p.get("target") == "xp":
        return torch.ones(n, device="cuda")
    OW = torch.ones(n, device="cuda"); OW[:7] = sw
    return OW


def r2_each(Y, P):
    out = []
    for k in range(min(7, Y.shape[1])):
        m = np.isfinite(Y[:, k]); out.append(float(1 - np.mean((Y[m, k] - P[m, k]) ** 2) / np.var(Y[m, k])))
    return out


def main() -> None:
    mode, arm = sys.argv[1], sys.argv[2]; over = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
    p = {**BASE, **over}
    if "QW_GPU_GB" in os.environ:
        torch.cuda.set_per_process_memory_fraction(min(1.0, float(os.environ["QW_GPU_GB"]) / 120.0))
    import head_boundary as HB
    import hpo_qpikan as H
    d, sota, c, fm, edge, lab = H.data(); tr, Ytr = lab["rows_train"], lab["y_train"].astype(np.float64)
    if mode == "cv":                                                          # §47 screen: stratified half of TRAIN shots, K shot folds
        k, K = int(sys.argv[4]), int(sys.argv[5]) if len(sys.argv) > 5 else 3
        half = HB.cv_shots(d, tr, Ytr, 0, 2); m = np.isin(d.shot[tr], half); tr, Ytr = tr[m], Ytr[m]
        held = HB.cv_shots(d, tr, Ytr, k, K); hm = np.isin(d.shot[tr], held)    # folds from the shape labels (identical for q / b)
        if p.get("target") == "qb":                                           # §47d: q95 / betaN of the same shots and frames
            Ytr = d.S[tr][:, [sota.LABELS.index("efit_q95"), sota.LABELS.index("efit_beta_n")]].astype(np.float64)
        elif p.get("target") == "psi":                                        # §48h: the 64 psi PCA coefficients of the same frames
            Ytr = np.asarray(d.C[tr][:, :64], np.float64)
        elif p.get("target") == "xp":                                         # §48j: X-point coordinates R_l, Z_l, R_u, Z_u (xp_labels.npz)
            Ytr = np.load(fx.OUT / "xp_labels.npz")["y"][tr].astype(np.float64)
        ep = int(p.get("epochs", os.environ.get("QF_EPOCHS", "20")))          # per-arm override (warm arms: a few epochs)
        fit = train_seq if p.get("seq") else train_eval                     # seq: whole-shot batches (V arms)
        p = {**p, "_fold": k}
        net, S, P = fit(p, d, sota, c, fm, edge, tr[~hm], Ytr[~hm], tr[hm], ep, seed=int(os.environ.get("QF_SEED", "1")), log=f" {arm} f{k}")
        each = r2_each(Ytr[hm], P); OUT.mkdir(exist_ok=True)
        np.savez(OUT / f"{arm}_f{k}.npz", rows=tr[hm], pred=P.astype(np.float32), y=Ytr[hm].astype(np.float32))
        torch.save({"net": net.state_dict(), "qf": p}, OUT / f"{arm}_f{k}.pt")      # warm-start source for later arms
        yh = Ytr[hm]; px = (f" psiR2 {1 - np.sum((yh - P) ** 2) / np.sum((yh - yh.mean(0)) ** 2):.6f} sse {np.sum((yh - P) ** 2):.6e}"
                            if p.get("target") == "psi" else "")              # §48h: pixel-equivalent (orthonormal basis) R2 + SSE
        if p.get("target") == "xp":                                           # §48j: standardised SSE over the finite X-point labels
            m_ = np.isfinite(yh); px = " nsse " + f"{sum(np.sum((yh[m_[:, k], k] - P[m_[:, k], k]) ** 2) / np.var(yh[m_[:, k], k]) for k in range(yh.shape[1])):.6e}"
        print(f"QF_CV {arm} fold {k}/{K} mean7 {np.mean(each):.5f} | " + " ".join(f"{v:.4f}" for v in each) + f" | alpha {net.alpha.item():+.4f}{px}", flush=True)


def final(arm, p, d, sota, c, fm, edge, lab, epochs, seed):
    """All TRAIN shots (+ every VALID / TEST frame with STRAT_ALL=1) -> head_boundary_v8_<arm>_s<seed>.pt in the heads' checkpoint format."""
    tr, Ytr = lab["rows_train"], lab["y_train"].astype(np.float64); allf = os.environ.get("STRAT_ALL", "") == "1"
    if allf:
        vt = np.load(fx.OUT / "boundary_labels_v8_vt.npz"); tr, Ytr = np.concatenate([tr, vt["rows"]]), np.concatenate([Ytr, vt["y"].astype(np.float64)])
    if p.get("target") == "psi":                                              # §48h: the 64 psi PCA coefficients of the same frames
        Ytr = np.asarray(d.C[np.sort(tr)][:, :64], np.float64)[np.argsort(np.argsort(tr))]
    va = lab["rows_valid"]
    net, S, P = train_seq(p, d, sota, c, fm, edge, tr, Ytr, va, epochs, seed, log=f" {arm} final s{seed}")
    stem = f"head_boundary_v8_{arm}_s{seed}"; np.savez(fx.OUT / f"{stem}.npz", rows_valid=va, pred_valid=P.astype(np.float32))
    torch.save({"qf": p, "net": net.state_dict(), "mu": S.mu, "sd": S.sd, "ymu": S.ymu, "ysd": S.ysd, "fm": fm, "edge": edge,
                "cfg": {**c, "width": p["width"], "depth": p["depth"], "drop": p["drop"]}, "inputs": "magfix",
                "canon": {"1": 1, "2": 2}.get(os.environ.get("STRAT_CANON", ""), 0), "magfix_name": os.environ.get("QW_MAGFIX", "X_magfix"),
                "thz_extra": os.environ.get("STRAT_THZ", "") == "1", "all_data": allf, "hfull": os.environ.get("STRAT_HFULL", "") == "1",
                "liphys": os.environ.get("STRAT_LIPHYS", "") == "1", "qphys": {"1": 1, "2": 2}.get(os.environ.get("STRAT_QPHYS", ""), 0),
                "qfast": os.environ.get("QW_QFAST", "") == "1"}, fx.OUT / f"{stem}.pt")
    print(f"QF_FINAL_DONE {arm} seed={seed} all={allf}", flush=True)


def main_final() -> None:
    """    .venv/bin/python paper/strat/fusion/quota_fusion.py final <arm> '<json>' <seed> [epochs]"""
    arm, p, seed = sys.argv[2], {**BASE, **json.loads(sys.argv[3])}, int(sys.argv[4]); ep = int(sys.argv[5]) if len(sys.argv) > 5 else 30
    if "QW_GPU_GB" in os.environ:
        torch.cuda.set_per_process_memory_fraction(min(1.0, float(os.environ["QW_GPU_GB"]) / 120.0))
    import hpo_qpikan as H
    d, sota, c, fm, edge, lab = H.data(); final(arm, p, d, sota, c, fm, edge, lab, ep, seed)


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "final":
    main_final(); sys.exit(0)

if __name__ == "__main__":
    main()
