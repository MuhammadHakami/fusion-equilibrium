"""Sophelio DIII-D (Challenge 1) SoTA-class pipeline: cached full dataset, hybrid model, train/evaluate.

Used by sota_train.py (single runs) and paper/nas/sophelio_nas.py (NAS + Optuna over classical and quantum parts).

Coupling modes of the quantum part (`quantum`):
  none    — classical trunk only;
  latent  — zero-init gated register bottleneck on the MLP latent (the campaign's standard insertion);
  srr     — spectral-routed register (32-dynamic-register-configuration.md): the register reads the latent and writes
            ONLY into the high-order psi band [k_lo, k) — where the trunk's error lives — through a zero-init map.
Each quantum configuration has a matched classical control (`-mlp` suffix of the Wave-G grammar), same coupling mode.
"""
from __future__ import annotations

import hashlib
import pathlib
import sys
import time

import numpy as np
import torch
import torch.nn as nn

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "paper"))
PROC = ROOT / "data" / "sophelio_proc" / "diii_d_train"
CACHE = ROOT / "data" / "sophelio_proc" / "cache_k256"
DEV = torch.device("cuda")
import gpucap  # noqa: E402

gpucap.cap_from_env()
LABELS = ["efit_beta_n", "efit_li", "efit_q95", "efit_r_axis", "efit_z_axis", "magnetics_dsep"]
HEADS = ["efit_q95", "efit_beta_n", "efit_li", "efit_r_axis", "efit_z_axis"]
KMAX = 256


def split_of(name: str) -> str:
    h = int(hashlib.sha256(name.encode()).hexdigest()[:8], 16) % 100
    return "test" if h < 5 else ("val" if h < 10 else "train")


# ----------------------------------------------------------------------------------------------- data cache
def build_cache() -> None:
    """One pass over all extracted shots: X, PCA-256 coefficients, labels for every frame; full psi kept only for
    VALID/TEST shots (memmap). PCA basis from 400 random TRAIN shots (GPU SVD). Idempotent."""
    CACHE.mkdir(parents=True, exist_ok=True)
    if (CACHE / "meta.pt").exists():
        return
    files = sorted(PROC.glob("*.npz"))
    names = [f.stem for f in files]
    tr = [f for f in files if split_of(f.stem) == "train"]
    rng = np.random.default_rng(0)
    sub = [tr[i] for i in rng.choice(len(tr), size=min(400, len(tr)), replace=False)]
    Ps = torch.tensor(np.concatenate([np.load(f)["psi"].reshape(-1, 4225) for f in sub]), device=DEV)
    Ps = Ps[torch.isfinite(Ps).all(1)]
    mean = Ps.mean(0); _, sv, Vh = torch.linalg.svd(Ps - mean, full_matrices=False)
    comps = Vh[:KMAX].contiguous(); del Ps
    X, C, S, shot, spl, keep_names, held = [], [], [], [], [], [], []
    skipped = 0
    for i, f in enumerate(files):
        d = np.load(f); psi = d["psi"].reshape(len(d["X"]), -1)
        if not (np.isfinite(psi).all() and np.isfinite(d["X"]).all()):
            skipped += 1
            continue
        j = len(keep_names); keep_names.append(names[i]); s = split_of(names[i])
        X.append(d["X"]); S.append(d["S"]); shot.append(np.full(len(psi), j, np.int32))
        spl.append(np.full(len(psi), {"train": 0, "val": 1, "test": 2}[s], np.int8))
        C.append(((torch.tensor(psi, device=DEV) - mean) @ comps.T).cpu().numpy())
        if s != "train":
            held.append(psi.astype(np.float32))
        if (i + 1) % 1000 == 0:
            print(f"[cache] {i+1}/{len(files)}", flush=True)
    X, C, S = np.concatenate(X), np.concatenate(C), np.concatenate(S)
    shot, spl = np.concatenate(shot), np.concatenate(spl)
    np.save(CACHE / "X.npy", X); np.save(CACHE / "C.npy", C); np.save(CACHE / "S.npy", S)
    np.save(CACHE / "shot.npy", shot); np.save(CACHE / "split.npy", spl)
    np.save(CACHE / "psi_heldout.npy", np.concatenate(held))          # rows = frames with split != 0, in order
    fn = (ROOT / "data" / "sophelio_proc" / "feature_names.txt").read_text().split()
    torch.save(dict(mean=mean.cpu(), comps=comps.cpu(), sv=sv[:KMAX].cpu(), names=keep_names, feature_names=fn,
                    skipped=skipped), CACHE / "meta.pt")
    print(f"[cache] done: frames {len(X)} shots {len(keep_names)} skipped {skipped}", flush=True)


class Data:
    def __init__(self):
        self.meta = torch.load(CACHE / "meta.pt", weights_only=False)
        self.X = np.load(CACHE / "X.npy", mmap_mode="r"); self.C = np.load(CACHE / "C.npy", mmap_mode="r")
        self.S = np.load(CACHE / "S.npy"); self.shot = np.load(CACHE / "shot.npy"); self.split = np.load(CACHE / "split.npy")
        self.psi_held = np.load(CACHE / "psi_heldout.npy", mmap_mode="r")
        self.held_pos = np.cumsum(self.split != 0) - 1       # row -> position in psi_heldout (valid for split != 0)
        self.fn = self.meta["feature_names"]
        self._edge = None

    def edge_feats(self) -> np.ndarray:
        """(N, 2) shot-phase features from the frame times only (inputs, never labels): time to the shot's last EFIT
        frame and the elapsed fraction of the shot. The worst 1 % of VALID frames sit at ramp-up / ramp-down."""
        if self._edge is None:
            t = np.asarray(self.X[:, self.fn.index("t")], np.float64)
            starts = np.r_[0, np.nonzero(np.diff(self.shot))[0] + 1]; n = np.diff(np.r_[starts, len(t)])
            t0 = np.repeat(np.minimum.reduceat(t, starts), n); t1 = np.repeat(np.maximum.reduceat(t, starts), n)
            self._edge = np.stack([t1 - t, (t - t0) / np.maximum(t1 - t0, 1e-9)], 1).astype(np.float32)
        return self._edge

    def inputs(self, rows: np.ndarray, fm: np.ndarray, edge: bool = False, hist: bool = False, thz: bool = False) -> np.ndarray:
        x = np.asarray(self.X[rows][:, fm])
        x = np.concatenate([x, self.edge_feats()[rows]], 1) if edge else x
        if hist:                                             # §41c long-window history (paper/strat/fusion/hist_feats.py)
            if getattr(self, "_hist", None) is None:
                self._hist = np.load(CACHE / "hist.npy", mmap_mode="r")
            x = np.concatenate([x, np.asarray(self._hist[rows])], 1)
        if thz:                                              # §41f Thomson on fixed physical Z / R (paper/strat/fusion/thz_feats.py)
            if getattr(self, "_thz", None) is None:
                self._thz = np.load(CACHE / "thz.npy", mmap_mode="r")
            x = np.concatenate([x, np.asarray(self._thz[rows])], 1)
        return x

    def psi_of(self, rows: np.ndarray) -> np.ndarray:
        assert (self.split[rows] != 0).all(), "full psi is only kept for VALID/TEST rows"
        return np.asarray(self.psi_held[self.held_pos[rows]])

    def feature_mask(self, lags: str, thomson: bool) -> np.ndarray:
        m = np.array([True] * len(self.fn))
        for i, n in enumerate(self.fn):
            if "_lag" in n:
                lag = int(n.split("_lag")[1].replace("ms", ""))
                m[i] = lags == "sym" or (lags == "back" and lag < 0)
            if n.startswith("thomson"):
                m[i] = thomson
        return m

    def shots_of(self, split: int) -> list[int]:
        return sorted(set(self.shot[self.split == split].tolist()))


# ----------------------------------------------------------------------------------------------- model
class Hybrid(nn.Module):
    def __init__(self, fin: int, k: int, width: int, depth: int, drop: float, quantum: str = "none",
                 qcfg: str | None = None, k_lo: int = 32, q_layers: int = 4, k_hi: int | None = None):
        super().__init__()
        from model import Bottleneck, build_config_core, parse_config
        self.inp = nn.Sequential(nn.Linear(fin, width), nn.GELU())
        self.blocks = nn.ModuleList([nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(), nn.Dropout(drop),
                                                   nn.Linear(width, width)) for _ in range(depth)])
        self.norm = nn.LayerNorm(width)
        self.psi = nn.Linear(width, k); self.scal = nn.Linear(width, len(HEADS)); self.skip = nn.Linear(fin, k)
        self.quantum, self.k_lo, self.k = quantum, min(k_lo, k), k
        self.k_hi = k if k_hi is None else min(k_hi, k)                 # srr writes into [k_lo, k_hi) (§29e: [0, 32))
        if quantum == "latent":
            core, out = build_config_core(parse_config(qcfg), width, q_layers)
            self.q = Bottleneck(width, core, out)
        elif quantum == "srr":
            core, out = build_config_core(parse_config(qcfg), width, q_layers)
            self.q_core = core
            self.q_write = nn.Linear(out, self.k_hi - self.k_lo)         # writes only into its band
            nn.init.zeros_(self.q_write.weight); nn.init.zeros_(self.q_write.bias)

    COUPLING = ("q", "q_core", "q_write")                         # parameter prefixes of the quantum / MLP coupling

    def latent(self, x):
        h = self.inp(x)
        for b in self.blocks:
            h = h + b(h)
        return self.norm(h)

    def head(self, h, skip):
        """Coupling + read-out from the trunk latent h and the linear skip(x) (split so a frozen trunk can be cached)."""
        if self.quantum == "latent":
            h = self.q(h)
        pc = self.psi(h) + skip
        if self.quantum == "srr":
            band = self.q_write(self.q_core(h))
            pc = torch.cat([pc[:, :self.k_lo], pc[:, self.k_lo:self.k_hi] + band, pc[:, self.k_hi:]], dim=1)
        return pc, self.scal(h)

    def forward(self, x):
        return self.head(self.latent(x), self.skip(x))


# ----------------------------------------------------------------------------------------------- train / eval
def r2(y, p) -> float:
    y, p = np.asarray(y, np.float64), np.asarray(p, np.float64); m = np.isfinite(y) & np.isfinite(p)
    return float(1 - ((y[m] - p[m]) ** 2).sum() / ((y[m] - y[m].mean()) ** 2).sum()) if m.sum() > 1 else float("nan")


DEFAULT = dict(k=128, width=1024, depth=4, drop=0.05, lr=1e-3, batch=4096, epochs=30, w_scal=1.0, gamma=0.0,
               lags="sym", thomson=True, quantum="none", qcfg=None, k_lo=32, q_layers=4, seed=0)


def moments64(X: np.ndarray, chunk: int = 131072):
    """Column mean and std in float64, chunked. float32 overflows on Thomson n_e / p_e (~1e19-1e20, squares ~1e39):
    the std became inf and those 84 features were silently standardised to 0 (bug in v1, fixed Sep 23)."""
    n = len(X); s = np.zeros(X.shape[1]); s2 = np.zeros(X.shape[1])
    for i in range(0, n, chunk):
        s += X[i:i + chunk].sum(0, dtype=np.float64)
    mu = s / n
    for i in range(0, n, chunk):
        s2 += ((X[i:i + chunk].astype(np.float64) - mu) ** 2).sum(0)
    return mu.astype(np.float32), (np.sqrt(s2 / n) + 1e-6).astype(np.float32)  # float32 is safe after centring


def train_eval(cfg: dict, data: Data, report=None, rungs=(), score_val: int = 40, score_test: int = 0,
               save: pathlib.Path | None = None) -> dict:
    """Train on TRAIN frames, evaluate on VALID (quick metrics + official scorer on `score_val` fixed shots) and,
    only if asked, on TEST. `report(epoch, proxy)` is called at each rung epoch (Optuna pruning hook)."""
    c = {**DEFAULT, **cfg}
    torch.manual_seed(c["seed"]); np.random.seed(c["seed"])
    fm = data.feature_mask(c["lags"], c["thomson"]); k = c["k"]; edge = bool(c.get("edge", False))
    tr = np.arange(len(data.split)) if c.get("all_data") else np.nonzero(data.split == 0)[0]    # §41q final build: all 7,041 shots
    if c.get("cv"):                                                    # §41r: out-of-fold psi - leave out stratified TRAIN fold k/K
        sys.path.insert(0, str(ROOT / "paper" / "strat" / "fusion")); import head_boundary as HB
        lab = np.load(ROOT / "runs" / "strat" / "fusion" / "boundary_labels_v8.npz"); fk, fK = map(int, str(c["cv"]).split("/"))
        held = HB.cv_shots(data, lab["rows_train"], lab["y_train"].astype(np.float64), fk, fK); tr = tr[~np.isin(data.shot[tr], held)]
    Xtr = data.inputs(tr, fm, edge, hist=bool(c.get("hist")), thz=bool(c.get("thz"))); mu, sd = moments64(Xtr)
    hidx = [LABELS.index(h) for h in HEADS]
    Str = data.S[tr][:, hidx]; smu, ssd = np.nanmean(Str, 0), np.nanstd(Str, 0) + 1e-6
    T = lambda z: torch.tensor(np.asarray(z), dtype=torch.float32, device=DEV)
    Xg = T(np.clip((Xtr - mu) / sd, -10, 10)); del Xtr
    Cg = T(data.C[tr][:, :k]); Sg = T((Str - smu) / ssd)
    sv = data.meta["sv"][:k].to(DEV)
    w = (sv / sv[0]) ** (-c["gamma"])                         # gamma=0: pixel MSE; >0 up-weights detail components
    w = w / w.mean()
    Wg = None                                                  # edge_w > 0: up-weight the first / last 10 % of each shot
    if c.get("edge_w", 0) > 0:
        pos = data.edge_feats()[tr, 1]
        Wg = T(1 + c["edge_w"] * ((pos < 0.1) | (pos > 0.9))); Wg = Wg / Wg.mean()
    model = Hybrid(Xg.shape[1], k, c["width"], c["depth"], c["drop"], c["quantum"], c["qcfg"], c["k_lo"], c["q_layers"],
                   c.get("k_hi")).to(DEV)
    frozen = bool(c.get("init")) and bool(c.get("freeze_trunk"))
    if c.get("init"):                                          # coupling stage: start from a stage-1 trunk (TRAIN-only fit)
        st = torch.load(c["init"], map_location=DEV, weights_only=False)["model"]
        missing, unexpected = model.load_state_dict(st, strict=False)
        assert not unexpected and all(m.split(".")[0] in Hybrid.COUPLING for m in missing), (missing, unexpected)
    if frozen:                                                 # train only the coupling on cached trunk latents
        for n_, p_ in model.named_parameters():
            p_.requires_grad_(n_.split(".")[0] in Hybrid.COUPLING)
        keep = torch.arange(0, len(Xg), int(c.get("frame_stride", 1)), device=DEV)
        with torch.no_grad():
            model.eval()
            Hc = torch.cat([model.latent(Xg[keep[i:i + 32768]]) for i in range(0, len(keep), 32768)])
            Kc = torch.cat([model.skip(Xg[keep[i:i + 32768]]) for i in range(0, len(keep), 32768)])
            model.train()
        Cg, Sg, Xg = Cg[keep], Sg[keep], None
        Wg = Wg[keep] if Wg is not None else None
    n_train = len(Cg)
    spe = n_train // c["batch"]; steps = c["epochs"] * spe
    fast = bool(c.get("fast", False)) and not frozen           # Oct 3: bf16 autocast + fused AdamW + compile (default off)
    opt = torch.optim.AdamW([p_ for p_ in model.parameters() if p_.requires_grad], lr=c["lr"], weight_decay=1e-4, fused=fast)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=c["lr"], total_steps=steps, pct_start=0.05)
    fwd = torch.compile(model) if fast else model
    torch.backends.cuda.matmul.allow_tf32 = fast
    denom = Cg.pow(2).sum(1).mean()
    t0 = time.time(); model.train()
    va_rows = np.nonzero(data.split == 1)[0]
    gt_va = data.psi_of(va_rows)

    def predict(rows):
        model.eval()
        with torch.no_grad():
            xs = T(np.clip((data.inputs(rows, fm, edge, hist=bool(c.get("hist")), thz=bool(c.get("thz"))) - mu) / sd, -10, 10))
            pc, ps = zip(*[model(xs[i:i + 32768]) for i in range(0, len(xs), 32768)])
            pc, ps = torch.cat(pc), torch.cat(ps).cpu().numpy() * ssd + smu
        model.train()
        return pc, ps

    def quick():
        pc, ps = predict(va_rows)
        comps = data.meta["comps"][:k].to(DEV); mean = data.meta["mean"].to(DEV)
        psi = (pc @ comps + mean).cpu().numpy()
        q = dict(r2_psi=r2(gt_va, psi))
        for j, h in enumerate(HEADS):
            q[f"r2_{h}"] = r2(data.S[va_rows, LABELS.index(h)], ps[:, j])
        q["proxy"] = 0.55 * q["r2_psi"] + 0.15 * 0.5 * (q["r2_efit_q95"] + q["r2_efit_beta_n"]) + \
            0.2 * np.mean([q["r2_efit_li"], q["r2_efit_r_axis"], q["r2_efit_z_axis"]])
        return q

    opw = float(c.get("opoint_w", 0.0))                         # §37m: O-point loss, |grad psi_pred| at the true axis
    if opw > 0:
        g = np.load(ROOT / "runs" / "strat" / "fusion" / "efit_grid.npz"); Rg, Zg = g["R"], g["Z"]
        dR, dZ = float(Rg[1] - Rg[0]), float(Zg[1] - Zg[0])
        basis = torch.cat([data.meta["mean"][None], data.meta["comps"][:k]]).to(DEV).float().reshape(-1, 65, 65)  # [Z, R]
        bZ = (torch.roll(basis, -1, 1) - torch.roll(basis, 1, 1)) / (2 * dZ)
        bR = (torch.roll(basis, -1, 2) - torch.roll(basis, 1, 2)) / (2 * dR)
        ax = np.stack([data.S[tr][:, LABELS.index("efit_r_axis")], data.S[tr][:, LABELS.index("efit_z_axis")]], 1)
        okax = T(np.isfinite(ax).all(1).astype(np.float32)); ax = np.nan_to_num(ax, nan=float(Rg.mean()))
        fr = T(np.clip((ax[:, 0] - Rg[0]) / dR, 1, 62.999)); fz = T(np.clip((ax[:, 1] - Zg[0]) / dZ, 1, 62.999))

        def opoint(idx, pc):
            r, z = fr[idx], fz[idx]; i, j = r.long(), z.long(); a, b = (r - i)[:, None], (z - j)[:, None]
            def at(B):                                        # bilinear basis values at the axis -> (batch, k+1)
                return ((1 - a) * (1 - b) * B[:, j, i].T + a * (1 - b) * B[:, j, i + 1].T
                        + (1 - a) * b * B[:, j + 1, i].T + a * b * B[:, j + 1, i + 1].T)
            coef = torch.cat([torch.ones_like(pc[:, :1]), pc], 1)
            gr, gz = (coef * at(bR)).sum(1), (coef * at(bZ)).sum(1)
            return ((gr.square() + gz.square()) * okax[idx]).sum() / okax[idx].sum().clamp(min=1) / 0.15
    lcw = float(c.get("lcfs_w", 0.0))                          # Fusion B1: psi error on the TRUE LCFS contour (32 points)
    lsw = float(c.get("lset_w", 0.0))                          # §42b: level-set loss (offset-free; see lset_loss)
    if lcw > 0 or lsw > 0:
        g = np.load(ROOT / "runs" / "strat" / "fusion" / "efit_grid.npz"); Rg, Zg = g["R"], g["Z"]
        dR, dZ = float(Rg[1] - Rg[0]), float(Zg[1] - Zg[0])
        lc = np.load(CACHE / "lcfs32.npy", mmap_mode="r")[tr]                 # (n_train, 32, 2) R, Z
        okl = T(np.isfinite(lc).all(2).astype(np.float32)); lc = np.nan_to_num(lc, nan=float(Rg.mean()))
        lr_ = T(np.clip((lc[..., 0] - Rg[0]) / dR, 0, 63.999)); lz_ = T(np.clip((lc[..., 1] - Zg[0]) / dZ, 0, 63.999)); del lc
        basl = torch.cat([data.meta["mean"][None], data.meta["comps"][:k]]).to(DEV).float().reshape(-1, 65, 65)   # [Z, R]

        def lcfs_loss(idx, pc):
            r, z = lr_[idx], lz_[idx]; i, j = r.long(), z.long(); a, b = r - i, z - j        # (B, 32)
            def at(B_):                                   # bilinear basis values at the contour points -> (B, 32, k+1)
                return ((1 - a)[..., None] * (1 - b)[..., None] * B_[:, j, i].permute(1, 2, 0) + a[..., None] * (1 - b)[..., None] * B_[:, j, i + 1].permute(1, 2, 0)
                        + (1 - a)[..., None] * b[..., None] * B_[:, j + 1, i].permute(1, 2, 0) + a[..., None] * b[..., None] * B_[:, j + 1, i + 1].permute(1, 2, 0))
            Bv = at(basl); one = torch.ones_like(pc[:, :1])
            d = (Bv * torch.cat([one, pc], 1)[:, None, :]).sum(-1) - (Bv * torch.cat([one, Cg[idx]], 1)[:, None, :]).sum(-1)
            m_ = okl[idx]
            return (d.square() * m_).sum() / m_.sum().clamp(min=1) / (denom / basl[0].numel())

        bRl = (torch.roll(basl, -1, 2) - torch.roll(basl, 1, 2)) / (2 * dR); bZl = (torch.roll(basl, -1, 1) - torch.roll(basl, 1, 1)) / (2 * dZ)

        def lset_loss(idx, pc):
            """§42b: the TRUE LCFS should be a level set of the predicted psi. Spread of psi_pred along the contour (minus its contour
            mean: offset- and level-free) over |grad psi_true| = normal displacement of the predicted level set (m), / 1.7 m (rgeo)
            -> D_LCFS units, squared. |grad| floored at 0.25 x the frame's median (X-point)."""
            r, z = lr_[idx], lz_[idx]; i, j = r.long(), z.long(); a, b = r - i, z - j
            def at(B_):
                return ((1 - a)[..., None] * (1 - b)[..., None] * B_[:, j, i].permute(1, 2, 0) + a[..., None] * (1 - b)[..., None] * B_[:, j, i + 1].permute(1, 2, 0)
                        + (1 - a)[..., None] * b[..., None] * B_[:, j + 1, i].permute(1, 2, 0) + a[..., None] * b[..., None] * B_[:, j + 1, i + 1].permute(1, 2, 0))
            one = torch.ones_like(pc[:, :1]); cp, ct = torch.cat([one, pc], 1)[:, None, :], torch.cat([one, Cg[idx]], 1)[:, None, :]
            vp = (at(basl) * cp).sum(-1); g = torch.sqrt((at(bRl) * ct).sum(-1).square() + (at(bZl) * ct).sum(-1).square())
            m_ = okl[idx]; g = torch.maximum(g, 0.25 * torch.where(m_ > 0, g, torch.nan).nanmedian(1).values[:, None]).clamp(min=1e-6)
            dev = vp - (vp * m_).sum(1, keepdim=True) / m_.sum(1, keepdim=True).clamp(min=1)
            return ((dev / g / 1.7).square() * m_).sum() / m_.sum().clamp(min=1)
    rung_set = set(rungs); history = []
    ema = None                                                 # ema > 0: evaluate / save an EMA of the weights
    if c.get("ema", 0) > 0:
        from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
        ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(c["ema"]))
    for step in range(steps):
        idx = torch.randint(0, n_train, (c["batch"],), device=DEV)
        if frozen:
            pc, ps = model.head(Hc[idx], Kc[idx])
        else:
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=fast):
                pc, ps = fwd(Xg[idx])
            pc, ps = pc.float(), ps.float()
        lp_i = (((pc - Cg[idx]) ** 2) * w).sum(1)
        lp = (lp_i * Wg[idx]).mean() / denom if Wg is not None else lp_i.mean() / denom
        tgt = Sg[idx]; msk = torch.isfinite(tgt)
        ls = ((ps - torch.nan_to_num(tgt)) ** 2 * msk).sum() / msk.sum().clamp(min=1)
        loss = lp + c["w_scal"] * ls
        if opw > 0 and not frozen:
            loss = loss + opw * opoint(idx, pc)
        if lcw > 0 and not frozen:
            loss = loss + lcw * lcfs_loss(idx, pc)
        if lsw > 0 and not frozen:
            loss = loss + lsw * lset_loss(idx, pc)
        opt.zero_grad(); loss.backward()
        if float(c.get("clip", 0)) > 0:                          # doc 35 §38c: grad-norm clipping (default off)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(c["clip"]))
        opt.step(); sched.step()
        if ema is not None:
            ema.update_parameters(model)
        ep = (step + 1) / spe
        if (step + 1) % spe == 0 and int(ep) in rung_set and report is not None:
            qv = quick(); history.append((int(ep), qv["proxy"])); report(int(ep), qv["proxy"])
    if ema is not None:
        model.load_state_dict(ema.module.state_dict()); del ema
    res = dict(cfg=c, params=sum(p.numel() for p in model.parameters()), train_seconds=time.time() - t0,
               trainable_params=sum(p.numel() for p in model.parameters() if p.requires_grad),
               valid_quick=quick(), history=history)
    comps = data.meta["comps"][:k].to(DEV); mean = data.meta["mean"].to(DEV)

    def official(split: int, n: int) -> dict:
        sys.path.insert(0, str(ROOT / "paper" / "sophelio"))
        import train_hybrid as TH  # noqa: E402  (vendored official scorer)
        shots = data.shots_of(split)
        pick = [shots[i] for i in np.random.default_rng(1).permutation(len(shots))[:n]]
        gts, preds = [], []
        for s in pick:
            rows = np.nonzero(data.shot == s)[0]
            pc, ps = predict(rows)
            psi = (pc @ comps + mean).cpu().numpy().reshape(-1, 65, 65)
            gt_psi = data.psi_of(rows).reshape(-1, 65, 65)
            gts.append(({"psi": gt_psi.astype(np.float32), "q95": data.S[rows, LABELS.index("efit_q95")].astype(np.float64),
                         "betaN": data.S[rows, LABELS.index("efit_beta_n")].astype(np.float64)}, data.meta["names"][s]))
            preds.append({"psirz": psi.astype(np.float64), "q95": ps[:, HEADS.index("efit_q95")].astype(np.float64),
                          "betaN": ps[:, HEADS.index("efit_beta_n")].astype(np.float64)})
        return TH.official_score(gts, preds, ROOT / "runs" / "sophelio_sota" / "ref_cache")

    if score_val:
        res["valid_official"] = official(1, score_val)
    if score_test:
        res["test_official"] = official(2, score_test)
    if save is not None:
        torch.save(dict(model=model.state_dict(), cfg=c, mu=mu, sd=sd, smu=smu, ssd=ssd, fm=fm), save)
    del Xg, Cg, Sg, model, opt
    if frozen:
        del Hc, Kc
    torch.cuda.empty_cache()
    return res
