"""Doc 35 §47b: JEPA pretraining of the VisionHOPE shot-map encoder (Atom-JEPA recipe, arXiv 2610.08400, baseline/atom-jepa, MIT).

Atom-JEPA splits a structure into a k-hop ego subgraph and its complement, encodes the context, and a predictor regresses the target
encoder's latents (an EMA copy, full structure) at two scales - each atom and the mean over the target part; views alternate; MSE.
Here the structure is a shot crop on the VisionHOPE map (rows = frames, columns = learned tokens, quota_fusion.VHopeMap): the
target part is a contiguous block of frames around an anchor frame (the ego-net analogue in time), its rows are replaced by a learned
mask token before the encoder, a one-block VisionHOPE predictor reads the context map, and the loss is MSE to the EMA target's
layer-normed latents on the block rows (per frame) plus on the block mean (the pooled scale); both the block and its complement
serve as target (alternate steps). Inputs only - no labels; in CV only the fold's training shots.

    from vhope_jepa import pretrain; pretrain(net.bb, Sh, shot_idx, steps)      (called by quota_fusion.train_seq when p["jepa"] > 0)
"""
from __future__ import annotations

import copy
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _tokens(enc, xs):
    B, L, _ = xs.shape
    return enc.ln(F.gelu(enc.tok(xs)).view(B, L, enc.G, enc.c)).permute(0, 3, 1, 2).contiguous()       # (B, c, L, G)


def _blocks(enc, h):
    for b in enc.blocks:
        h = b(h)
    return h


def pretrain(enc, Sh, shots, steps, crop=32, per=128, lr=5e-4, tau=(0.996, 1.0), block=(4, 12), seed=0, log=""):
    """In-place JEPA pretraining of enc (VHopeMap) on random crops of the given shots; returns the final loss."""
    from visionhope.models.block import VisionHOPEBlock                         # on sys.path: VHopeMap's import put it there
    rng = np.random.default_rng(seed); dev = torch.device("cuda")
    target = copy.deepcopy(enc).eval()
    for q in target.parameters():
        q.requires_grad_(False)
    mixer = enc.blocks[0].mixer_dim if len(enc.blocks) else None
    pred = VisionHOPEBlock(enc.c, head_dim=16, mixer_dim=mixer).to(dev); mask_tok = nn.Parameter(torch.zeros(enc.c, device=dev))
    params = list(enc.parameters()) + list(pred.parameters()) + [mask_tok]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.05)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.05); t0 = time.time(); enc.train()
    for step in range(steps):
        xs, valid, _ = Sh.crops(rng.choice(shots, per, replace=False), crop, rng)
        B, L = valid.shape; n = valid.sum(1)
        a = (rng.random(B) * n.cpu().numpy()).astype(np.int64); w = rng.integers(block[0], block[1] + 1, B)
        ar = torch.arange(L, device=dev)[None]; lo = torch.tensor(a - w // 2, device=dev)[:, None]
        blk = (ar >= lo) & (ar < lo + torch.tensor(w, device=dev)[:, None]) & valid
        tgt_rows = blk if step % 2 == 0 else (~blk & valid)                    # alternate: block / complement as the target part
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            ht = _blocks(target, _tokens(target, xs)).float()                     # (B, c, L, G) full crop
            ht = F.layer_norm(ht.permute(0, 2, 3, 1), (enc.c,))                  # (B, L, G, c)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h0 = _tokens(enc, xs); m = tgt_rows[:, None, :, None]
            h0 = torch.where(m, mask_tok.to(h0.dtype)[None, :, None, None].expand_as(h0), h0)
            hp = pred(_blocks(enc, h0)).float().permute(0, 2, 3, 1)               # (B, L, G, c)
        hp = F.layer_norm(hp, (enc.c,)); mt = tgt_rows[..., None, None].float(); cnt = tgt_rows.sum(1).clamp(min=1)[:, None, None]
        l_frame = ((hp - ht).square() * mt).sum() / (mt.sum() * enc.G * enc.c).clamp(min=1)
        l_mean = (((hp * mt).sum(1) / cnt - (ht * mt).sum(1) / cnt).square()).mean()        # pooled (block-mean) scale
        loss = l_frame + l_mean
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sch.step()
        t_ = tau[0] + (tau[1] - tau[0]) * step / max(steps - 1, 1)
        with torch.no_grad():
            for pt, po in zip(target.parameters(), enc.parameters()):
                pt.lerp_(po, 1 - t_)
        if (step + 1) % max(1, steps // 10) == 0:
            with torch.no_grad():
                sd_ = float(ht[tgt_rows].std(0).mean())                           # collapse check: target latent spread per dim
            print(f"[jepa]{log} step {step + 1}/{steps} loss {loss.item():.4f} (frame {l_frame.item():.4f} mean {l_mean.item():.4f}) "
                  f"target sd {sd_:.3f} {time.time() - t0:.0f}s", flush=True)
    return float(loss.item())
