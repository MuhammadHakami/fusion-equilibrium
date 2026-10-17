"""Small FNO2d trunk with a swappable bottleneck module.

Variants (identical everywhere except the bottleneck):
  none     — pure FNO trunk
  mlp      — pooled-latent MLP residual, parameter-matched to the quantum variants
  hea<n>   — QuantumBlock HEA ansatz, n qubits
  ryd<n>   — QuantumBlock Rydberg-analog ansatz, n qubits
  gutz30   — mr30ryd registers, mean-field (Gutzwiller) propagation: no entanglement

Bottleneck interface mirrors the paper architecture: global pool -> module ->
zero-init gated residual broadcast back onto the feature map.
"""

import torch
import torch.nn as nn

from qlayer import QuantumBlock


class SpectralConv2d(nn.Module):
    def __init__(self, ch: int, m1: int, m2: int):
        super().__init__()
        self.m1, self.m2 = m1, m2
        scale = 1.0 / (ch * ch)
        self.w = nn.Parameter(
            scale * torch.randn(2, ch, ch, m1, m2, 2))  # [corner, i, o, kx, ky, re/im]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        xf = torch.fft.rfft2(x, norm="ortho")
        out = torch.zeros(b, c, h, w // 2 + 1, dtype=torch.cfloat, device=x.device)
        w0 = torch.view_as_complex(self.w[0])
        w1 = torch.view_as_complex(self.w[1])
        out[:, :, : self.m1, : self.m2] = torch.einsum(
            "bihw,iohw->bohw", xf[:, :, : self.m1, : self.m2], w0)
        out[:, :, -self.m1 :, : self.m2] = torch.einsum(
            "bihw,iohw->bohw", xf[:, :, -self.m1 :, : self.m2], w1)
        return torch.fft.irfft2(out, s=(h, w), norm="ortho")


class FNOBlock(nn.Module):
    def __init__(self, ch: int, m1: int, m2: int):
        super().__init__()
        self.spec = SpectralConv2d(ch, m1, m2)
        self.skip = nn.Conv2d(ch, ch, 1)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.spec(x) + self.skip(x))


class Bottleneck(nn.Module):
    """pool -> core -> zero-init gated residual broadcast."""

    def __init__(self, ch: int, core: nn.Module, core_out: int):
        super().__init__()
        self.core = core
        self.up = nn.Linear(core_out, ch)
        self.alpha = nn.Parameter(torch.zeros(ch))

    def forward(self, x):
        spatial = tuple(range(2, x.dim()))           # 2D or 3D fields; () for vector latents
        h = x.mean(dim=spatial) if spatial else x    # (B, ch)  (mean(dim=()) would reduce everything)
        o = self.up(self.core(h))                    # (B, ch)
        gate = (self.alpha * o).reshape(*o.shape, *([1] * len(spatial)))
        return x + gate


class MLPCore(nn.Module):
    """Classical control matched to an explicit parameter budget."""

    def __init__(self, ch: int, out_n: int, target_params: int):
        super().__init__()
        k = max(4, (target_params - out_n) // (ch + out_n + 1))
        self.net = nn.Sequential(nn.Linear(ch, k), nn.Tanh(), nn.Linear(k, out_n))

    def forward(self, h):
        return self.net(h)


def _count(m) -> int:
    return sum(p.numel() for p in m.parameters())


def parse_config(variant: str) -> dict | None:
    """Wave-G register grammar (Sep 21): 'q30[-geo=chain][-sp=0.7][-part=6x5][-link=cascade][-gate]'
    -> dict(n, registers, geometry, spacing, link, gate, total) or None if `variant` is not a q-config.
    'q30' alone = the reference mr30ryd register (3x10 grid, spacing 1, independent, ungated).
    The matched control is 'q30...-mlp' (same suffix): an MLPCore with the parameter count of that
    exact configuration."""
    if not variant.startswith("q"):
        return None
    head, *opts = variant.split("-")
    total = int(head[1:])
    cfg = dict(total=total, registers=3, n=total // 3, geometry="grid", spacing=1.0,
               link="independent", gate=False, control=False, budget=0, random_mask=False)
    for o in opts:
        if o == "mlp":
            cfg["control"] = True               # matched to THIS configuration's register
        elif o == "mlpref":
            cfg["control"] = "ref"              # matched to the reference register (HPO protocol)
        elif o == "gate":
            cfg["gate"] = True
        elif o.startswith("geo="):
            cfg["geometry"] = o[4:]
        elif o.startswith("sp="):
            cfg["spacing"] = float(o[3:])
        elif o.startswith("part="):
            r, n = o[5:].split("x"); cfg["registers"], cfg["n"] = int(r), int(n)
            assert cfg["registers"] * cfg["n"] == total, f"{o} does not partition {total} qubits"
        elif o.startswith("link="):
            cfg["link"] = o[5:]
        elif o.startswith("k="):
            cfg["budget"] = int(o[2:])          # atoms driven per register per input (exact top-k)
        elif o == "randk":
            cfg["random_mask"] = True           # control: fixed random k-subset for every input
        elif o.startswith("L="):
            cfg["layers"] = int(o[2:])          # register depth (Trotter layers); default = trainer's q_layers
        else:
            raise ValueError(f"unknown option {o!r} in variant {variant!r}")
    return cfg


def build_config_core(cfg: dict, width: int, q_layers: int = 4):
    """(core, core_out) for a parsed wave-G configuration. '-mlp' = control matched to this configuration's
    register; '-mlpref' = control matched to the reference register (fixed budget for the HPO protocol)."""
    from qlayer import MultiRegister
    if cfg["control"] == "ref":
        # HPO control: budget pinned to the REFERENCE register (q30 at the trainer's default depth) so the
        # control's capacity is fixed while the Q arm tunes its own depth/geometry (its count is reported)
        ref = MultiRegister(width, cfg["total"] // 3, 3, q_layers, "rydberg")
        return MLPCore(width, cfg["total"], _count(ref)), cfg["total"]
    q_layers = cfg.get("layers", q_layers)
    if cfg["geometry"] == "3d":                  # doc 35 §23: per-input 3-D positions, detunings and tethers
        from qlayer import TetheredGeometry3D
        reg = TetheredGeometry3D(width, cfg["n"], cfg["registers"], q_layers)
        if cfg["control"]:
            return MLPCore(width, cfg["total"], _count(reg)), cfg["total"]
        return reg, cfg["total"]
    reg = MultiRegister(width, cfg["n"], cfg["registers"], q_layers, "rydberg", geometry=cfg["geometry"],
                        spacing=cfg["spacing"], link=cfg["link"], gate=cfg["gate"], budget=cfg["budget"],
                        random_mask=cfg["random_mask"])
    if cfg["control"]:                       # per-configuration matched control (wave G, as pre-registered)
        return MLPCore(width, cfg["total"], _count(reg)), cfg["total"]
    return reg, cfg["total"]


def build_model(variant: str, in_ch: int, width: int = 24, modes: int = 12,
                q_layers: int = 4) -> nn.Module:
    blocks_pre = nn.Sequential(FNOBlock(width, modes, modes),
                               FNOBlock(width, modes, modes))
    blocks_post = nn.Sequential(FNOBlock(width, modes, modes),
                                FNOBlock(width, modes, modes))
    from qlayer import ConditionedGeometryBlock, KMixtureRegister, MultiRegister

    if variant == "none":
        mid: nn.Module = nn.Identity()
    elif parse_config(variant) is not None:
        core, out_n = build_config_core(parse_config(variant), width, q_layers)
        mid = Bottleneck(width, core, out_n)
    elif variant.startswith("cg30"):
        dims = 3 if variant.endswith("3d") else 2
        core = nn.ModuleList([ConditionedGeometryBlock(width, 10, q_layers, dims)
                              for _ in range(3)])

        class _CG(nn.Module):
            def __init__(self, blocks):
                super().__init__()
                self.blocks = blocks

            def forward(self, h):
                return torch.cat([b(h) for b in self.blocks], dim=-1)

        mid = Bottleneck(width, _CG(core), 30)
    elif variant == "kmix30":
        mid = Bottleneck(width, KMixtureRegister(width, 10, 3, 4, q_layers), 30)
    elif variant == "mlp":
        ref = QuantumBlock(width, 16, q_layers, "hea")
        mid = Bottleneck(width, MLPCore(width, 16, _count(ref)), 16)
    elif variant == "mlp30":
        ref = MultiRegister(width, 10, 3, q_layers, "hea")
        mid = Bottleneck(width, MLPCore(width, 30, _count(ref)), 30)
    elif variant.startswith("mr10"):
        ansatz = "hea" if variant.endswith("hea") else "rydberg"
        mid = Bottleneck(width, MultiRegister(width, 10, 1, q_layers, ansatz), 10)
    elif variant.startswith("mr20"):
        ansatz = "hea" if variant.endswith("hea") else "rydberg"
        mid = Bottleneck(width, MultiRegister(width, 10, 2, q_layers, ansatz), 20)
    elif variant.startswith("gutz30"):
        # Gutzwiller control: mr30ryd registers with mean-field propagation
        mid = Bottleneck(width, MultiRegister(width, 10, 3, q_layers, "rydberg_mf"), 30)
    elif variant.startswith("mr30"):
        ansatz = "hea" if variant.endswith("hea") else "rydberg"
        mid = Bottleneck(width, MultiRegister(width, 10, 3, q_layers, ansatz), 30)
    elif variant.startswith(("hea", "ryd")):
        n = int(variant[3:])
        ansatz = "hea" if variant.startswith("hea") else "rydberg"
        mid = Bottleneck(width, QuantumBlock(width, n, q_layers, ansatz), n)
    else:
        raise ValueError(f"unknown variant {variant}")

    class FNO(nn.Module):
        def __init__(self):
            super().__init__()
            self.lift = nn.Conv2d(in_ch, width, 1)
            self.pre, self.mid, self.post = blocks_pre, mid, blocks_post
            self.proj = nn.Sequential(nn.Conv2d(width, 64, 1), nn.GELU(),
                                      nn.Conv2d(64, in_ch, 1))

        def forward(self, x):
            z = self.pre(self.lift(x))
            z = self.mid(z)
            return self.proj(self.post(z))

    return FNO()
