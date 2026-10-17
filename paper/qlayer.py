"""Batched, differentiable statevector quantum layers (pure PyTorch).

Two ansaetze, both input-conditioned (angles depend on the pooled latent h):
- HEA: L x [RY, RZ per qubit + CZ-ring diagonal mask]
- Rydberg-analog: L x [global RX drive, per-qubit detuning RZ (input-conditioned),
  diagonal blockade phase exp(-i tau * sum_{i<j} C6/|ri-rj|^6 n_i n_j)]
  with TRAINABLE 2D atom positions (geometry-as-parameter).
- Rydberg mean-field ("rydberg_mf", Gutzwiller control): identical parameters
  and drive, but the blockade term is replaced by its mean-field decoupling
  theta_i = tau * sum_{j!=i} V_ij <n_j>; the register stays a product state
  (interactions kept, entanglement forbidden).

State tensor: (B, 2^n) complex64. All ops autograd-safe (no in-place).
cudaq must NEVER be imported in this process (LLVM clash) — cross-checks are
subprocess-only (see bench/cudaq_check.py).
"""

import os

import numpy as np
import torch
import torch.nn as nn


def _bits(n: int) -> torch.Tensor:
    """(2^n, n) float matrix of basis-state bits, qubit 0 = MSB."""
    idx = torch.arange(2**n)
    return torch.stack([(idx >> (n - 1 - i)) & 1 for i in range(n)], -1).float()


def apply_1q(state: torch.Tensor, gate: torch.Tensor, k: int, n: int) -> torch.Tensor:
    """gate: (B, 2, 2) per-sample, or (2, 2) shared. state: (B, 2^n)."""
    b = state.shape[0]
    s = state.reshape(b, 2**k, 2, 2 ** (n - k - 1))
    if gate.dim() == 2:
        out = torch.einsum("ij,bkjr->bkir", gate, s)
    else:
        out = torch.einsum("bij,bkjr->bkir", gate, s)
    return out.reshape(b, -1)


def ry_gate(theta: torch.Tensor) -> torch.Tensor:
    """theta: (B,) -> (B, 2, 2) complex64."""
    c, s = torch.cos(theta / 2), torch.sin(theta / 2)
    row0 = torch.stack([c, -s], -1)
    row1 = torch.stack([s, c], -1)
    return torch.stack([row0, row1], -2).to(torch.complex64)


def rx_gate(theta: torch.Tensor) -> torch.Tensor:
    c = torch.cos(theta / 2).to(torch.complex64)
    s = (-1j) * torch.sin(theta / 2).to(torch.complex64)
    row0 = torch.stack([c, s], -1)
    row1 = torch.stack([s, c], -1)
    return torch.stack([row0, row1], -2)


def rz_phases(theta: torch.Tensor) -> torch.Tensor:
    """theta: (B,) -> (B, 2) diagonal of RZ."""
    e = torch.exp(-0.5j * theta.to(torch.complex64))
    return torch.stack([e, e.conj()], -1)


def apply_rz(state: torch.Tensor, theta: torch.Tensor, k: int, n: int) -> torch.Tensor:
    b = state.shape[0]
    s = state.reshape(b, 2**k, 2, 2 ** (n - k - 1))
    d = rz_phases(theta).reshape(b, 1, 2, 1)
    return (s * d).reshape(b, -1)


def _kron_gates(g: torch.Tensor) -> torch.Tensor:
    """g (B, m, 2, 2) per-qubit gates (qubit 0 = MSB) -> (B, 2^m, 2^m) = g_0 (x) g_1 (x) ... (x) g_{m-1}."""
    u = g[:, 0]
    for k in range(1, g.shape[1]):
        b, d = u.shape[0], u.shape[1]
        u = (u[:, :, None, :, None] * g[:, k, None, :, None, :]).reshape(b, 2 * d, 2 * d)
    return u


def apply_layer_1q(state: torch.Tensor, g: torch.Tensor, n: int) -> torch.Tensor:
    """One layer of single-qubit gates g (B, n, 2, 2) on every qubit, as two batched (2^h x 2^h) matmuls on the
    state reshaped (B, 2^h, 2^(n-h)) instead of n passes of apply_1q (gates on different qubits commute, so the
    product is exact; only the float summation order differs). QW_QFAST=1 selects it in QuantumBlock.probs."""
    h = n // 2; b = state.shape[0]
    s = state.reshape(b, 2**h, 2 ** (n - h))
    return (_kron_gates(g[:, :h]) @ s @ _kron_gates(g[:, h:]).transpose(1, 2)).reshape(b, -1)


def fast_layers() -> bool:
    return os.environ.get("QW_QFAST", "") == "1"


def readout_mode(explicit: str | None = None) -> str:
    """Readout family for every register (FW-A, Sep 19 sweep): env QREADOUT or explicit.
    z     : fixed <Z_i> (the pre-registered default; all published numbers)
    diag  : trainable diagonal observables O_i = Z_i + D_i on the computational basis
            (diagonal ANO, arXiv 2605.15410: same function space as full k-local ANO),
            2^n * n parameters per register, zero-init delta => identical to `z` at init
    cond2 : input-conditioned 2-local observables O_i(h) = Z_i + sum_S c_{i,S}(h) Z_S over
            all 1- and 2-local Z-strings S (QFWP-ANO style, arXiv 2609.18655), coefficients
            tanh-bounded from a small hypernet with zero-init output => identical at init
    Parameter matching is automatic: the MLP control budget is _count() of a reference
    register built under the same mode."""
    m = (explicit or os.environ.get("QREADOUT", "z")).strip().lower()
    if m not in ("z", "diag", "cond2"):
        raise ValueError(f"QREADOUT must be z|diag|cond2, got {m!r}")
    return m


class Readout(nn.Module):
    """Register readout from basis probabilities (B, 2^n) [+ latent h] -> features (B, n)."""

    HYPER = 16

    def __init__(self, n: int, d_in: int, mode: str):
        super().__init__()
        self.n, self.mode = n, mode
        bits = _bits(n)
        zdiag = 1.0 - 2.0 * bits                                    # (2^n, n)  <Z_i>
        iu = torch.triu_indices(n, n, offset=1)
        zz = zdiag[:, iu[0]] * zdiag[:, iu[1]]                       # (2^n, P2) <Z_i Z_j>
        # non-persistent: state dicts of `z`-readout registers stay byte-identical to the pre-FW-A checkpoints
        self.register_buffer("zall", torch.cat([zdiag, zz], 1), persistent=False)   # (2^n, P)
        self.P = n + iu.shape[1]
        if mode == "diag":
            self.delta = nn.Parameter(torch.zeros(2**n, n))
        elif mode == "cond2":
            self.hyper = nn.Sequential(nn.Linear(d_in, self.HYPER), nn.Tanh(),
                                       nn.Linear(self.HYPER, n * self.P))
            nn.init.zeros_(self.hyper[-1].weight)
            nn.init.zeros_(self.hyper[-1].bias)

    def forward(self, probs: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        n = self.n
        if self.mode == "z":
            return probs @ self.zall[:, :n]
        if self.mode == "diag":
            return probs @ (self.zall[:, :n] + self.delta)
        feats = probs @ self.zall                                    # (B, P) all <Z_S>
        return self._cond2(feats, h)

    def _cond2(self, feats: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        # coefficients in (-1, 1), averaged over the P strings => O_i(h) stays in [-2, 2], same scale as <Z_i>
        c = torch.tanh(self.hyper(h)).reshape(h.shape[0], self.n, self.P)
        return feats[:, : self.n] + torch.einsum("bip,bp->bi", c, feats) / self.P

    def from_product_state(self, z: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """Mean-field (product-state) path: <Z_i Z_j> = <Z_i><Z_j>. `diag` needs the
        full basis distribution and is not defined for the Gutzwiller control."""
        if self.mode == "z":
            return z
        if self.mode == "diag":
            raise NotImplementedError("QREADOUT=diag is not defined for the mean-field control")
        iu = torch.triu_indices(self.n, self.n, offset=1, device=z.device)
        feats = torch.cat([z, z[:, iu[0]] * z[:, iu[1]]], 1)
        return self._cond2(feats, h)


def topology_positions(n: int, geometry: str = "grid", spacing: float = 1.0) -> torch.Tensor:
    """(n, 2) atom positions for a topology family (nearest-neighbour distance = `spacing`, in
    blockade-radius units). The Rydberg graph is V_ij = |ri - rj|^-6, so the family fixes which
    atoms talk (chain: 2 neighbours; ring: 2 + a loop; grid: up to 4; tri: up to 6) and the spacing
    fixes how strongly (0.7: 8.5x the unit coupling; 1.5: 0.09x, near product state)."""
    if geometry == "grid":
        side = int(np.ceil(np.sqrt(n)))
        g = torch.stack(torch.meshgrid(torch.arange(side).float(), torch.arange(side).float(),
                                       indexing="ij"), -1).reshape(-1, 2)[:n]
    elif geometry == "chain":
        g = torch.stack([torch.arange(n).float(), torch.zeros(n)], -1)
    elif geometry == "ring":
        r = 1.0 / (2.0 * np.sin(np.pi / n))                          # unit neighbour spacing
        th = 2.0 * np.pi * torch.arange(n).float() / n
        g = torch.stack([r * torch.cos(th), r * torch.sin(th)], -1)
    elif geometry == "tri":
        side = int(np.ceil(np.sqrt(n)))
        rows = []
        for i in range(side):
            for j in range(side):
                rows.append([j + 0.5 * (i % 2), i * np.sqrt(3) / 2])
        g = torch.tensor(rows[:n], dtype=torch.float32)
    else:
        raise ValueError(f"unknown geometry {geometry!r} (grid|chain|ring|tri); a 2-D star with the hub as every leaf's nearest neighbour is not packable beyond 5 leaves")
    return g * spacing


class QuantumBlock(nn.Module):
    """h (B, d_in) -> expectation readout (B, n_qubits). Angles are
    input-conditioned: phi = base + pi*tanh(W h) (data re-uploading each layer).
    Readout family via `readout` / env QREADOUT (see readout_mode)."""

    def __init__(self, d_in: int, n: int, layers: int = 4, ansatz: str = "hea",
                 readout: str | None = None, geometry: str = "grid", spacing: float = 1.0,
                 gate: bool = False, train_pos: bool = True, budget: int = 0, random_mask: bool = False):
        super().__init__()
        self.n, self.layers, self.ansatz = n, layers, ansatz
        self.geometry, self.spacing, self.gated = geometry, spacing, gate
        # per-input BUDGET (wave G follow-up): each input drives exactly `budget` of the n atoms, chosen by a
        # straight-through top-k over the gate logits (allocation, not pricing); `random_mask` = the control
        # that drives a fixed random subset of the same size for every input (is *which* atoms what matters?)
        self.budget, self.random_mask = budget, random_mask
        if budget:
            assert 0 < budget <= n, (budget, n)
            gate = True
            if random_mask:
                self.register_buffer("fixed_mask", (torch.randperm(n) < budget).float())
        self.readout = Readout(n, d_in, readout_mode(readout))
        self.base = nn.Parameter(0.02 * torch.randn(layers, n, 2))
        self.cond = nn.Linear(d_in, layers * n * 2)
        nn.init.zeros_(self.cond.weight)
        nn.init.zeros_(self.cond.bias)
        bits = _bits(n)
        self.register_buffer("zdiag", 1.0 - 2.0 * bits)          # (2^n, n)
        czm = torch.ones(2**n)
        for i in range(n - 1):
            czm = czm * torch.where((bits[:, i] * bits[:, i + 1]).bool(), -1.0, 1.0)
        self.register_buffer("cz_mask", czm.to(torch.complex64))
        self.register_buffer("bitsf", bits)
        if ansatz in ("rydberg", "rydberg_mf"):
            # atom positions (units of blockade radius): the interaction graph V_ij = |ri-rj|^-6 is
            # set by the topology family and the lattice spacing (wave G, Sep 21: graph as a factor)
            g = topology_positions(n, geometry, spacing)
            self.pos = nn.Parameter(g + 0.05 * torch.randn(n, 2), requires_grad=train_pos)
            self.omega = nn.Parameter(torch.full((layers,), 0.5))   # global drive
            self.tau = nn.Parameter(torch.full((layers,), 0.3))     # evolution time
        if gate or budget:
            self.gated = True
            # on-the-go participation: per-input, per-qubit drive gate g_i(h) in (0, 1). An undriven
            # atom stays in |0> (n_i = 0), so it also drops out of the blockade graph -> the input
            # decides how many of the scarce qubits it uses. Init: gate = 1 everywhere (identical
            # to the ungated block); a bias of +4 keeps the sigmoid near 1 until trained.
            self.gate_lin = nn.Linear(d_in, n)
            nn.init.zeros_(self.gate_lin.weight)
            nn.init.constant_(self.gate_lin.bias, 0.0 if budget else 4.0)
            if budget:
                nn.init.normal_(self.gate_lin.weight, std=0.1)   # break the top-k tie at init

    def _pair_couplings(self) -> torch.Tensor:
        """(n, n) blockade couplings C6/r^6 from trainable positions, zero
        diagonal; distances clamped at 0.5 R_b bound the divergent r^-6 term."""
        d = torch.cdist(self.pos, self.pos) + torch.eye(
            self.n, device=self.pos.device)
        v = d.clamp(min=0.5).pow(-6)
        return v - torch.diag(torch.diag(v))

    def _blockade_energy(self) -> torch.Tensor:
        """(2^n,) diagonal interaction energy from trainable positions."""
        v = self._pair_couplings()
        return 0.5 * ((self.bitsf @ v) * self.bitsf).sum(-1)        # (2^n,)

    def _forward_mean_field(self, ang: torch.Tensor) -> torch.Tensor:
        """Gutzwiller control: product-state amplitudes (B, n, 2) with the
        blockade phase replaced by per-qubit mean-field phases. Diagonal phases
        leave occupations unchanged, so the explicit per-layer step is the
        exact mean-field dynamics for this Trotter structure."""
        b, n = ang.shape[0], self.n
        v = self._pair_couplings()                                   # (n, n)
        amp = torch.zeros(b, n, 2, dtype=torch.complex64, device=ang.device)
        amp[..., 0] = 1.0  # |0> on every qubit; in-place on a fresh tensor
        for l in range(self.layers):
            gx = rx_gate(self.omega[l].expand(1))[0]                 # (2, 2)
            d = rz_phases(ang[:, l, :, 0]).unsqueeze(-1)             # (B, n, 2, 1)
            amp = torch.einsum("bnij,bnj->bni", d * gx, amp)         # fused RZ∘RX
            occ = amp[..., 1].abs().square()                         # (B, n) <n_j>
            theta = self.tau[l] * (occ @ v)                          # tau*sum_j V_ij <n_j>
            phase = torch.exp(-1j * theta.to(torch.complex64))
            amp = torch.stack([amp[..., 0], amp[..., 1] * phase], -1)
        return amp[..., 0].abs().square() - amp[..., 1].abs().square()  # <Z_i>

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        if self.ansatz == "rydberg_mf":
            return self.readout.from_product_state(self._forward_mean_field(self._angles(h)), h)
        return self.readout(self.probs(h), h)

    def _gates(self, h: torch.Tensor) -> torch.Tensor:
        """(B, n) drive gates: sigmoid price gate, or an exact-k allocation (top-k of the logits with a
        straight-through gradient), or the fixed random-k control."""
        if not self.budget:
            return torch.sigmoid(self.gate_lin(h))
        if self.random_mask:
            return self.fixed_mask.unsqueeze(0).expand(h.shape[0], -1)
        logits = self.gate_lin(h)
        soft = torch.softmax(logits, dim=-1)
        idx = logits.topk(self.budget, dim=-1).indices
        hard = torch.zeros_like(logits).scatter_(-1, idx, 1.0)
        return hard + soft - soft.detach()                            # forward: exact k atoms; backward: softmax

    @torch.no_grad()
    def participation(self, h: torch.Tensor) -> torch.Tensor:
        """(B, n) drive gates (1 everywhere for ungated blocks): mean over the batch = qubits in use."""
        if not self.gated:
            return torch.ones(h.shape[0], self.n, device=h.device)
        return self._gates(h)

    def _angles(self, h: torch.Tensor) -> torch.Tensor:
        b, n = h.shape[0], self.n
        return self.base.unsqueeze(0) + np.pi * torch.tanh(
            self.cond(h).reshape(b, self.layers, n, 2))

    def probs(self, h: torch.Tensor, init_state: torch.Tensor | None = None) -> torch.Tensor:
        """Computational-basis distribution (B, 2^n) after the circuit; `init_state` (B, 2^n complex)
        replaces |0...0> (used by the register diagnostics for butterfly / perturbation responses)."""
        if self.ansatz == "rydberg_mf":
            raise NotImplementedError("probs() needs the full state; the mean-field control is a product state")
        b, n = h.shape[0], self.n
        ang = self._angles(h)
        fast = fast_layers() and n >= 2
        if init_state is not None:
            state = init_state.to(torch.complex64)
        elif fast:                                                   # |0...0> without an in-place write (CUDA-graph safe)
            state = (torch.arange(2**n, device=h.device) == 0).to(torch.complex64).expand(b, -1)
        else:
            state = torch.zeros(b, 2**n, dtype=torch.complex64, device=h.device)
            state[:, 0] = 1.0  # |0...0>; in-place on fresh tensor, autograd-safe
        if self.ansatz == "rydberg":
            eng = self._blockade_energy()
        if fast:
            return self._probs_layered(h, ang, state, eng if self.ansatz == "rydberg" else None)
        for l in range(self.layers):
            if self.ansatz == "hea":
                for k in range(n):
                    # fused RZ∘RY: diag(RZ) row-scales RY — one apply per qubit
                    g = ry_gate(ang[:, l, k, 0])
                    d = rz_phases(ang[:, l, k, 1]).unsqueeze(-1)
                    state = apply_1q(state, d * g, k, n)
                state = state * self.cz_mask
            else:  # rydberg-analog
                if self.gated:
                    gates = self._gates(h)                           # (B, n) participation
                for k in range(n):
                    # fused RZ∘RX with data-driven detunings (drive gated per qubit when `gated`)
                    d = rz_phases(ang[:, l, k, 0]).unsqueeze(-1)
                    if self.gated:
                        gx = rx_gate(self.omega[l] * gates[:, k])    # (B, 2, 2)
                        state = apply_1q(state, d * gx, k, n)
                    else:
                        gx = rx_gate(self.omega[l].expand(1))[0]    # shared drive
                        state = apply_1q(state, d * gx.unsqueeze(0), k, n)
                phase = torch.exp(-1j * self.tau[l].to(torch.complex64) *
                                  eng.to(torch.complex64))
                state = state * phase
        return state.abs().square()                                 # (B, 2^n)

    def _probs_layered(self, h: torch.Tensor, ang: torch.Tensor, state: torch.Tensor, eng: torch.Tensor | None) -> torch.Tensor:
        """probs() with each layer's n single-qubit gates built at once and applied by apply_layer_1q (same circuit)."""
        n = self.n
        gates = self._gates(h) if self.ansatz != "hea" and self.gated else None
        for l in range(self.layers):
            if self.ansatz == "hea":
                g = rz_phases(ang[:, l, :, 1]).unsqueeze(-1) * ry_gate(ang[:, l, :, 0])
                state = apply_layer_1q(state, g, n) * self.cz_mask
            else:
                drive = self.omega[l] * gates if gates is not None else self.omega[l].expand(1, n)    # (B | 1, n)
                g = rz_phases(ang[:, l, :, 0]).unsqueeze(-1) * rx_gate(drive)                        # fused RZ∘RX, (B, n, 2, 2)
                phase = torch.exp(-1j * self.tau[l].to(torch.complex64) * eng.to(torch.complex64))
                state = apply_layer_1q(state, g, n) * phase
        return state.abs().square()


class ConditionedGeometryBlock(nn.Module):
    """Rydberg block whose atom GEOMETRY is input-conditioned (qubits-on-the-go).

    positions(x) = base (learnable) + max_shift * tanh(hypernet(h)) — bounded so
    every emitted register stays inside the hardware envelope (min spacing via
    distance clamp; 2D or 3D per `dims`). Blockade energies are computed
    per-sample and batched. Detunings remain input-conditioned as before.
    """

    MIN_DIST = 0.5   # in blockade-radius units (envelope; exact numbers from AG-0)
    MAX_SHIFT = 0.6

    def __init__(self, d_in: int, n: int, layers: int = 4, dims: int = 2,
                 readout: str | None = None):
        super().__init__()
        self.n, self.layers, self.dims = n, layers, dims
        self.readout = Readout(n, d_in, readout_mode(readout))
        self.base_ang = nn.Parameter(0.02 * torch.randn(layers, n))
        self.cond = nn.Linear(d_in, layers * n)          # detunings
        nn.init.zeros_(self.cond.weight); nn.init.zeros_(self.cond.bias)
        side = int(np.ceil(n ** (1.0 / dims)))
        grid = torch.stack(torch.meshgrid(
            *([torch.arange(side).float()] * dims), indexing="ij"), -1
        ).reshape(-1, dims)[:n]
        self.pos0 = nn.Parameter(grid + 0.05 * torch.randn(n, dims))
        self.geo = nn.Sequential(nn.Linear(d_in, 32), nn.Tanh(),
                                 nn.Linear(32, n * dims))
        nn.init.zeros_(self.geo[-1].weight); nn.init.zeros_(self.geo[-1].bias)
        self.omega = nn.Parameter(torch.full((layers,), 0.5))
        self.tau = nn.Parameter(torch.full((layers,), 0.3))
        bits = _bits(n)
        self.register_buffer("zdiag", 1.0 - 2.0 * bits)
        iu = torch.triu_indices(n, n, offset=1)
        self.register_buffer("pairbits", (bits[:, iu[0]] * bits[:, iu[1]]))  # (2^n, P)
        self.register_buffer("iu", iu)

    def _energies(self, h: torch.Tensor) -> torch.Tensor:
        """(B, 2^n) per-sample diagonal blockade energy from conditioned geometry."""
        b = h.shape[0]
        shift = self.MAX_SHIFT * torch.tanh(self.geo(h)).reshape(b, self.n, self.dims)
        pos = self.pos0.unsqueeze(0) + shift                        # (B, n, dims)
        d = torch.cdist(pos, pos).clamp(min=self.MIN_DIST)
        v = d.pow(-6)[:, self.iu[0], self.iu[1]]                    # (B, P)
        return v @ self.pairbits.T                                  # (B, 2^n)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        b, n = h.shape[0], self.n
        det = self.base_ang.unsqueeze(0) + np.pi * torch.tanh(
            self.cond(h).reshape(b, self.layers, n))
        state = torch.zeros(b, 2**n, dtype=torch.complex64, device=h.device)
        state[:, 0] = 1.0
        eng = self._energies(h).to(torch.complex64)                 # (B, 2^n)
        for l in range(self.layers):
            gx = rx_gate(self.omega[l].expand(1))[0]
            for k in range(n):
                dphase = rz_phases(det[:, l, k]).unsqueeze(-1)
                state = apply_1q(state, dphase * gx.unsqueeze(0), k, n)
            state = state * torch.exp(-1j * self.tau[l].to(torch.complex64) * eng)
        return self.readout(state.abs().square(), h)                # (B, n)


class TetheredGeometry3D(nn.Module):
    """R Rydberg registers of n atoms with per-input 3-D positions (x, y, z), per-input detunings and per-input
    tethers between consecutive registers (doc 35 §23).

    Each register is a ConditionedGeometryBlock(dims=3): positions = base + MAX_SHIFT * tanh(hypernet(h)) on a 3-D
    lattice with random initial offsets, so every coordinate, z included, receives gradient (the zero-offset planar
    start of the routed co-processor left z untrained, doc 35 §22). Register r >= 1 reads the readout of register
    r - 1 scaled by a per-input tether strength t_r(h) = sigmoid(w_r . h + b_r) (b_r = -2 at init, t ~ 0.12);
    the columns that read the upstream readout start at zero, so training starts from independent registers.
    """

    def __init__(self, d_in: int, n: int, registers: int, layers: int = 4):
        super().__init__()
        self.n, self.registers = n, registers
        self.regs = nn.ModuleList([ConditionedGeometryBlock(d_in, n, layers, dims=3)] +
                                  [ConditionedGeometryBlock(d_in + n, n, layers, dims=3) for _ in range(registers - 1)])
        for r in self.regs[1:]:
            with torch.no_grad():
                r.cond.weight[:, d_in:].zero_(); r.geo[0].weight[:, d_in:].zero_()
        self.tether = nn.Linear(d_in, max(registers - 1, 1))
        nn.init.zeros_(self.tether.weight); nn.init.constant_(self.tether.bias, -2.0)
        self.last_tethers = None

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        t = torch.sigmoid(self.tether(h))                               # (B, R-1) per-input tether strengths
        self.last_tethers = t.detach()
        outs = [self.regs[0](h)]
        for k, r in enumerate(self.regs[1:]):
            outs.append(r(torch.cat([h, t[:, k:k + 1] * outs[-1]], dim=-1)))
        return torch.cat(outs, dim=-1)                                  # (B, R*n)

    @torch.no_grad()
    def configuration(self, h: torch.Tensor) -> dict:
        """Per-input atom positions (B, R, n, 3) and tether strengths (B, R-1) for figures."""
        t = torch.sigmoid(self.tether(h)); pos, outs = [], [self.regs[0](h)]
        for k, r in enumerate(self.regs):
            hin = h if k == 0 else torch.cat([h, t[:, k - 1:k] * outs[-1]], dim=-1)
            if k > 0:
                outs.append(r(hin))
            shift = r.MAX_SHIFT * torch.tanh(r.geo(hin)).reshape(h.shape[0], r.n, r.dims)
            pos.append(r.pos0.unsqueeze(0) + shift)
        return {"pos": torch.stack(pos, 1), "tethers": t}


class KMixtureRegister(nn.Module):
    """K precompiled registers + learned router (hardware-trivial deployment:
    soft mixture in training, argmax register at inference)."""

    def __init__(self, d_in: int, n: int, registers: int, k_bank: int = 4,
                 layers: int = 4):
        super().__init__()
        self.banks = nn.ModuleList([
            MultiRegister(d_in, n, registers, layers, "rydberg")
            for _ in range(k_bank)])
        self.router = nn.Linear(d_in, k_bank)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # Hard Gumbel routing (straight-through): ONE bank per sample — 4x
        # cheaper than the soft mixture and preserves per-register dynamics
        # (wave-1 showed soft averaging washes them out). Gradients flow to the
        # router via the ST estimator and to the selected bank only.
        w = torch.nn.functional.gumbel_softmax(self.router(h), tau=1.0, hard=True)
        idx = w.argmax(dim=-1)                                      # (B,)
        out = torch.zeros(h.shape[0], self.banks[0].regs[0].n * len(self.banks[0].regs),
                          device=h.device)
        for k, bank in enumerate(self.banks):
            m = idx == k
            if m.any():
                out[m] = bank(h[m])
        # straight-through: attach router gradient path
        sel_w = (w * torch.nn.functional.one_hot(idx, len(self.banks)).float()).sum(-1)
        return out * sel_w.unsqueeze(-1)


class MultiRegister(nn.Module):
    """R registers of n qubits each = R*n simulated qubits total.

    Hardware-honest scaling: one neutral-atom register with R far-separated
    patches (Rydberg blockade decouples them). Cost is R * cost(n) instead of
    cost(R*n) — trainable at 30+ total qubits.

    Inter-register link (`link`, wave G):
      independent — every patch conditions on the same pooled latent (reference);
      cascade     — patch k also receives patch k-1's readout (a classical feed-forward
                    link between patches: mid-circuit measurement + feed-forward on hardware;
                    zero-init link weights -> identical to `independent` at init);
      shared      — one detuning/drive controller for all patches (patches differ only by
                    their geometry: tests whether graph diversity alone carries the effect).
    `geometry`/`spacing` set each patch's interaction graph; `gate` adds per-input participation.
    """

    def __init__(self, d_in: int, n: int, registers: int, layers: int = 4,
                 ansatz: str = "hea", geometry: str = "grid", spacing: float = 1.0,
                 link: str = "independent", gate: bool = False, train_pos: bool = True,
                 budget: int = 0, random_mask: bool = False):
        super().__init__()
        self.link, self.n = link, n
        kw = dict(geometry=geometry, spacing=spacing, gate=gate, train_pos=train_pos, budget=budget, random_mask=random_mask)
        if link == "independent":
            self.regs = nn.ModuleList([QuantumBlock(d_in, n, layers, ansatz, **kw) for _ in range(registers)])
        elif link == "cascade":
            self.regs = nn.ModuleList([QuantumBlock(d_in, n, layers, ansatz, **kw)] +
                                      [QuantumBlock(d_in + n, n, layers, ansatz, **kw) for _ in range(registers - 1)])
            # zero-init the part of each downstream controller that reads the upstream readout,
            # so the cascade starts as the independent register
            for r in self.regs[1:]:
                with torch.no_grad():
                    r.cond.weight[:, d_in:].zero_()
                    if gate or budget:
                        r.gate_lin.weight[:, d_in:].zero_()
        elif link == "shared":
            first = QuantumBlock(d_in, n, layers, ansatz, **kw)
            self.regs = nn.ModuleList([first] + [QuantumBlock(d_in, n, layers, ansatz, **kw) for _ in range(registers - 1)])
            for r in self.regs[1:]:                      # tie the controllers (detunings + drive + time)
                r.cond, r.base, r.omega, r.tau = first.cond, first.base, first.omega, first.tau
                if gate or budget:
                    r.gate_lin = first.gate_lin
        else:
            raise ValueError(f"unknown link {link!r} (independent|cascade|shared)")

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        if self.link != "cascade":
            return torch.cat([r(h) for r in self.regs], dim=-1)     # (B, R*n)
        outs = [self.regs[0](h)]
        for r in self.regs[1:]:
            outs.append(r(torch.cat([h, outs[-1]], dim=-1)))
        return torch.cat(outs, dim=-1)

    @torch.no_grad()
    def participation(self, h: torch.Tensor) -> torch.Tensor:
        """(B, R*n) drive gates: how many of the scarce qubits each input actually uses."""
        if self.link != "cascade":
            return torch.cat([r.participation(h) for r in self.regs], dim=-1)
        outs, gates = [self.regs[0](h)], [self.regs[0].participation(h)]
        for r in self.regs[1:]:
            hin = torch.cat([h, outs[-1]], dim=-1)
            gates.append(r.participation(hin)); outs.append(r(hin))
        return torch.cat(gates, dim=-1)
