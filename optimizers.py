"""
Optimizers: Muon, Dion, Dion2

References:
- Muon: Jordan et al., 2024 (https://kellerjordan.github.io/posts/muon/)
- Dion: Ahn et al., 2025 (arXiv:2504.05295)
- Dion2: Ahn et al., 2025 (arXiv:2512.16928)

Single-GPU (unsharded) implementations suitable for benchmarking and
small/medium-scale experiments. Sharded variants (1D/2D) are out of scope here.
"""

import math
import torch
from torch.optim.optimizer import Optimizer


# ---------------------------------------------------------------------------
# Newton-Schulz iteration for Muon
# ---------------------------------------------------------------------------

@torch.no_grad()
def newton_schulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Quintic Newton-Schulz iteration to approximate the orthogonalization
    of the matrix G (i.e. compute U V^T where G = U S V^T is the SVD).

    Coefficients follow Jordan et al. (2024). Internal compute is bf16 for
    GPU speed; final result is cast back to G's dtype.
    """
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.to(torch.bfloat16) if G.is_cuda else G.float()
    # Normalize so the spectral norm <= 1 (using Frobenius as upper bound)
    X = X / (X.norm() + eps)
    # If wide matrix, transpose to make tall (cheaper Newton-Schulz)
    transposed = False
    if X.size(0) < X.size(1):
        X = X.T
        transposed = True
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X.to(G.dtype)


# ---------------------------------------------------------------------------
# Muon optimizer
# ---------------------------------------------------------------------------

class Muon(Optimizer):
    """Muon optimizer for 2D (matrix-valued) parameters.

    Update rule:
        M_t = mu * M_{t-1} + g_t            (heavy-ball momentum, Nesterov variant)
        O_t = NewtonSchulz(M_t)             (or with Nesterov-style buffer)
        W_{t+1} = W_t - lr * sqrt(fan_out / fan_in) * O_t

    Non-matrix params should be optimized with AdamW (passed separately).
    """

    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True,
                 ns_steps=5, weight_decay=0.0):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov,
                        ns_steps=ns_steps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            mu = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.ndim != 2:
                    raise ValueError("Muon only supports 2D parameters; "
                                     "use AdamW for biases/embeddings/norms.")
                g = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p)
                buf = state["momentum_buffer"]
                buf.mul_(mu).add_(g)
                update = g.add(buf, alpha=mu) if nesterov else buf

                O = newton_schulz5(update, steps=ns_steps)

                fan_out, fan_in = p.shape
                scale = math.sqrt(max(fan_out, 1) / max(fan_in, 1))

                if wd != 0:
                    p.mul_(1 - lr * wd)
                p.add_(O, alpha=-lr * scale)

        return loss


# ---------------------------------------------------------------------------
# Dion optimizer (unsharded)
#
# Algorithm 2 from Ahn et al. (2025, "Dion: Distributed Orthonormalized Updates")
# Single-step amortized power iteration on the momentum matrix, with error
# feedback into the momentum buffer.
# ---------------------------------------------------------------------------

@torch.no_grad()
def _orthonormalize_qr(P: torch.Tensor) -> torch.Tensor:
    """Return an orthonormal basis for the columns of P via Householder QR.

    Most stable, slowest. Default fallback when faster methods fail.
    """
    Q, _ = torch.linalg.qr(P, mode="reduced")
    return Q


@torch.no_grad()
def _orthonormalize_cholesky(P: torch.Tensor, eps: float = 0.0) -> torch.Tensor:
    """Cholesky-QR.

    For tall matrix P of shape (m, r) with m >= r, an orthonormal basis Q
    for the columns of P satisfies P = Q R with R upper-triangular and
    R^T R = P^T P =: G. With G = L L^T (Cholesky), R = L^T, so
        Q = P R^{-1} = P L^{-T}
        Q^T = L^{-1} P^T

    `eps` (default 0) optionally adds a tiny diagonal ridge to G; only
    useful for borderline cases. When P is well-conditioned this gives
    machine-precision orthonormality. Caller falls back to QR on failure.
    """
    G = P.T @ P                                           # (r, r)
    if eps > 0.0:
        G = G + eps * torch.eye(G.size(0), device=G.device, dtype=G.dtype)
    L = torch.linalg.cholesky(G)                          # G = L L^T, L lower
    # Q^T = L^{-1} @ P^T  (solve L X = P^T  for X = Q^T)
    Q_T = torch.linalg.solve_triangular(L, P.T, upper=False)
    return Q_T.T


@torch.no_grad()
def _orthonormalize_rcqr(P: torch.Tensor, oversample: float = 1.25,
                         eps: float = 0.0) -> torch.Tensor:
    """Randomized Cholesky-QR (RCQR), unsharded version of Algorithm 5.

    Two-stage scheme:
      (i)  Sketch P with S of shape (~1.25 r, m), do thin QR on (S P) to get
           a tiny triangular R1; pre-whiten B = P R1^{-1}.
      (ii) Cholesky-QR on the well-conditioned B.

    The pre-whitening drastically reduces the condition number of B,
    making the subsequent Cholesky-QR much more stable than plain CQR.
    """
    m, r = P.shape
    rt = max(r + 1, int(math.ceil(oversample * r)))
    # Sketch S: (rt, m). The 1/sqrt(rt) scaling makes E[S^T S] = I.
    S = torch.randn(rt, m, device=P.device, dtype=P.dtype) / math.sqrt(rt)
    P_tilde = S @ P                                       # (rt, r)
    _, R1 = torch.linalg.qr(P_tilde, mode="reduced")      # R1: (r, r), upper
    # Pre-whiten: B = P R1^{-1}; B^T = R1^{-T} P^T = solve(R1^T, P^T)  with lower
    B_T = torch.linalg.solve_triangular(R1.T, P.T, upper=False)
    B = B_T.T                                             # (m, r)
    # Cholesky-QR on B
    G = B.T @ B
    if eps > 0.0:
        G = G + eps * torch.eye(G.size(0), device=G.device, dtype=G.dtype)
    L = torch.linalg.cholesky(G)
    Q_T = torch.linalg.solve_triangular(L, B.T, upper=False)
    return Q_T.T


@torch.no_grad()
def _safe_orthonormalize(P: torch.Tensor, method: str) -> torch.Tensor:
    """Try the requested method; fall back to standard QR on failure.

    `method` is one of: "qr", "cholesky", "rcqr".
    Cholesky / RCQR can blow up when P is rank-deficient or extremely
    ill-conditioned (cond(P) >> 5e3 in practice; see Dion paper Appendix A).
    On torch.linalg.LinAlgError or non-finite output we silently retry with
    full Householder QR.
    """
    try:
        if method == "cholesky":
            Q = _orthonormalize_cholesky(P)
        elif method == "rcqr":
            Q = _orthonormalize_rcqr(P)
        else:
            return _orthonormalize_qr(P)
        if not torch.isfinite(Q).all():
            raise RuntimeError("non-finite output from fast orthonormalization")
        return Q
    except (torch.linalg.LinAlgError, RuntimeError):
        return _orthonormalize_qr(P)


class Dion(Optimizer):
    """Dion optimizer for 2D parameters (unsharded version, Algorithm 2).

    Hyperparameters:
        lr: learning rate (multiplied by sqrt(fan_out/fan_in) per matrix)
        rank_fraction: r/min(m,n)  (e.g. 0.25 = quarter-rank)
        beta: error-feedback coefficient (paper uses 0.05)
        weight_decay: decoupled weight decay
    """

    def __init__(self, params, lr=0.01, rank_fraction=0.25, beta=0.05,
                 weight_decay=0.0, eps=1e-8,
                 qr_method="qr", qr_warmup_steps=200):
        """
        qr_method: orthonormalization method for power iteration.
            "qr"        - Householder QR (default, most stable, slowest)
            "cholesky"  - Cholesky-QR (Appendix A; ~3-10x faster than QR
                          but unstable when cond(P) > 5e3)
            "rcqr"      - randomized Cholesky-QR (Algorithm 5; safer than
                          plain Cholesky-QR thanks to a sketch-based
                          pre-whitening step)
        qr_warmup_steps: use plain QR for the first this many steps before
            switching to the requested method. Recommended for "cholesky"
            because cond(P) is large early in training (see paper Fig. 5).
            Set to 0 to use the fast method from step 1.
        """
        if qr_method not in ("qr", "cholesky", "rcqr"):
            raise ValueError("qr_method must be 'qr', 'cholesky', or 'rcqr'")
        defaults = dict(lr=lr, rank_fraction=rank_fraction, beta=beta,
                        weight_decay=weight_decay, eps=eps,
                        qr_method=qr_method,
                        qr_warmup_steps=qr_warmup_steps)
        super().__init__(params, defaults)
        # global step count (shared across all parameters in this optimizer)
        self._global_step = 0

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._global_step += 1

        for group in self.param_groups:
            lr = group["lr"]
            rf = group["rank_fraction"]
            beta = group["beta"]
            wd = group["weight_decay"]
            eps = group["eps"]
            qr_method = group["qr_method"]
            warmup = group["qr_warmup_steps"]
            # use plain QR during warmup, requested method afterwards
            method = "qr" if self._global_step <= warmup else qr_method

            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.ndim != 2:
                    raise ValueError("Dion only supports 2D parameters.")

                g = p.grad
                m, n = p.shape
                r = max(1, int(round(rf * min(m, n))))

                state = self.state[p]
                if "M" not in state:
                    state["M"] = torch.zeros_like(p)  # momentum
                    # right factor V: (n, r), initialised with random orthonormal columns
                    V0 = torch.randn(n, r, device=p.device, dtype=p.dtype)
                    V0, _ = torch.linalg.qr(V0, mode="reduced")
                    state["V"] = V0

                M = state["M"]
                V = state["V"]

                # 1. accumulate gradient into momentum
                M.add_(g)

                # 2. PowerIter1: P = M @ V, U = orthonormalize(P), W = M^T @ U
                P = M @ V                          # (m, r)
                U = _safe_orthonormalize(P, method)  # (m, r), orthonormal cols
                W = M.T @ U                        # (n, r)

                # 3. error feedback: M <- M - beta * U W^T
                M.sub_(U @ W.T, alpha=beta)

                # 4. column normalization of W -> V_new
                col_norm = W.norm(dim=0, keepdim=True).clamp_min(eps)
                V_new = W / col_norm
                state["V"] = V_new

                # 5. orthonormal update O = U V_new^T, scaled by sqrt(m/n)
                O = U @ V_new.T
                scale = math.sqrt(max(m, 1) / max(n, 1))

                if wd != 0:
                    p.mul_(1 - lr * wd)
                p.add_(O, alpha=-lr * scale)

        return loss


# ---------------------------------------------------------------------------
# Dion2 optimizer
#
# Algorithm 1 from Ahn, Amsel, Langford (2025, arXiv:2512.16928).
# Selects an alpha-fraction of rows (or columns) of the momentum matrix,
# orthonormalises only that submatrix via Newton-Schulz, and decays only
# the selected rows/columns of the momentum (selective decay = error feedback).
# ---------------------------------------------------------------------------

class Dion2(Optimizer):
    """Dion2 optimizer for 2D parameters.

    Hyperparameters:
        lr: learning rate (multiplied by sqrt(fan_out/fan_in))
        alpha: fraction of rows/columns selected per step (e.g. 0.25)
        momentum_decay: mu in the paper (default 0.95)
        selection: "l1"  -> top-alpha rows/cols by L1 norm of momentum
                   "random" -> uniformly random
        ns_steps: Newton-Schulz iterations
        weight_decay: decoupled weight decay
    """

    def __init__(self, params, lr=0.02, alpha=0.25, momentum_decay=0.95,
                 selection="l1", ns_steps=5, weight_decay=0.0):
        if selection not in ("l1", "random"):
            raise ValueError("selection must be 'l1' or 'random'")
        defaults = dict(lr=lr, alpha=alpha, momentum_decay=momentum_decay,
                        selection=selection, ns_steps=ns_steps,
                        weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def _select(self, M: torch.Tensor, k: int, axis: int, mode: str):
        """Return indices of k rows (axis=0) or columns (axis=1) of M."""
        if mode == "random":
            n_total = M.shape[axis]
            return torch.randperm(n_total, device=M.device)[:k]
        # L1-norm based selection
        norms = M.abs().sum(dim=1 - axis)  # sum over the *other* axis
        return torch.topk(norms, k=k, largest=True).indices

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            alpha = group["alpha"]
            mu = group["momentum_decay"]
            sel_mode = group["selection"]
            ns_steps = group["ns_steps"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.ndim != 2:
                    raise ValueError("Dion2 only supports 2D parameters.")

                g = p.grad
                m, n = p.shape

                state = self.state[p]
                if "M" not in state:
                    state["M"] = torch.zeros_like(p)
                M = state["M"]

                # 1. accumulate gradient into momentum
                M.add_(g)

                # 2. select alpha-fraction along the SHORTER dimension
                if m <= n:
                    axis = 0  # rows
                    k = max(1, int(round(alpha * m)))
                    idx = self._select(M, k, axis=0, mode=sel_mode)
                    sub = M.index_select(0, idx)             # (k, n)
                    O_sub = newton_schulz5(sub, steps=ns_steps)
                    # 3. selective decay: only on selected rows
                    M.index_copy_(0, idx, M.index_select(0, idx).mul(mu))
                    # 4. sparse update: only selected rows
                    fan_out, fan_in = p.shape
                    scale = math.sqrt(max(fan_out, 1) / max(fan_in, 1))
                    if wd != 0:
                        p.mul_(1 - lr * wd)
                    p.index_add_(0, idx, O_sub, alpha=-lr * scale)
                else:
                    axis = 1  # columns
                    k = max(1, int(round(alpha * n)))
                    idx = self._select(M, k, axis=1, mode=sel_mode)
                    sub = M.index_select(1, idx)             # (m, k)
                    O_sub = newton_schulz5(sub, steps=ns_steps)
                    M.index_copy_(1, idx, M.index_select(1, idx).mul(mu))
                    fan_out, fan_in = p.shape
                    scale = math.sqrt(max(fan_out, 1) / max(fan_in, 1))
                    if wd != 0:
                        p.mul_(1 - lr * wd)
                    p.index_add_(1, idx, O_sub, alpha=-lr * scale)

        return loss


# ---------------------------------------------------------------------------
# Helper: split parameters into "matrix" (for Muon/Dion/Dion2) and "scalar"
# (for AdamW). Matches the practice of the papers.
# ---------------------------------------------------------------------------

def split_params(model, exclude_names=("embed", "lm_head", "norm", "bias")):
    """Return (matrix_params, scalar_params).

    Matrix parameters are 2D weight matrices NOT in `exclude_names`
    (typically these are attention QKV/output and MLP weight matrices).
    Everything else (embeddings, LM head, biases, norms) goes to scalar_params
    and should be optimized with AdamW.
    """
    matrix_params, scalar_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 2 and not any(k in name for k in exclude_names):
            matrix_params.append(p)
        else:
            scalar_params.append(p)
    return matrix_params, scalar_params
