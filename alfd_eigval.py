"""
Eigenvalue-density-based ALFD power bound for m_W = 3 (p = 4).

Uses the noncentral-Wishart joint density of (real) ordered eigenvalues of a
noncentral Wishart matrix. The matrix hypergeometric 0F1^(2)((1/2)k; (1/4)Ω, S)
is evaluated via Plamen Koev's mhg algorithm (Koev & Edelman 2006). We call a
standalone C port of Koev's mhg.c directly from Python (ctypes) — no MATLAB or
Octave required.

The calculation bounds tests measurable with respect to the ordered eigenvalue
vector (the invariant class studied in the reference application).  It is a
direct m_W=3 extension of the GKM Supplement D.3.2 implementation of EMW:
one pooled ordinary-importance-sampling null bank, the fixed EMW weight update,
the Step-6 mixture calibration, and the Step-8/9 grid adjustment.  Numerical
matrix-hypergeometric truncation remains adaptive for every density pair.

Setup (one-time):
    # Koev's mhg15 package lives in ~/Oxford/subvector_AR_HW/koev/mhg15/
    # Build the shared library once (needs only clang / Xcode CLT):
    sh ~/Oxford/subvector_AR_HW/koev/mhg15/build.sh
    # -> produces libmhg.dylib and its mhg_core.c SHA-256 stamp next to mhg.py.

Run:
    python3 alfd_eigval.py --version 1009590 --profile production --preflight-only
    python3 alfd_eigval.py --version 1009590 --profile production \
        --acknowledge-expensive
"""

import os
import sys
import time
import json
import math
import hashlib
import platform
from collections import Counter
from dataclasses import asdict, dataclass

import numpy as np
import scipy
from scipy.linalg import inv, sqrtm
from scipy.special import logsumexp


# ============================================================
# Allowed kappa configurations (m_W = 3)
# Each provides a *starting* 0F1 order.  Every density evaluation now checks
# its coefficient tail and increases the order locally when needed, so this is
# no longer a numerical result that users have to tune by rerunning a curve.
# ============================================================

RESULT_SCHEMA_VERSION = 4
ALGORITHM_VERSION = "gkm_eigval_mw3_adaptive_v4"
CALIBRATION_METHOD = "gkm_step6_reused_pooled_bank"
BOUND_KIND = "gkm_d3_2_grid_adjusted_mc_power_bound"
COMMON_GRID_METHOD = "strength_shape_3d_v1"
POOLED_IS_METHOD = "gkm_stratified_equal_null_mixture_v1"
GRID_DESIGN_BETA_COUNT = 81

ALLOWED_CONFIGS = {
    (35, 25, 15): dict(
        M_start=20,
        standard=[(35.0, 25.0, 15.0), (50.0, 25.0, 5.0),
                  (15.0, 10.0, 5.0), (5.0, 3.0, 1.0)],
    ),
    (100, 30, 15): dict(
        M_start=20,
        standard=[(100.0, 30.0, 15.0), (100.0, 30.0, 5.0),
                  (60.0, 30.0, 10.0), (30.0, 20.0, 5.0)],
    ),
    (100, 95, 90): dict(
        M_start=20,
        standard=[(100.0, 95.0, 90.0), (100.0, 95.0, 30.0),
                  (100.0, 60.0, 40.0), (60.0, 55.0, 45.0)],
    ),
}

# Version labels selectable on the command line (python alfd_eigval.py --version <label>).
# Each maps to a key in ALLOWED_CONFIGS.
VERSION_LABELS = {
    "352515": (35, 25, 15),
    "1003015": (100, 30, 15),
    "1009590": (100, 95, 90),
}


# ============================================================
# Self-contained sampling / NCP utilities (HW Appendix A.3 DGP)
# ============================================================

def build_M(kappas, k_eff):
    """k_eff x p matrix M with M'M = diag(kappas). Requires k_eff >= p."""
    p = len(kappas)
    M = np.zeros((k_eff, p))
    for j in range(p):
        if kappas[j] > 0:
            M[j, j] = np.sqrt(kappas[j])
    return M


def simulate_Xi(M, n_sim, rng):
    """n_sim independent Xi ~ Matrix-Normal(M, I_{kp})."""
    k, p = M.shape
    return rng.standard_normal((n_sim, k, p)) + M[None, :, :]


def eigenvalues_descending(Xi_batch):
    """Sorted (descending) eigenvalues of Xi'Xi for each sample, shape (S, p)."""
    XiTXi = np.einsum('ski,skj->sij', Xi_batch, Xi_batch)
    return np.linalg.eigvalsh(XiTXi)[:, ::-1]


def log_sum_exp(x, axis=0):
    """Numerically stable log-sum-exp along `axis`."""
    x_max = np.max(x, axis=axis, keepdims=True)
    return np.squeeze(x_max, axis=axis) + np.log(np.sum(np.exp(x - x_max), axis=axis))


def _dgp_constants(k, n):
    """Fixed pieces of the HW Appendix A.3 DGP (m_W = 3)."""
    Sigma = np.array([
        [1.0, 0.1, 0.3, 0.2, 0.8],
        [0.1, 1.0, 0.3, 0.2, 0.1],
        [0.3, 0.3, 1.0, 0.3, 0.2],
        [0.2, 0.2, 0.3, 1.0, 0.3],
        [0.8, 0.1, 0.2, 0.3, 1.0],
    ])
    A = np.array([
        [1/np.sqrt(n*3), 0, 0],
        [1/np.sqrt(n*3), 0, 0],
        [1/np.sqrt(n*3), 0, 0],
        [0, 1/np.sqrt(n*2), 0],
        [0, 1/np.sqrt(n*2), 0],
        [0, 0, 1/np.sqrt(n*2)],
        [0, 0, 1/np.sqrt(n*2)],
    ])
    pi_x = (4.0 / np.sqrt(k * n)) * np.array([1, 1, 1, -1, 1, 1, 1])
    gamma_params = np.array([-1.0, 1.0, 1.0])
    return Sigma, A, pi_x, gamma_params


def asymptotic_ncp_eigenvalues(beta, kappas, k, n):
    """
    Asymptotic NCP eigenvalues (descending) for the limit experiment at
    given beta, for the HW Appendix A.3 DGP.

    Pi* = [beta*pi_x + Pi_W*gamma, Pi_W] (k x p, p = m_W+1); Sigma_uu is the
    p x p noise covariance of (y_0_residual, V_W). The NCP eigenvalues are
    those of Sigma_uu^{-1/2'} (n Pi*' Pi*) Sigma_uu^{-1/2}.
    """
    m_W = len(kappas)
    Sigma, A, pi_x, gamma_params = _dgp_constants(k, n)

    Sigma_eps_eps = Sigma[0, 0]
    Sigma_eps_Vw = Sigma[0, 2:]
    Sigma_Vw_Vw = Sigma[2:, 2:]
    Sigma_Vw_Vw_eps = Sigma_Vw_Vw - np.outer(Sigma_eps_Vw, Sigma_eps_Vw) / Sigma_eps_eps
    sqrt_Sigma_w = sqrtm(Sigma_Vw_Vw_eps)

    K = np.diag(kappas)
    Pi_W = A @ sqrtm(K) @ sqrt_Sigma_w
    pi_y0 = beta * pi_x + Pi_W @ gamma_params
    Pi_star = np.column_stack([pi_y0, Pi_W])

    sel = np.zeros(5)
    sel[0] = 1.0
    sel[1] = beta
    sel[2:] = gamma_params
    Var_u_y0 = float(sel @ Sigma @ sel)
    cov_uy0_Vw = sel @ Sigma[:, 2:]

    Sigma_uu = np.zeros((m_W + 1, m_W + 1))
    Sigma_uu[0, 0] = Var_u_y0
    Sigma_uu[0, 1:] = cov_uy0_Vw
    Sigma_uu[1:, 0] = cov_uy0_Vw
    Sigma_uu[1:, 1:] = Sigma_Vw_Vw

    sqrt_Sigma_uu_inv = inv(np.real(sqrtm(Sigma_uu)))
    NCP_sym = sqrt_Sigma_uu_inv.T @ (n * Pi_star.T @ Pi_star) @ sqrt_Sigma_uu_inv
    eigvals = np.linalg.eigvalsh(NCP_sym)
    return np.sort(np.real(eigvals))[::-1]


# ============================================================
# mhg setup (standalone C core via ctypes — no Octave/MATLAB)
# ============================================================

KOEV_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'koev')
MHG_DIR = os.path.join(KOEV_DIR, 'mhg15')


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_mhg_build_provenance(mhg_dir, lib_name):
    """Fail closed unless the shared library was built from this C source."""
    source = os.path.join(mhg_dir, 'mhg_core.c')
    stamp = os.path.join(mhg_dir, f'{lib_name}.mhg_core.sha256')
    build = os.path.join(mhg_dir, 'build.sh')
    if not os.path.isfile(source):
        raise FileNotFoundError(f"mhg source not found: {source}")
    if not os.path.isfile(stamp):
        raise RuntimeError(
            f"mhg build provenance stamp not found: {stamp}\n"
            f"Rebuild with:  sh {build}")
    try:
        with open(stamp, "r", encoding="ascii") as handle:
            fields = handle.read().split()
    except (OSError, UnicodeError) as exc:
        raise RuntimeError(
            f"Could not read mhg build provenance stamp {stamp}: {exc}\n"
            f"Rebuild with:  sh {build}") from exc
    stamped_hash = fields[0].lower() if len(fields) == 1 else ""
    if (len(stamped_hash) != 64
            or any(ch not in "0123456789abcdef" for ch in stamped_hash)):
        raise RuntimeError(
            f"Invalid mhg build provenance stamp: {stamp}\n"
            f"Rebuild with:  sh {build}")
    source_hash = _sha256_file(source)
    if stamped_hash != source_hash:
        raise RuntimeError(
            "mhg shared library is stale relative to mhg_core.c.\n"
            f"  built source sha256: {stamped_hash}\n"
            f"  current source sha256: {source_hash}\n"
            f"Rebuild with:  sh {build}")
    return source_hash


def setup_mhg(mhg_dir=MHG_DIR):
    """Import the standalone C mhg wrapper (Koev & Edelman 2006).

    Returns the callable ``mhg(arg0, alpha, p, q, x, y=None)`` from
    ``koev/mhg15/mhg.py``, which dispatches to ``libmhg.dylib``.
    """
    lib_name = 'libmhg.dylib' if sys.platform == 'darwin' else 'libmhg.so'
    lib = os.path.join(mhg_dir, lib_name)
    if not os.path.isfile(lib):
        raise FileNotFoundError(
            f"{lib_name} not found in {mhg_dir}.\n"
            f"Build it with:  sh {os.path.join(mhg_dir, 'build.sh')}"
        )
    _verify_mhg_build_provenance(mhg_dir, lib_name)
    if mhg_dir not in sys.path:
        sys.path.insert(0, mhg_dir)
    from mhg import mhg as mhg_eval
    return mhg_eval



_mhg = setup_mhg()


# ============================================================
# mhg wrappers
# ============================================================

# The C routine accumulates the per-degree coefficients internally whether or
# not Python asks to receive them.  Returning them therefore adds negligible
# work and lets every call make an explicit convergence decision.
MHG_CONV_TOL = 1e-10
MHG_DEFAULT_STEP = 20
MHG_DEFAULT_MAX = 300
MHG_RATIO_WINDOW = 5
MHG_TRACE_MARGIN = 40
MHG_LARGE_TRACE_THRESHOLD = 120.0
MHG_LARGE_TRACE_MARGIN = 60.0
MHG_BENCHMARK_SEED = 0x454D57
MHG_DEFAULT_BENCHMARK_SAMPLES = {
    (35, 25, 15): 64,
    (100, 30, 15): 16,
    (100, 95, 90): 2,
}
MHG_MAX_BENCHMARK_SAMPLES = 256


class MHGConvergenceError(RuntimeError):
    """Raised rather than silently accepting an under-truncated 0F1 series."""


@dataclass(frozen=True)
class MHGSeriesAssessment:
    converged: bool
    last_block_ratio: float
    estimated_remainder_ratio: float
    block_ratio: float
    peak_degree: int
    last_positive_degree: int
    numerical_collapse: bool


def assess_mhg_series(value, coefficients, tol=MHG_CONV_TOL,
                      ratio_window=MHG_RATIO_WINDOW):
    """Assess a positive 0F1 series from its per-total-degree coefficients.

    The old check inspected only ``coef[-1]``.  That can be tiny even while a
    sizeable omitted tail remains.  Here we require a run of positive,
    contracting per-degree coefficients after the peak and extrapolate them
    geometrically.  An abrupt ratio collapse while the preceding coefficient
    still carries material mass is treated as arithmetic underflow, not
    convergence.  A failed assessment causes the caller to increase the order
    for this *one density value*.

    The estimate is a numerical stopping criterion, not a symbolic remainder
    theorem for matrix 0F1.  The run records the tolerance and maximum selected
    order, and fails at ``M_max`` so accuracy cannot pass silently.
    """
    if not np.isfinite(tol) or not (1e-13 <= float(tol) < 1.0):
        raise ValueError("tol must be finite and lie in [1e-13, 1)")
    if (not isinstance(ratio_window, (int, np.integer))
            or isinstance(ratio_window, (bool, np.bool_))
            or int(ratio_window) < 2):
        raise ValueError("ratio_window must be an integer >= 2")

    coef = np.asarray(coefficients, dtype=float)
    if coef.ndim != 1 or coef.size < 3 or not np.isfinite(value):
        return MHGSeriesAssessment(False, np.inf, np.inf, np.inf, -1, -1, True)
    if not np.all(np.isfinite(coef)) or value <= 0.0:
        return MHGSeriesAssessment(False, np.inf, np.inf, np.inf, -1, -1, True)

    # Coefficients are nonnegative for the arguments used here.  Permit only
    # round-off-sized negative noise; a material negative coefficient signals
    # a broken evaluation rather than convergence.
    neg_mass = float(-coef[coef < 0.0].sum())
    scale = max(abs(float(value)), 1.0)
    if neg_mass > 1e-13 * scale:
        peak = int(np.argmax(coef))
        return MHGSeriesAssessment(False, np.inf, np.inf, np.inf,
                                   peak, -1, True)
    coef = np.maximum(coef, 0.0)
    total = float(coef.sum())
    if not np.isfinite(total) or total <= 0.0:
        return MHGSeriesAssessment(False, np.inf, np.inf, np.inf, -1, -1, True)

    peak = int(np.argmax(coef))
    positive = np.flatnonzero(coef > 0.0)
    if positive.size == 0:
        return MHGSeriesAssessment(False, np.inf, np.inf, np.inf,
                                   peak, -1, True)
    last_positive = int(positive[-1])
    window = int(ratio_window)
    if last_positive - peak < window or last_positive < window:
        last_ratio = float(coef[last_positive] / total)
        trailing_material_zero = bool(
            last_positive < coef.size - 1 and last_ratio > tol)
        return MHGSeriesAssessment(False, last_ratio, np.inf, np.inf,
                                   peak, last_positive,
                                   trailing_material_zero)

    # Detect the exact signature of representational underflow: a coefficient
    # ratio suddenly collapses even though the previous degree is still
    # material.  Merely increasing M cannot repair this and must not produce a
    # false plateau.
    # Start only after a short post-peak run.  Early ratios can legitimately
    # drop quickly (for scalar 0F1, q_d=z/((c+d-1)d)); the arithmetic-failure
    # fingerprint is a late collapse from a still-large ratio.
    post_start = max(peak + int(ratio_window), 1)
    post_coef = coef[post_start - 1:last_positive + 1]
    post_ratios = post_coef[1:] / post_coef[:-1]
    previous_mass = post_coef[:-1] / total
    collapse = bool(np.any(
        (post_ratios[1:] < 0.5 * post_ratios[:-1])
        & (post_ratios[:-1] > 0.1)
        & (previous_mass[1:] > max(tol, 1e-14)))) if post_ratios.size > 1 else False

    lo = last_positive - window + 1
    ratios = coef[lo:last_positive + 1] / coef[lo - 1:last_positive]
    monotone = bool(np.all(ratios[1:] <= 1.01 * ratios[:-1]))
    contracting = bool(np.all((ratios >= 0.0) & (ratios < 1.0)))
    rho = float(np.max(ratios)) if ratios.size else np.inf
    remainder = (float(coef[last_positive] * rho / (1.0 - rho) / total)
                 if contracting else np.inf)
    last_ratio = float(coef[last_positive] / total)
    # The historical C failure ended a still-material series with exact zeros.
    # A zero tail is harmless only after the observed coefficient and its
    # geometric continuation are already below tolerance.
    if last_positive < coef.size - 1 and (last_ratio > tol or remainder > tol):
        collapse = True
    converged = bool(not collapse and monotone and contracting
                     and remainder <= tol)
    return MHGSeriesAssessment(converged, last_ratio, remainder, rho, peak,
                               last_positive, collapse)


def mhg_two_matrix(c, Omega_eigs, S_eigs, M_trunc=50):
    """Single 0F1^(2)(c; (1/4)Ω, S) call via Koev's mhg (C core)."""
    Om = np.asarray(Omega_eigs, dtype=float) / 4.0
    S = np.asarray(S_eigs, dtype=float)
    return float(_mhg(M_trunc, 2.0, [], [float(c)], Om, y=S))


def _validated_mhg_arguments(c, Omega_eigs, S_eigs):
    """Validate and roundoff-clip the nonnegative 0F1 eigenvalue arguments."""
    if not np.isfinite(c) or float(c) <= 0.0:
        raise ValueError("c must be finite and positive")
    Omega_raw = np.asarray(Omega_eigs, dtype=float)
    sample = np.asarray(S_eigs, dtype=float)
    if (Omega_raw.ndim != 1 or Omega_raw.size == 0 or sample.ndim != 1
            or Omega_raw.shape != sample.shape
            or not np.all(np.isfinite(Omega_raw))
            or not np.all(np.isfinite(sample))):
        raise ValueError("Omega and S must be aligned finite eigenvalue vectors")
    omega_tol = 1e-12 * max(1.0, float(np.max(np.abs(Omega_raw))))
    sample_tol = 1e-12 * max(1.0, float(np.max(np.abs(sample))))
    if np.min(Omega_raw) < -omega_tol or np.min(sample) < -sample_tol:
        raise ValueError("matrix 0F1 arguments must be positive semidefinite")
    return float(c), np.maximum(Omega_raw, 0.0), np.maximum(sample, 0.0)


def _validated_integer(name, value, minimum=1):
    if (not isinstance(value, (int, np.integer))
            or isinstance(value, (bool, np.bool_))):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def mhg_converged(c, Omega_eigs, S_eigs, M_trunc, tol=MHG_CONV_TOL,
                  ratio_window=MHG_RATIO_WINDOW):
    """Return ``(value, converged, estimated_remainder_ratio)`` at one order."""
    c, Omega_raw, S = _validated_mhg_arguments(c, Omega_eigs, S_eigs)
    M_trunc = _validated_integer("M_trunc", M_trunc)
    Om = Omega_raw / 4.0
    s, coef = _mhg(M_trunc, 2.0, [], [c], Om, y=S, want_coef=True)
    assessment = assess_mhg_series(s, coef, tol=tol,
                                   ratio_window=ratio_window)
    return float(s), assessment.converged, assessment.estimated_remainder_ratio


def mhg_two_matrix_adaptive(c, Omega_eigs, S_eigs, M_start=20,
                            M_step=MHG_DEFAULT_STEP, M_max=MHG_DEFAULT_MAX,
                            tol=MHG_CONV_TOL,
                            ratio_window=MHG_RATIO_WINDOW,
                            trace_margin=MHG_TRACE_MARGIN):
    """Evaluate 0F1^(2), increasing total degree only for difficult pairs.

    Returns ``(value, selected_order, assessment, raw_evaluations)``.  This is
    the mechanism that replaces curve-level trial runs at several M values.
    """
    M_start = _validated_integer("M_start", M_start)
    M_step = _validated_integer("M_step", M_step)
    M_max = _validated_integer("M_max", M_max)
    ratio_window = _validated_integer("ratio_window", ratio_window, minimum=2)
    if not (M_start <= M_max):
        raise ValueError("Require 1 <= M_start <= M_max")
    if not np.isfinite(tol) or not (1e-13 <= float(tol) < 1.0):
        raise ValueError("tol must be finite and lie in [1e-13, 1)")
    if not np.isfinite(trace_margin) or float(trace_margin) < 0.0:
        raise ValueError("trace_margin must be finite and nonnegative")
    c, Omega_raw, S = _validated_mhg_arguments(c, Omega_eigs, S_eigs)
    Om = Omega_raw / 4.0
    if np.all(Om == 0.0) or np.all(S == 0.0):
        assessment = MHGSeriesAssessment(
            True, 0.0, 0.0, 0.0, 0, 0, False)
        return 1.0, 0, assessment, 0
    tau = float(np.sqrt(max(0.0, float(np.sum(Om)) * float(np.sum(S)))))
    effective_margin = max(
        float(trace_margin),
        MHG_LARGE_TRACE_MARGIN if tau >= MHG_LARGE_TRACE_THRESHOLD else 0.0)
    # Strong p=4 calls consistently need about tau+60 terms.  Starting there
    # avoids recomputing the entire partition traversal once.  If the heuristic
    # exceeds the user cap, evaluate at M_max before failing; the trace proxy is
    # not itself a convergence theorem.
    order = min(max(M_start, int(np.ceil(tau + effective_margin))), M_max)
    raw_evaluations = 0
    while True:
        value, coef = _mhg(order, 2.0, [], [c], Om, y=S,
                           want_coef=True)
        raw_evaluations += 1
        assessment = assess_mhg_series(value, coef, tol=tol,
                                       ratio_window=ratio_window)
        if assessment.converged:
            return float(value), order, assessment, raw_evaluations
        if assessment.numerical_collapse:
            raise MHGConvergenceError(
                "0F1^(2) coefficient sequence shows an abrupt numerical "
                "collapse before its tail is negligible.\n"
                f"  M={order}, tol={tol:.2e}, tau={tau:.2f}\n"
                f"  Omega={Omega_raw.tolist()}\n"
                f"  S={S.tolist()}\n"
                f"  peak_degree={assessment.peak_degree}, "
                f"last_positive_degree={assessment.last_positive_degree}\n"
                "The C core needs scaled arithmetic; increasing M is not a fix."
            )
        if order >= M_max:
            raise MHGConvergenceError(
                "0F1^(2) did not meet the adaptive truncation criterion.\n"
                f"  M_start={M_start}, M_max={M_max}, tol={tol:.2e}\n"
                f"  Omega={Omega_raw.tolist()}\n"
                f"  S={S.tolist()}\n"
                f"  peak_degree={assessment.peak_degree}, "
                f"last_positive_degree={assessment.last_positive_degree}, "
                f"last_coefficient/total={assessment.last_block_ratio:.3e}, "
                f"estimated_remainder/total="
                f"{assessment.estimated_remainder_ratio:.3e}\n"
                "Increase M_max; this value has not been used."
            )
        order = min(order + M_step, M_max)


def mhg_two_matrix_batch(c, Omegas, S_batch, M_trunc=20,
                         M_step=MHG_DEFAULT_STEP, M_max=MHG_DEFAULT_MAX,
                         mhg_tol=MHG_CONV_TOL,
                         ratio_window=MHG_RATIO_WINDOW,
                         return_diagnostics=False):
    """
    Vectorized: returns (G, N) matrix of 0F1^(2)(c; (1/4)Ω_g, S_n) values.

    Omegas : (G, p) array — NCP eigenvalue tuples (each row)
    S_batch: (N, p) array — observed eigenvalue tuples (each row)

    Each (g, n) pair is a direct C call (no IPC), so a plain Python loop is
    fine — and faster than the old single-Octave-round-trip batching.
    """
    Omegas = np.asarray(Omegas, dtype=float)
    S = np.asarray(S_batch, dtype=float)
    if (Omegas.ndim != 2 or S.ndim != 2 or Omegas.shape[1] != S.shape[1]):
        raise ValueError("Omegas and S_batch must be aligned 2D arrays")
    G, N = Omegas.shape[0], S.shape[0]
    out = np.empty((G, N))
    counts = Counter()
    raw_evaluations = 0
    max_remainder = 0.0
    for g in range(G):
        for n in range(N):
            value, order, assessment, n_raw = mhg_two_matrix_adaptive(
                c, Omegas[g], S[n], M_start=M_trunc, M_step=M_step,
                M_max=M_max, tol=mhg_tol, ratio_window=ratio_window)
            out[g, n] = value
            counts[order] += 1
            raw_evaluations += n_raw
            max_remainder = max(max_remainder,
                                assessment.estimated_remainder_ratio)
    diagnostics = dict(pairs=G * N, raw_evaluations=raw_evaluations,
                       order_counts=dict(counts),
                       max_order=max(counts) if counts else 0,
                       max_remainder_ratio=max_remainder)
    return (out, diagnostics) if return_diagnostics else out


def _mhg_chunk_worker(args):
    """
    Worker function for one chunk of S samples (used by multiprocessing.Pool).
    Module-level so it pickles cleanly. Relies on `_mhg` being set at import
    time in each spawned worker process (setup_mhg() runs in every worker).

    args : (chunk_id, c, Omegas, S_chunk, adaptive-options)
    returns : (chunk_id, ndarray of shape (G, len(S_chunk)), diagnostics)
    """
    (chunk_id, c, Omegas, S_chunk, M_start, M_step, M_max,
     mhg_tol, ratio_window) = args
    G, N = Omegas.shape[0], S_chunk.shape[0]
    out = np.empty((G, N))
    counts = Counter()
    raw_evaluations = 0
    max_remainder = 0.0
    for g in range(G):
        for n in range(N):
            value, order, assessment, n_raw = mhg_two_matrix_adaptive(
                c, Omegas[g], S_chunk[n], M_start=M_start, M_step=M_step,
                M_max=M_max, tol=mhg_tol, ratio_window=ratio_window)
            out[g, n] = value
            counts[order] += 1
            raw_evaluations += n_raw
            max_remainder = max(max_remainder,
                                assessment.estimated_remainder_ratio)
    diagnostics = dict(pairs=G * N, raw_evaluations=raw_evaluations,
                       order_counts=dict(counts),
                       max_order=max(counts) if counts else 0,
                       max_remainder_ratio=max_remainder)
    return chunk_id, out, diagnostics


def _merge_mhg_diagnostics(target, source):
    target["pairs"] += int(source["pairs"])
    target["raw_evaluations"] += int(source["raw_evaluations"])
    target["max_order"] = max(target["max_order"], int(source["max_order"]))
    target["max_remainder_ratio"] = max(
        target["max_remainder_ratio"], float(source["max_remainder_ratio"]))
    target["order_counts"].update(
        {int(k): int(v) for k, v in source["order_counts"].items()})


def chunked_mhg_batch(c, Omegas, S_batch, M_trunc=20,
                      chunk_size=100, progress_label="mhg",
                      progress_interval_sec=15.0,
                      n_workers=1, M_step=MHG_DEFAULT_STEP,
                      M_max=MHG_DEFAULT_MAX, mhg_tol=MHG_CONV_TOL,
                      ratio_window=MHG_RATIO_WINDOW,
                      return_diagnostics=False):
    """
    Run mhg over S_batch in chunks of `chunk_size` samples, printing
    progress and an ETA based on observed throughput.

    n_workers : 1 = serial (original behaviour).  >1 = use multiprocessing.Pool
                with that many worker processes; chunks are dispatched
                independently via imap_unordered.
    """
    Omegas = np.asarray(Omegas, dtype=float)
    S_batch = np.asarray(S_batch, dtype=float)
    G = Omegas.shape[0]
    N = S_batch.shape[0]
    total_calls = G * N
    out = np.empty((G, N))
    diagnostics = dict(pairs=0, raw_evaluations=0, order_counts=Counter(),
                       max_order=0, max_remainder_ratio=0.0)

    # When running in parallel, size chunks so there are ~4x as many chunks as
    # workers -- otherwise (e.g. fixed chunk_size=100 with only ~1350 samples)
    # only a handful of chunks exist and most cores sit idle. This matters most
    # at large adaptively selected orders, where one density pair can take seconds.
    if n_workers > 1:
        chunk_size = max(1, N // (4 * n_workers))

    chunk_bounds = [(i * chunk_size, min((i + 1) * chunk_size, N))
                    for i in range((N + chunk_size - 1) // chunk_size)]
    n_chunks = len(chunk_bounds)
    t_start = time.time()
    last_print = t_start

    if n_workers <= 1:
        # ----- Serial path -----
        for ci, (i0, i1) in enumerate(chunk_bounds):
            _, chunk_result, chunk_diag = _mhg_chunk_worker(
                (ci, c, Omegas, S_batch[i0:i1], M_trunc, M_step, M_max,
                 mhg_tol, ratio_window))
            out[:, i0:i1] = chunk_result
            _merge_mhg_diagnostics(diagnostics, chunk_diag)
            now = time.time()
            done = i1
            calls_done = done * G
            rate = calls_done / (now - t_start) if now > t_start else 0
            eta = (total_calls - calls_done) / rate if rate > 0 else float('inf')
            if (now - last_print >= progress_interval_sec) or (ci == n_chunks - 1):
                raw_note = (f", {diagnostics['raw_evaluations']:,} raw C evaluations"
                            if ci == n_chunks - 1 else "")
                print(f"      [{progress_label}] {done:>5d}/{N} S × {G} Ω "
                      f"= {calls_done:>7,}/{total_calls:,} density pairs{raw_note} "
                      f"({rate:6.1f}/s, {now-t_start:5.1f}s elapsed, "
                      f"~{eta:5.1f}s left)", flush=True)
                last_print = now
    else:
        # ----- Parallel path: Pool of workers -----
        from multiprocessing import Pool
        args_list = [(ci, c, Omegas, S_batch[i0:i1], M_trunc, M_step,
                      M_max, mhg_tol, ratio_window)
                     for ci, (i0, i1) in enumerate(chunk_bounds)]
        completed_calls = 0
        completed_chunks = 0
        with Pool(processes=n_workers) as pool:
            for chunk_id, chunk_result, chunk_diag in pool.imap_unordered(
                    _mhg_chunk_worker, args_list):
                i0, i1 = chunk_bounds[chunk_id]
                out[:, i0:i1] = chunk_result
                _merge_mhg_diagnostics(diagnostics, chunk_diag)
                completed_calls += (i1 - i0) * G
                completed_chunks += 1
                now = time.time()
                rate = completed_calls / (now - t_start) if now > t_start else 0
                eta = (total_calls - completed_calls) / rate if rate > 0 else float('inf')
                if (now - last_print >= progress_interval_sec) or (completed_chunks == n_chunks):
                    raw_note = (f", {diagnostics['raw_evaluations']:,} raw C evaluations"
                                if completed_chunks == n_chunks else "")
                    print(f"      [{progress_label}-p{n_workers}] "
                          f"{completed_chunks:>4d}/{n_chunks} chunks "
                          f"({completed_calls:>7,}/{total_calls:,} density pairs"
                          f"{raw_note}, "
                          f"{rate:6.1f}/s, {now-t_start:5.1f}s elapsed, "
                          f"~{eta:5.1f}s left)", flush=True)
                    last_print = now
    diagnostics["order_counts"] = dict(sorted(diagnostics["order_counts"].items()))
    if return_diagnostics:
        return out, diagnostics
    return out


# ============================================================
# Sanity check
# ============================================================

def verify_mhg():
    """Quick checks that the mhg C core is wired up correctly."""
    print("Verifying mhg (C core) setup...")

    # (1) Zero arguments => 0F1 = 1
    v = mhg_two_matrix(3.5, [0, 0, 0, 0], [0, 0, 0, 0], M_trunc=10)
    print(f"  0F1(3.5; 0, 0)              = {v:.6f}  (expect 1.0)")
    if abs(v - 1.0) >= 1e-6:
        raise RuntimeError("mhg zero-argument regression failed")

    # (2) p=1 scalar consistency: 0F1(c; x, y) at p=1 should equal scalar 0F1(c; x*y/4)
    #     (with our convention Omega/4)
    from scipy.special import hyp0f1
    c = 2.5
    x_om, y_s = 4.0, 3.0
    v_mat = mhg_two_matrix(c, [x_om], [y_s], M_trunc=30)
    v_sca = float(hyp0f1(c, (x_om / 4.0) * y_s))
    print(f"  0F1({c}; [{x_om}], [{y_s}])  = {v_mat:.6f}  vs scalar 0F1 = {v_sca:.6f}")
    if abs(v_mat - v_sca) / max(1.0, abs(v_sca)) >= 1e-3:
        raise RuntimeError("mhg scalar regression failed")

    # This large scalar case detects the old reciprocal-factor underflow.  The
    # broken core falsely plateaued 33% low while reporting a zero tail.
    z_large = 10_000.0
    v_large, order_large, assessment_large, _ = mhg_two_matrix_adaptive(
        3.5, [4.0 * z_large], [1.0], M_start=20, M_max=300,
        tol=MHG_CONV_TOL)
    expected_large = float(hyp0f1(3.5, z_large))
    rel_large = abs(v_large - expected_large) / expected_large
    print(f"  large scalar z={z_large:g}: rel.err={rel_large:.2e}, "
          f"adaptive M={order_large}")
    if not assessment_large.converged or rel_large > MHG_CONV_TOL:
        raise RuntimeError("mhg large-scalar underflow regression failed")

    # (3) Batched call: small (G, N)
    Om_mat = np.array([[4.0], [8.0]])
    S_mat = np.array([[3.0], [5.0]])
    vb = mhg_two_matrix_batch(c, Om_mat, S_mat, M_trunc=30)
    print(f"  batch shape {vb.shape}, values:\n{vb}")
    print("  mhg sanity checks passed.\n")


# ============================================================
# Log eigenvalue density (HW Appendix A.2), partial form
# ============================================================

def log_eigval_density_partial(S_batch, Omegas, c, M_trunc=20,
                              chunk_size=100, progress_label="mhg",
                              n_workers=1, M_step=MHG_DEFAULT_STEP,
                              M_max=MHG_DEFAULT_MAX,
                              mhg_tol=MHG_CONV_TOL,
                              return_diagnostics=False):
    """
    Log of the eigenvalue density f(x_1,...,x_p | Ω) up to an x-dependent
    constant. The omitted constant cancels in importance weights and in the
    LR-style test comparison used by ALFD.

    partial[g, n] = log 0F1^(2)(c; (1/4)Ω_g, S_n) - 0.5 * sum(Ω_g)
    """
    vals, diagnostics = chunked_mhg_batch(
        c, Omegas, S_batch, M_trunc=M_trunc, chunk_size=chunk_size,
        progress_label=progress_label, n_workers=n_workers,
        M_step=M_step, M_max=M_max, mhg_tol=mhg_tol,
        return_diagnostics=True)

    # Finiteness guard: a nan/inf here means the mhg series overflowed (the
    # historical [100,95,90] full-Omega bug) or otherwise misbehaved.  A nan
    # used to slip silently through np.maximum(nan,1e-300)=nan -> log(nan)=nan,
    # collapsing the power bound to 0.  Fail loudly instead, naming the
    # offending (Omega, S) so it can never pass unnoticed again.
    if not np.all(np.isfinite(vals)):
        gbad, nbad = np.argwhere(~np.isfinite(vals))[0]
        Om_arr = np.asarray(Omegas, dtype=float)
        raise AssertionError(
            "Non-finite 0F1^(2) value from mhg at M_start=%d.\n"
            "  Omega = %s\n  S     = %s\n  raw   = %r\n"
            "This indicates mhg overflow/under-truncation; do not proceed with "
            "a silent nan (it would zero out the power bound)."
            % (M_trunc, Om_arr[gbad].tolist(),
               np.asarray(S_batch, dtype=float)[nbad].tolist(),
               vals[gbad, nbad])
        )

    # 0F1^(2)(c; nonneg, nonneg) >= 1 always (its degree-0 term is 1 and all
    # coefficients are nonnegative).  Reject a material violation; clamp only
    # round-off at the last ulps rather than masking a broken evaluation.
    if np.any(vals < 1.0 - 1e-12):
        gbad, nbad = np.argwhere(vals < 1.0 - 1e-12)[0]
        raise AssertionError(
            "0F1^(2) value fell materially below its degree-zero term: "
            f"Omega index {gbad}, sample index {nbad}, value={vals[gbad, nbad]!r}")
    vals = np.maximum(vals, 1.0)
    sum_Om = np.asarray(Omegas, dtype=float).sum(axis=1)
    result = np.log(vals) - 0.5 * sum_Om[:, None]
    if return_diagnostics:
        return result, diagnostics
    return result


# ============================================================
# Null grid (lean: alt-nuisance + a few standard points)
# ============================================================



def common_null_grid_3d(alt_nuisance_rows, config_kappas, standard_points=None,
                        n_shapes=6, n_strengths=7, max_strength=100.0):
    """Deterministic common null grid for the three-dimensional nuisance cone.

    GKM use 42 log-spaced points for their *scalar* nuisance parameter.  The
    direct higher-dimensional analogue must cover both strength and shape.
    This construction crosses nested shape directions with log-spaced largest
    eigenvalues and adds the origin.  Defaults give ``1 + 6*7 = 43`` points.

    Shape candidates prioritize the configuration and any supplied stress
    directions, followed by rank-one/rank-two/rank-three boundaries and the
    alternative-path shapes having the smallest and largest third-eigenvalue
    share.  The production default requests nine shapes, retaining all of
    those directions for the configured experiments.  Additional shapes are
    deterministic interior directions.  The same ordered grid is used for
    every beta, which permits the GKM pooled null table to be cached and reused.
    Supplied stress points themselves are then appended as exact anchors when
    they are not already present on a ray; this preserves the historical null
    checks as well as their directions.
    """
    n_shapes = _validated_integer("n_shapes", n_shapes)
    n_strengths = _validated_integer("n_strengths", n_strengths)
    if n_shapes < 1 or n_strengths < 1:
        raise ValueError("n_shapes and n_strengths must be positive")
    if (not np.isfinite(max_strength)) or max_strength <= 0.1:
        raise ValueError("max_strength must be finite and greater than 0.1")

    config = np.asarray(config_kappas, dtype=float)
    alternatives = np.asarray(alt_nuisance_rows, dtype=float)
    standards = (np.empty((0, 3), dtype=float) if standard_points is None
                 else np.asarray(standard_points, dtype=float))
    if config.shape != (3,) or alternatives.ndim != 2 \
            or alternatives.shape[1] != 3 or standards.ndim != 2 \
            or standards.shape[1] != 3:
        raise ValueError("common_null_grid_3d requires three-dimensional rows")
    if (not np.all(np.isfinite(config))
            or not np.all(np.isfinite(alternatives))
            or not np.all(np.isfinite(standards))
            or np.any(config < 0.0) or np.any(alternatives < 0.0)
            or np.any(standards < 0.0)
            or np.any(np.diff(config) > 1e-10)
            or np.any(np.diff(alternatives, axis=1) > 1e-10)
            or np.any(np.diff(standards, axis=1) > 1e-10)
            or config[0] <= 0.0):
        raise ValueError("grid inputs must be finite, nonnegative and descending")

    def normalized(row):
        row = np.asarray(row, dtype=float)
        if row[0] <= 0.0:
            return None
        return tuple((row / row[0]).tolist())

    alt_shapes = [normalized(row) for row in alternatives if row[0] > 0.0]
    alt_shapes = [row for row in alt_shapes if row is not None]
    if alt_shapes:
        low_third = min(alt_shapes, key=lambda row: (row[2], row[1]))
        high_third = max(alt_shapes, key=lambda row: (row[2], row[1]))
    else:
        low_third, high_third = (1.0, 0.75, 0.15), (1.0, 0.7, 0.55)

    candidates = [normalized(config)]
    candidates.extend(normalized(row) for row in standards)
    candidates.extend([
        (1.0, 0.0, 0.0),
        (1.0, 1.0, 0.0),
        (1.0, 1.0, 1.0),
        low_third,
        high_third,
        (1.0, 0.25, 0.0),
        (1.0, 0.5, 0.0),
        (1.0, 0.75, 0.0),
        (1.0, 0.35, 0.15),
        (1.0, 0.5, 0.25),
        (1.0, 0.75, 0.5),
        (1.0, 0.9, 0.1),
        (1.0, 0.9, 0.6),
        (1.0, 0.6, 0.55),
    ])
    shapes, seen = [], set()
    for row in candidates:
        if row is None:
            continue
        ordered = tuple(sorted((float(x) for x in row), reverse=True))
        key = tuple(round(x, 12) for x in ordered)
        if key not in seen:
            seen.add(key)
            shapes.append(ordered)
        if len(shapes) == n_shapes:
            break
    if len(shapes) < n_shapes:
        # A deterministic low-discrepancy fallback for unusually large grids.
        index = 1
        while len(shapes) < n_shapes:
            second = ((index * 0.6180339887498949) % 1.0)
            third = ((index * 0.4142135623730950) % 1.0) * second
            row = (1.0, second, third)
            key = tuple(round(x, 12) for x in row)
            if key not in seen:
                seen.add(key)
                shapes.append(row)
            index += 1

    strengths = np.geomspace(0.1, float(max_strength), n_strengths)
    grid = [(0.0, 0.0, 0.0)]
    for shape in shapes:
        for strength in strengths:
            grid.append(tuple(float(strength * x) for x in shape))
    cartesian_size = 1 + n_shapes * n_strengths
    if len(grid) != cartesian_size:
        raise AssertionError("common-grid construction lost rows")
    grid_keys = {tuple(round(x, 12) for x in row) for row in grid}
    for row in standards:
        anchor = tuple(float(x) for x in row)
        key = tuple(round(x, 12) for x in anchor)
        if key not in grid_keys:
            grid_keys.add(key)
            grid.append(anchor)
    return grid


# ============================================================
# EMW/GKM calibration primitives
# ============================================================

@dataclass(frozen=True)
class TailRule:
    threshold: float
    tie_probability: float
    empirical_size: float
    method: str


@dataclass
class EMWFitResult:
    weights: np.ndarray
    log_weights: np.ndarray
    mu: np.ndarray
    rejection_probabilities: np.ndarray
    training_rule: TailRule
    iterations: int
    converged: bool
    complementarity_residual: float


@dataclass
class PooledISBank:
    """Stratified sample from an equal mixture of finite null laws.

    ``base_weights`` contain the *fixed* stratified integration factors.  They
    sum to one, but the target contributions ``base*f_j/q`` must never be
    self-normalized: GKM equation (D.1) is ordinary importance sampling.
    """
    grid: np.ndarray
    eigs: np.ndarray
    log_f: np.ndarray
    log_q: np.ndarray
    base_weights: np.ndarray
    strata: np.ndarray
    n_per_stratum: int
    role: str
    bank_id: str
    sampling_scheme: str = "stratified_null_gkm_is"
    mhg_diagnostics: dict = None
    sampling_seed: int = None
    k_eff: int = None
    experiment_signature: str = None
    settings_json: str = None
    content_signature: str = None


@dataclass
class GKMDirectResult:
    r"""Outputs of GKM Supplement D.3.2 Steps 5--9 for one alternative.

    ``mixture_power`` is :math:`\bar\pi` from Step 7.  ``bound`` is the
    grid-adjusted :math:`\tilde\pi` from Step 9, the quantity GKM use as the
    point-optimal power bound in their Figure 3.  Both are Monte Carlo point
    estimates, not confidence endpoints.
    """
    bound: float
    bound_se: float
    mixture_power: float
    mixture_power_se: float
    epsilon_grid: float
    weights: np.ndarray
    log_weights: np.ndarray
    fit_rejection_probabilities: np.ndarray
    grid_rejection_probabilities: np.ndarray
    fit_iterations: int
    mixture_rule: TailRule
    grid_rule: TailRule
    importance_diagnostics: dict
    mhg_diagnostics: dict


def _softmax(x):
    x = np.asarray(x, dtype=float)
    return np.exp(x - logsumexp(x))


def _log_probability_weights(weights):
    weights = np.asarray(weights, dtype=float)
    result = np.full(weights.shape, -np.inf)
    np.log(weights, out=result, where=weights > 0.0)
    return result


def tail_rejection_probabilities(scores, rule):
    """Expected rejection indicators, including explicit tie randomization."""
    scores = np.asarray(scores, dtype=float)
    return ((scores > rule.threshold).astype(float)
            + rule.tie_probability * (scores == rule.threshold))




def calibrate_raw_weighted_tail(scores, raw_contributions, alpha,
                                method="gkm_ordinary_is_exact"):
    """Calibrate an absolute ordinary-IS tail without self-normalization.

    GKM equation (D.1) estimates an integral with fixed stratified weights.
    Normalizing those contributions by their realized sum would silently turn
    it into a different, biased finite-sample estimator.  Ties are randomized
    so the *raw* estimated integral equals ``alpha`` exactly.
    """
    scores = np.asarray(scores, dtype=float).ravel()
    weights = np.asarray(raw_contributions, dtype=float).ravel()
    if scores.size == 0 or scores.shape != weights.shape:
        raise ValueError("scores and raw contributions must be nonempty and aligned")
    if (not np.all(np.isfinite(scores)) or not np.all(np.isfinite(weights))
            or np.any(weights < 0.0) or not (0.0 < alpha < 1.0)):
        raise ValueError("raw calibration inputs must be finite/nonnegative")
    total = float(weights.sum())
    if total < alpha - 1e-15:
        raise ValueError(
            f"ordinary-IS mass {total:.6g} is below alpha={alpha}; "
            "the calibration bank is unusable")

    order = np.argsort(scores, kind="mergesort")[::-1]
    sorted_scores = scores[order]
    sorted_weights = weights[order]
    above = 0.0
    i = 0
    while i < scores.size:
        j = i + 1
        while j < scores.size and sorted_scores[j] == sorted_scores[i]:
            j += 1
        tie_mass = float(sorted_weights[i:j].sum())
        if above <= alpha + 1e-15 and alpha <= above + tie_mass + 1e-15:
            if tie_mass <= 0.0:
                if np.isclose(above, alpha, rtol=0.0, atol=1e-15):
                    return TailRule(float(sorted_scores[i]), 0.0,
                                    float(above), method)
                i = j
                continue
            rho = float(np.clip((alpha - above) / tie_mass, 0.0, 1.0))
            return TailRule(float(sorted_scores[i]), rho,
                            float(above + rho * tie_mass), method)
        above += tie_mass
        i = j
    raise RuntimeError("failed to locate ordinary-IS tail quantile")


def _validate_pooled_is_bank(bank):
    if not isinstance(bank, PooledISBank):
        raise TypeError("expected a PooledISBank")
    grid = np.asarray(bank.grid, dtype=float)
    log_f = np.asarray(bank.log_f, dtype=float)
    log_q = np.asarray(bank.log_q, dtype=float)
    base = np.asarray(bank.base_weights, dtype=float)
    raw_strata = np.asarray(bank.strata)
    if not np.issubdtype(raw_strata.dtype, np.integer):
        raise ValueError("invalid pooled GKM importance-sampling bank: "
                         "strata must have an integer dtype")
    strata = raw_strata.astype(int, copy=False)
    eigs = np.asarray(bank.eigs, dtype=float)
    H = grid.shape[0] if grid.ndim == 2 else 0
    N = eigs.shape[0] if eigs.ndim == 2 else 0
    try:
        n_per_stratum = _validated_integer(
            "n_per_stratum", bank.n_per_stratum, minimum=2)
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid pooled GKM importance-sampling bank") from exc
    if (H == 0 or N == 0 or log_f.shape != (H, N)
            or log_q.shape != (N,) or base.shape != (N,)
            or strata.shape != (N,) or not np.all(np.isfinite(grid))
            or not np.all(np.isfinite(eigs)) or not np.all(np.isfinite(log_f))
            or not np.all(np.isfinite(log_q)) or not np.all(np.isfinite(base))
            or np.any(base <= 0.0) or not np.isclose(base.sum(), 1.0,
                                                     atol=1e-12)
            or grid.ndim != 2 or eigs.ndim != 2
            or eigs.shape[1] != grid.shape[1] + 1
            or np.any(grid < 0.0) or np.any(np.diff(grid, axis=1) > 1e-10)
            or np.any(eigs < -1e-10) or np.any(np.diff(eigs, axis=1) > 1e-8)
            or np.any(strata < 0) or np.any(strata >= H)
            or bank.sampling_scheme != "stratified_null_gkm_is"
            or bank.role != "gkm"
            or not isinstance(bank.bank_id, str) or not bank.bank_id):
        raise ValueError("invalid pooled GKM importance-sampling bank")
    expected_base = np.full(N, 1.0 / N)
    if not np.allclose(
            base, expected_base, rtol=0.0,
            atol=10.0 * np.finfo(float).eps / max(1, N)):
        raise ValueError(
            "invalid pooled GKM importance-sampling bank: equal-mixture "
            "stratification requires base_i=1/N")
    counts = np.bincount(strata, minlength=H)
    expected_strata = np.repeat(np.arange(H, dtype=int), n_per_stratum)
    if (np.any(counts != n_per_stratum)
            or N != H * n_per_stratum
            or not np.array_equal(strata, expected_strata)):
        raise ValueError("pooled bank strata are incomplete or unbalanced")
    expected_q = logsumexp(log_f - math.log(H), axis=0)
    if not np.allclose(expected_q, log_q, rtol=0.0, atol=2e-12):
        raise ValueError("pooled bank log_q is inconsistent with its null rows")
    return H, N


def _pooled_is_ratios(bank):
    H, _ = _validate_pooled_is_bank(bank)
    ratios = np.exp(bank.log_f - bank.log_q[None, :])
    if (not np.all(np.isfinite(ratios)) or np.any(ratios < 0.0)
            or np.max(ratios) > H * (1.0 + 2e-11)):
        raise RuntimeError("GKM importance ratios violate the equal-mixture cap")
    return ratios


def gkm_importance_rejection_probabilities(bank, rejection_probabilities):
    """Ordinary stratified-IS rejection probabilities for every grid null."""
    rejection = np.asarray(rejection_probabilities, dtype=float).ravel()
    _, N = _validate_pooled_is_bank(bank)
    if (rejection.shape != (N,) or not np.all(np.isfinite(rejection))
            or np.any((rejection < 0.0) | (rejection > 1.0))):
        raise ValueError("rejection probabilities must be finite values in [0,1]")
    ratios = _pooled_is_ratios(bank)
    return ratios @ (bank.base_weights * rejection)


def pooled_is_diagnostics(bank, rejection_probabilities=None):
    """Mass, cap and ESS diagnostics for the ordinary GKM weights."""
    ratios = _pooled_is_ratios(bank)
    contributions = ratios * bank.base_weights[None, :]
    mass = contributions.sum(axis=1)
    ess = mass * mass / np.sum(contributions * contributions, axis=1)
    result = dict(
        raw_mass=mass,
        observed_max_ratio=np.max(ratios, axis=1),
        theoretical_max_ratio=float(bank.grid.shape[0]),
        kish_ess=ess,
        kish_ess_fraction=ess / ratios.shape[1],
    )
    if rejection_probabilities is not None:
        rejection = np.asarray(rejection_probabilities, dtype=float).ravel()
        if (rejection.shape != (ratios.shape[1],)
                or not np.all(np.isfinite(rejection))
                or np.any((rejection < 0.0) | (rejection > 1.0))):
            raise ValueError(
                "rejection probabilities must be finite values in [0,1]")
        weighted = contributions * rejection[None, :]
        tail_mass = weighted.sum(axis=1)
        denom = np.sum(weighted * weighted, axis=1)
        tail_ess = np.zeros_like(tail_mass)
        np.divide(tail_mass * tail_mass, denom, out=tail_ess, where=denom > 0.0)
        result["tail_mass"] = tail_mass
        result["tail_ess"] = tail_ess
    return result


def common_grid_raw_is_tail_rule(scores, raw_target_contributions, alpha,
                                 minimum_rule=None):
    """GKM Step 8 common cutoff using ordinary-IS null contributions."""
    scores = np.asarray(scores, dtype=float).ravel()
    rows = np.asarray(raw_target_contributions, dtype=float)
    if (rows.ndim != 2 or rows.shape[1] != scores.size
            or not np.all(np.isfinite(rows)) or np.any(rows < 0.0)):
        raise ValueError("raw target contributions must be H by N and nonnegative")
    if minimum_rule is not None:
        if (not isinstance(minimum_rule, TailRule)
                or not np.isfinite(minimum_rule.threshold)
                or not (0.0 <= minimum_rule.tie_probability <= 1.0)):
            raise ValueError("minimum_rule must be a finite TailRule")
    individual = [calibrate_raw_weighted_tail(
        scores, row, alpha, method="gkm_grid_row_ordinary_is") for row in rows]
    threshold = max(rule.threshold for rule in individual)
    if minimum_rule is not None:
        threshold = max(threshold, minimum_rule.threshold)

    rho = 1.0
    sizes = []
    for row in rows:
        above = float(row[scores > threshold].sum())
        tied = float(row[scores == threshold].sum())
        if above > alpha + 2e-12:
            raise RuntimeError("ordinary-IS grid cutoff failed size control")
        if tied > 0.0:
            rho = min(rho, max(0.0, (alpha - above) / tied))
        sizes.append((above, tied))
    if minimum_rule is not None and threshold == minimum_rule.threshold:
        rho = min(rho, minimum_rule.tie_probability)
    rho = float(np.clip(rho, 0.0, 1.0))
    attained = max(above + rho * tied for above, tied in sizes)
    return TailRule(float(threshold), rho, float(attained),
                    "gkm_grid_ordinary_is_size_at_most_alpha")






def fit_gkm_weights_is(bank, log_g, alpha=0.05, n_iter=600,
                       step_size=2.0, active_weight_tol=1e-12):
    """Run GKM Supplement D.3.2 Step 5 without algorithmic additions.

    GKM initialize every coordinate of ``mu`` at -2, use the fixed scalar
    step ``omega=2``, and perform exactly ``O=600`` updates.  In particular,
    this implementation has no sign-switch damping, early stopping, or
    best-iterate selection.  ``n_iter`` remains configurable for plumbing
    tests, but reportable runs use 600.
    """
    H, N = _validate_pooled_is_bank(bank)
    log_g = np.asarray(log_g, dtype=float).ravel()
    n_iter = _validated_integer("n_iter", n_iter)
    numeric = np.asarray([alpha, step_size, active_weight_tol], dtype=float)
    if (log_g.shape != (N,) or not np.all(np.isfinite(log_g))
            or not np.all(np.isfinite(numeric))
            or not (0.0 < alpha < 1.0) or step_size <= 0.0
            or not (0.0 <= active_weight_tol < 1.0)):
        raise ValueError("invalid direct-GKM fit inputs")

    ratios = _pooled_is_ratios(bank)
    mu = np.full(H, -2.0)
    for _ in range(n_iter):
        log_threshold = logsumexp(mu[:, None] + bank.log_f, axis=0)
        rejection = (log_g > log_threshold).astype(float)
        rp = ratios @ (bank.base_weights * rejection)
        mu += float(step_size) * (rp - alpha)

    # Keep the normalized weights in log form for every likelihood-ratio
    # calculation.  With 600 fixed updates, a mathematically positive GKM
    # weight can be smaller than the least representable float; exponentiating
    # it must not silently remove that null row from the mixture.
    log_weights = mu - logsumexp(mu)
    weights = np.exp(log_weights)
    log_mix = logsumexp(
        log_weights[:, None] + bank.log_f, axis=0)
    scores = log_g - log_mix
    mixture_contributions = (
        bank.base_weights * np.exp(log_mix - bank.log_q))
    mixture_rule = calibrate_raw_weighted_tail(
        scores, mixture_contributions, alpha,
        method="gkm_step6_mixture_ordinary_is")
    rejection = tail_rejection_probabilities(scores, mixture_rule)
    rp = ratios @ (bank.base_weights * rejection)

    # This is a diagnostic only; GKM always use the O-th iterate.
    active = weights > active_weight_tol
    active_residual = (float(np.max(np.abs(rp[active] - alpha)))
                       if np.any(active) else 0.0)
    slack_residual = (float(np.max(np.maximum(rp[~active] - alpha, 0.0)))
                      if np.any(~active) else 0.0)
    residual = max(active_residual, slack_residual)
    max_contribution = float(np.max(
        ratios * bank.base_weights[None, :]))
    diagnostic_tolerance = max(2.0 * max_contribution, 5e-4)
    return EMWFitResult(
        weights=weights, log_weights=log_weights, mu=mu,
        rejection_probabilities=rp,
        training_rule=mixture_rule, iterations=n_iter,
        converged=bool(residual <= diagnostic_tolerance),
        complementarity_residual=float(residual))




def _pooled_experiment_settings(grid, k_eff, M_start, M_step, M_max,
                                mhg_tol, metadata=None):
    """Canonical identity of the reusable direct-GKM null experiment."""
    settings = dict(
        schema_version=2, method=POOLED_IS_METHOD,
        grid=np.asarray(grid, dtype=float).tolist(), k_eff=int(k_eff),
        M_start=int(M_start), M_step=int(M_step), M_max=int(M_max),
        mhg_tol=float(mhg_tol), density_formula="wishart_eigen_kernel_v1",
    )
    if metadata:
        settings["provenance"] = _json_safe(metadata)
    return settings


def _canonical_pooled_mhg_diagnostics(diagnostics, expected_pairs):
    """Validate and canonicalize the numerical diagnostics attached to a bank."""
    if not isinstance(diagnostics, dict):
        raise ValueError("pooled bank MHG diagnostics must be a dictionary")
    required = {
        "pairs", "raw_evaluations", "order_counts", "max_order",
        "max_remainder_ratio",
    }
    if not required.issubset(diagnostics):
        raise ValueError("pooled bank MHG diagnostics are incomplete")
    try:
        pairs = int(diagnostics["pairs"])
        raw = int(diagnostics["raw_evaluations"])
        max_order = int(diagnostics["max_order"])
        max_remainder = float(diagnostics["max_remainder_ratio"])
        counts = {int(key): int(value)
                  for key, value in diagnostics["order_counts"].items()}
    except (TypeError, ValueError, AttributeError) as exc:
        raise ValueError("pooled bank MHG diagnostics are malformed") from exc
    # A pair with an exactly zero hypergeometric argument is evaluated by the
    # analytic 0F1=1 shortcut and is recorded at order zero without entering
    # the C routine.  Every other pair must contribute at least one raw C
    # evaluation; adaptive retries may make ``raw`` larger.
    minimum_raw = pairs - counts.get(0, 0)
    if (pairs != int(expected_pairs) or raw < minimum_raw or max_order < 0
            or not np.isfinite(max_remainder) or max_remainder < 0.0
            or any(key < 0 or value < 0 for key, value in counts.items())
            or sum(counts.values()) != pairs
            or (counts and max(counts) != max_order)):
        raise ValueError("pooled bank MHG diagnostics are inconsistent")
    return json.dumps(
        _json_safe(diagnostics), sort_keys=True, separators=(",", ":"))


def _pooled_bank_content_signature(diagnostics_json=None, **arrays):
    """Hash the exact numeric payload of a reusable pooled bank."""
    digest = hashlib.sha256()
    for name in sorted(arrays):
        array = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("utf-8"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    if diagnostics_json is not None:
        digest.update(b"mhg_diagnostics_json")
        digest.update(str(diagnostics_json).encode("utf-8"))
    return digest.hexdigest()


def _pooled_bank_settings(grid, role, k_eff, n_per_stratum, seed,
                          M_start, M_step, M_max, mhg_tol, metadata=None):
    experiment = _pooled_experiment_settings(
        grid, k_eff, M_start, M_step, M_max, mhg_tol, metadata)
    experiment_json = json.dumps(
        experiment, sort_keys=True, separators=(",", ":"))
    settings = dict(
        **experiment, role=str(role), n_per_stratum=int(n_per_stratum),
        seed=int(seed),
        experiment_signature=hashlib.sha256(
            experiment_json.encode()).hexdigest())
    return settings


def build_or_load_pooled_is_bank(grid, k_eff, n_per_stratum, seed,
                                 M_start=20, M_step=MHG_DEFAULT_STEP,
                                 M_max=MHG_DEFAULT_MAX, mhg_tol=MHG_CONV_TOL,
                                 n_workers=1, role="gkm",
                                 cache_dir=None, cache_metadata=None):
    """Build/cache one beta-invariant GKM stratified null bank."""
    grid = _validated_null_grid(grid, 3, "common pooled null grid")
    n_per_stratum = _validated_integer(
        "n_per_stratum", n_per_stratum, minimum=2)
    k_eff = _validated_integer("k_eff", k_eff)
    if role != "gkm":
        raise ValueError("direct calculation requires pooled bank role 'gkm'")
    settings = _pooled_bank_settings(
        grid, role, k_eff, n_per_stratum, seed, M_start, M_step, M_max,
        mhg_tol, cache_metadata)
    canonical = json.dumps(settings, sort_keys=True, separators=(",", ":"))
    bank_id = hashlib.sha256(canonical.encode()).hexdigest()
    cache_path = (None if cache_dir is None else os.path.join(
        cache_dir, f"pooled_{role}_{bank_id[:16]}.npz"))

    if cache_path is not None and os.path.isfile(cache_path):
        try:
            with np.load(cache_path, allow_pickle=False) as archive:
                saved_id = str(np.asarray(archive["bank_id"]).item())
                if saved_id != bank_id:
                    raise ValueError("bank signature differs")
                saved_settings = str(
                    np.asarray(archive["settings_json"]).item())
                if saved_settings != canonical:
                    raise ValueError("canonical bank settings differ")
                raw_strata = np.asarray(archive["strata"])
                if not np.issubdtype(raw_strata.dtype, np.integer):
                    raise ValueError("cached strata are not integer-valued")
                saved_diagnostics_json = str(np.asarray(
                    archive["mhg_diagnostics_json"]).item())
                diagnostics = json.loads(saved_diagnostics_json)
                canonical_diagnostics_json = \
                    _canonical_pooled_mhg_diagnostics(
                        diagnostics,
                        np.asarray(archive["log_f"]).shape[0]
                        * np.asarray(archive["log_f"]).shape[1])
                if saved_diagnostics_json != canonical_diagnostics_json:
                    raise ValueError("cached MHG diagnostics are not canonical")
                saved_content_signature = str(np.asarray(
                    archive["content_signature"]).item())
                bank = PooledISBank(
                    grid=np.asarray(archive["grid"], dtype=float).copy(),
                    eigs=np.asarray(archive["eigs"], dtype=float).copy(),
                    log_f=np.asarray(archive["log_f"], dtype=float).copy(),
                    log_q=np.asarray(archive["log_q"], dtype=float).copy(),
                    base_weights=np.asarray(
                        archive["base_weights"], dtype=float).copy(),
                    strata=raw_strata.astype(int, copy=True),
                    n_per_stratum=int(
                        np.asarray(archive["n_per_stratum"]).item()),
                    role=str(np.asarray(archive["role"]).item()),
                    bank_id=saved_id, mhg_diagnostics=diagnostics,
                    sampling_seed=int(
                        np.asarray(archive["sampling_seed"]).item()),
                    k_eff=int(np.asarray(archive["k_eff"]).item()),
                    experiment_signature=str(np.asarray(
                        archive["experiment_signature"]).item()),
                    settings_json=saved_settings,
                    content_signature=saved_content_signature)
            computed_content_signature = _pooled_bank_content_signature(
                grid=bank.grid, eigs=bank.eigs, log_f=bank.log_f,
                log_q=bank.log_q, base_weights=bank.base_weights,
                strata=bank.strata,
                diagnostics_json=canonical_diagnostics_json)
            if computed_content_signature != bank.content_signature:
                raise ValueError("cached bank numeric content hash differs")
            _validate_pooled_is_bank(bank)
            if (bank.role != role or bank.sampling_seed != int(seed)
                    or bank.k_eff != k_eff
                    or bank.experiment_signature
                    != settings["experiment_signature"]
                    or not np.array_equal(
                        bank.grid, np.asarray(grid, dtype=float))):
                raise ValueError("cached bank metadata differs from the request")
            print(f"  loaded compatible {role} pooled bank: {cache_path}",
                  flush=True)
            return bank
        except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"Cannot trust pooled-bank cache {cache_path}: {exc}. "
                "Move the damaged cache aside and rerun.") from exc

    H = len(grid)
    p = 4
    rng = np.random.default_rng(seed)
    eigs = np.empty((H * n_per_stratum, p))
    strata = np.repeat(np.arange(H, dtype=int), n_per_stratum)
    for j, row in enumerate(grid):
        start = j * n_per_stratum
        stop = start + n_per_stratum
        M_null = build_M(list(row) + [0.0], k_eff)
        eigs[start:stop] = eigenvalues_descending(
            simulate_Xi(M_null, n_per_stratum, rng))
    omegas = np.asarray([list(row) + [0.0] for row in grid], dtype=float)
    log_f, diagnostics = log_eigval_density_partial(
        eigs, omegas, k_eff / 2.0, M_trunc=M_start, chunk_size=100,
        progress_label=f"common-{role}-null", n_workers=n_workers,
        M_step=M_step, M_max=M_max, mhg_tol=mhg_tol,
        return_diagnostics=True)
    log_q = logsumexp(log_f - math.log(H), axis=0)
    base = np.full(H * n_per_stratum, 1.0 / (H * n_per_stratum))
    diagnostics_json = _canonical_pooled_mhg_diagnostics(
        diagnostics, H * (H * n_per_stratum))
    content_signature = _pooled_bank_content_signature(
        grid=np.asarray(grid, dtype=float), eigs=eigs, log_f=log_f,
        log_q=log_q, base_weights=base, strata=strata,
        diagnostics_json=diagnostics_json)
    bank = PooledISBank(
        grid=np.asarray(grid, dtype=float), eigs=eigs, log_f=log_f,
        log_q=log_q, base_weights=base, strata=strata,
        n_per_stratum=n_per_stratum, role=role, bank_id=bank_id,
        mhg_diagnostics=diagnostics, sampling_seed=int(seed),
        k_eff=k_eff,
        experiment_signature=settings["experiment_signature"],
        settings_json=canonical, content_signature=content_signature)
    _validate_pooled_is_bank(bank)
    if cache_path is not None:
        os.makedirs(cache_dir, exist_ok=True)
        _atomic_savez(
            cache_path, bank_id=np.array(bank_id),
            settings_json=np.array(canonical), role=np.array(role),
            grid=bank.grid, eigs=bank.eigs, log_f=bank.log_f,
            log_q=bank.log_q, base_weights=bank.base_weights,
            strata=bank.strata,
            n_per_stratum=np.array(bank.n_per_stratum),
            sampling_seed=np.array(bank.sampling_seed),
            k_eff=np.array(bank.k_eff),
            experiment_signature=np.array(bank.experiment_signature),
            content_signature=np.array(bank.content_signature),
            mhg_diagnostics_json=np.array(diagnostics_json))
        print(f"  saved {role} pooled bank cache: {cache_path}", flush=True)
    return bank


def _authenticated_pooled_bank_settings(bank):
    """Parse and authenticate a bank's settings and numeric content."""
    _validate_pooled_is_bank(bank)
    if (not isinstance(bank.settings_json, str) or not bank.settings_json
            or not isinstance(bank.content_signature, str)
            or len(bank.content_signature) != 64):
        raise ValueError("pooled bank lacks authenticated settings/content")
    try:
        settings = json.loads(bank.settings_json)
    except json.JSONDecodeError as exc:
        raise ValueError("pooled bank settings_json is malformed") from exc
    canonical = json.dumps(settings, sort_keys=True, separators=(",", ":"))
    if canonical != bank.settings_json:
        raise ValueError("pooled bank settings_json is not canonical")
    if hashlib.sha256(canonical.encode()).hexdigest() != bank.bank_id:
        raise ValueError("pooled bank_id does not authenticate settings_json")
    required = {
        "schema_version", "method", "role", "grid", "k_eff",
        "n_per_stratum", "seed", "M_start", "M_step", "M_max",
        "mhg_tol", "density_formula", "experiment_signature",
    }
    if not required.issubset(settings):
        raise ValueError("pooled bank settings are incomplete")
    experiment = {key: value for key, value in settings.items()
                  if key not in {
                      "role", "n_per_stratum", "seed",
                      "experiment_signature"}}
    experiment_json = json.dumps(
        experiment, sort_keys=True, separators=(",", ":"))
    expected_experiment_signature = hashlib.sha256(
        experiment_json.encode()).hexdigest()
    if (settings["schema_version"] != 2
            or settings["method"] != POOLED_IS_METHOD
            or settings["role"] != bank.role
            or int(settings["n_per_stratum"]) != bank.n_per_stratum
            or int(settings["seed"]) != bank.sampling_seed
            or int(settings["k_eff"]) != bank.k_eff
            or settings["experiment_signature"]
            != expected_experiment_signature
            or bank.experiment_signature != expected_experiment_signature
            or not np.array_equal(
                np.asarray(settings["grid"], dtype=float), bank.grid)):
        raise ValueError("pooled bank metadata is inconsistent")
    diagnostics_json = _canonical_pooled_mhg_diagnostics(
        bank.mhg_diagnostics, bank.log_f.shape[0] * bank.log_f.shape[1])
    expected_content_signature = _pooled_bank_content_signature(
        grid=bank.grid, eigs=bank.eigs, log_f=bank.log_f,
        log_q=bank.log_q, base_weights=bank.base_weights,
        strata=bank.strata, diagnostics_json=diagnostics_json)
    if expected_content_signature != bank.content_signature:
        raise ValueError("pooled bank content signature differs")
    return settings


# ============================================================
# ALFD with eigenvalue density
# ============================================================



def _score_from_log_densities(log_densities, weights=None, *,
                              log_weights=None):
    """Log likelihood-ratio score for a normalized finite null mixture.

    Direct GKM calculations pass ``log_weights`` so subnormal mixture masses
    remain represented.  ``weights`` is retained for small public/test uses.
    """
    if (weights is None) == (log_weights is None):
        raise ValueError("provide exactly one of weights or log_weights")
    if log_weights is None:
        normalized_logs = _log_probability_weights(weights)
    else:
        normalized_logs = np.asarray(log_weights, dtype=float)
        if (normalized_logs.ndim != 1
                or not np.all(np.isfinite(normalized_logs))
                or not np.isclose(logsumexp(normalized_logs), 0.0,
                                  rtol=0.0, atol=2e-12)):
            raise ValueError("log_weights must be finite and normalized")
    G = len(normalized_logs)
    log_mix = logsumexp(
        normalized_logs[:, None] + log_densities[:G], axis=0)
    return log_densities[G] - log_mix


def _combine_phase_mhg_diagnostics(phases):
    counts = Counter()
    combined = dict(pairs=0, raw_evaluations=0, max_order=0,
                    max_remainder_ratio=0.0, phases=phases)
    for diagnostics in phases.values():
        combined["pairs"] += int(diagnostics["pairs"])
        combined["raw_evaluations"] += int(diagnostics["raw_evaluations"])
        combined["max_order"] = max(combined["max_order"],
                                    int(diagnostics["max_order"]))
        combined["max_remainder_ratio"] = max(
            combined["max_remainder_ratio"],
            float(diagnostics["max_remainder_ratio"]))
        counts.update({int(k): int(v)
                       for k, v in diagnostics["order_counts"].items()})
    combined["order_counts"] = dict(sorted(counts.items()))
    return combined




def _exact_gkm_result(alpha, G):
    rule = TailRule(0.0, float(alpha), float(alpha),
                    "exact_null_randomization")
    return GKMDirectResult(
        bound=float(alpha), bound_se=0.0,
        mixture_power=float(alpha), mixture_power_se=0.0,
        epsilon_grid=0.0, weights=np.full(G, 1.0 / G),
        log_weights=np.full(G, -math.log(G)),
        fit_rejection_probabilities=np.full(G, alpha),
        grid_rejection_probabilities=np.full(G, alpha),
        fit_iterations=0, mixture_rule=rule, grid_rule=rule,
        importance_diagnostics=dict(exact_null=True),
        mhg_diagnostics=dict(
            pairs=0, raw_evaluations=0, max_order=0,
            max_remainder_ratio=0.0, order_counts={}, phases={}))


def _validated_null_grid(rows, dimension, name):
    """Return finite, nonnegative, descending nuisance-eigenvalue tuples."""
    try:
        result = [tuple(float(x) for x in row) for row in rows]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric rows") from exc
    if not result:
        raise ValueError(f"{name} must be nonempty")
    for row in result:
        values = np.asarray(row, dtype=float)
        if (len(row) != dimension or not np.all(np.isfinite(values))
                or np.any(values < 0.0) or np.any(np.diff(values) > 1e-10)):
            raise ValueError(
                f"each {name} tuple must be finite, nonnegative, descending, "
                f"and have length {dimension}")
    return result






def gkm_eigval_bound_from_pooled_bank(
        kappas_alt, bank, k_eff, alpha=0.05, n_sim_power=100000,
        n_iter=600, seed=42, verbose=True, n_workers=1,
        M_trunc=20, M_step=MHG_DEFAULT_STEP,
        M_max=MHG_DEFAULT_MAX, mhg_tol=MHG_CONV_TOL):
    """Direct m_W=3 implementation of GKM Supplement D.3.2 Steps 3--9.

    The supplied bank is GKM's single Step-1/2 pooled ``N0`` experiment and
    is reused for fitting, Step-6 mixture calibration, and Step-8 grid
    adjustment.  Step 7 and Step 9 are evaluated on the same fresh ``N1``
    alternative draws.  No support compression, independent audit,
    sample-split recalibration, or Monte Carlo confidence correction is used.
    """
    k_eff = _validated_integer("k_eff", k_eff)
    n_sim_power = _validated_integer(
        "n_sim_power", n_sim_power, minimum=2)
    n_iter = _validated_integer("n_iter", n_iter)
    seed = _validated_integer("seed", seed, minimum=0)
    M_trunc = _validated_integer("M_trunc", M_trunc)
    M_step = _validated_integer("M_step", M_step)
    M_max = _validated_integer("M_max", M_max)
    if (M_trunc > M_max or not np.isfinite(mhg_tol)
            or not (1e-13 <= mhg_tol < 1.0)):
        raise ValueError("invalid adaptive-M settings")

    H, _ = _validate_pooled_is_bank(bank)
    if bank.role != "gkm":
        raise ValueError("direct GKM calculation requires a role='gkm' bank")
    settings = _authenticated_pooled_bank_settings(bank)
    if bank.k_eff is None or int(bank.k_eff) != k_eff:
        raise ValueError("pooled bank was built for a different k_eff")
    if seed == int(bank.sampling_seed):
        raise ValueError("alternative-power seed must differ from the bank seed")
    if (int(settings["M_start"]) != M_trunc
            or int(settings["M_step"]) != M_step
            or int(settings["M_max"]) != M_max
            or float(settings["mhg_tol"]) != float(mhg_tol)):
        raise ValueError(
            "call-time adaptive-M settings differ from pooled-bank settings")

    kappas_alt = np.asarray(kappas_alt, dtype=float)
    p = kappas_alt.size
    if (p != 4 or not np.all(np.isfinite(kappas_alt))
            or np.any(kappas_alt < 0.0)
            or np.any(np.diff(kappas_alt) > 1e-10)):
        raise ValueError("direct GKM p=4 alternative eigenvalues are invalid")
    if k_eff < p or not (0.0 < alpha < 1.0):
        raise ValueError("require k_eff >= 4 and 0 < alpha < 1")

    omega_alt = kappas_alt[None, :]
    phase_diagnostics = {}
    log_g_train_matrix, phase_diagnostics["training_alternative"] = \
        log_eigval_density_partial(
            bank.eigs, omega_alt, k_eff / 2.0,
            M_trunc=M_trunc, chunk_size=100,
            progress_label="gkm-train-g", n_workers=n_workers,
            M_step=M_step, M_max=M_max, mhg_tol=mhg_tol,
            return_diagnostics=True)
    log_g_train = log_g_train_matrix[0]
    fit = fit_gkm_weights_is(
        bank, log_g_train, alpha=alpha, n_iter=n_iter, step_size=2.0)

    log_mix_train = logsumexp(
        fit.log_weights[:, None] + bank.log_f, axis=0)
    scores_train = log_g_train - log_mix_train
    target_contributions = (
        _pooled_is_ratios(bank) * bank.base_weights[None, :])
    grid_rule = common_grid_raw_is_tail_rule(
        scores_train, target_contributions, alpha,
        minimum_rule=fit.training_rule)
    mixture_rejection_train = tail_rejection_probabilities(
        scores_train, fit.training_rule)
    grid_rejection_train = tail_rejection_probabilities(
        scores_train, grid_rule)
    grid_rejection_probabilities = (
        target_contributions @ grid_rejection_train)
    if np.max(grid_rejection_probabilities) > alpha + 2e-12:
        raise AssertionError("GKM Step-8 rule exceeds alpha on the null grid")
    importance_diagnostics = pooled_is_diagnostics(
        bank, mixture_rejection_train)
    importance_diagnostics.update(
        final_complementarity_residual=float(
            fit.complementarity_residual),
        final_complementarity_diagnostic_passed=bool(fit.converged))

    if verbose:
        print(f"    direct GKM common grid H={H}; pooled N0 observations="
              f"{bank.eigs.shape[0]:,}; full mixture support={H}")
        print(f"    Step 5 fixed update: mu0=-2, omega=2, O={n_iter}; "
              f"final residual={fit.complementarity_residual:.3e}")
        print(f"    ordinary-IS mass range="
              f"[{np.min(importance_diagnostics['raw_mass']):.4f}, "
              f"{np.max(importance_diagnostics['raw_mass']):.4f}]; "
              f"min ESS fraction="
              f"{np.min(importance_diagnostics['kish_ess_fraction']):.3f}",
              flush=True)

    omegas_null = np.asarray(
        [list(row) + [0.0] for row in bank.grid], dtype=float)
    omegas_score = np.vstack([omegas_null, omega_alt])
    rng_power = np.random.default_rng(seed)
    M_alt = build_M(kappas_alt, k_eff)
    eigs_power = eigenvalues_descending(
        simulate_Xi(M_alt, n_sim_power, rng_power))
    log_power, phase_diagnostics["alternative_power"] = \
        log_eigval_density_partial(
            eigs_power, omegas_score, k_eff / 2.0,
            M_trunc=M_trunc, chunk_size=100,
            progress_label="gkm-power", n_workers=n_workers,
            M_step=M_step, M_max=M_max, mhg_tol=mhg_tol,
            return_diagnostics=True)
    scores_power = _score_from_log_densities(
        log_power, log_weights=fit.log_weights)

    mixture_rejection = tail_rejection_probabilities(
        scores_power, fit.training_rule)
    grid_rejection = tail_rejection_probabilities(scores_power, grid_rule)
    mixture_power = float(np.mean(mixture_rejection))
    bound = float(np.mean(grid_rejection))
    mixture_power_se = float(
        np.std(mixture_rejection, ddof=1) / np.sqrt(n_sim_power))
    bound_se = float(np.std(grid_rejection, ddof=1) / np.sqrt(n_sim_power))
    epsilon = mixture_power - bound
    if bound > mixture_power + 1e-12 or epsilon < -1e-12:
        raise AssertionError(
            "GKM Step-9 grid-adjusted power exceeds Step-7 mixture power")
    if not np.isclose(
            fit.training_rule.empirical_size, alpha, rtol=0.0, atol=5e-13):
        raise AssertionError("GKM Step-6 mixture calibration missed alpha")

    # The shared bank's diagnostic is included for convenient max-order
    # inspection.  Runtime accounting in main records its pairs only once.
    all_phases = dict(phase_diagnostics)
    all_phases["shared_null_bank"] = bank.mhg_diagnostics
    mhg_diagnostics = _combine_phase_mhg_diagnostics(all_phases)
    result = GKMDirectResult(
        bound=bound, bound_se=bound_se,
        mixture_power=mixture_power,
        mixture_power_se=mixture_power_se,
        epsilon_grid=epsilon,
        weights=fit.weights,
        log_weights=fit.log_weights,
        fit_rejection_probabilities=fit.rejection_probabilities,
        grid_rejection_probabilities=grid_rejection_probabilities,
        fit_iterations=fit.iterations,
        mixture_rule=fit.training_rule,
        grid_rule=grid_rule,
        importance_diagnostics=importance_diagnostics,
        mhg_diagnostics=mhg_diagnostics)
    if verbose:
        print(f"    GKM Step 7 bar(pi)={mixture_power:.5f} "
              f"(MC SE {mixture_power_se:.5f}); Step 9 tilde(pi)="
              f"{bound:.5f} (MC SE {bound_se:.5f}); "
              f"epsilon={result.epsilon_grid:.5f}", flush=True)
    return result


# ============================================================
# Driver: common-grid production curve (9 beta points by default)
# ============================================================

class _Tee:
    """Duplicate writes to several streams (e.g. console + a log file), so the
    run log lives in the output folder regardless of how the job is launched."""
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self._streams:
            s.flush()


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (TailRule,)):
        return asdict(value)
    return value


def _atomic_savez(path, **arrays):
    temporary = path + ".tmp.npz"
    np.savez(temporary, **arrays)
    os.replace(temporary, path)


def _format_duration(seconds):
    seconds = float(seconds)
    if seconds < 120.0:
        return f"{seconds:.0f} s"
    if seconds < 7200.0:
        return f"{seconds / 60.0:.1f} min"
    if seconds < 172800.0:
        return f"{seconds / 3600.0:.1f} h"
    return f"{seconds / 86400.0:.1f} d"


def _gkm_budget_diagnostics(alpha, common_grid_size, n_nonnull, budget):
    """Exact logical-pair accounting for the direct GKM D.3.2 path."""
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must lie in (0,1)")
    H = _validated_integer("common_grid_size", common_grid_size)
    B = _validated_integer("n_nonnull", n_nonnull)
    n0 = _validated_integer("n_fit", budget["n_fit"], minimum=2)
    n1 = _validated_integer("n_power", budget["n_power"], minimum=2)
    n_iter = _validated_integer("n_iter", budget["n_iter"])
    phase_pairs = dict(
        shared_null_bank=int(H * H * n0),
        beta_training_alternative=int(B * H * n0),
        beta_power=int(B * (H + 1) * n1),
    )
    return dict(
        common_grid_size=H, n_nonnull=B, n_fit=n0, n_power=n1,
        n_iter=n_iter, pooled_observations=int(H * n0),
        null_tail_se_reference=float(
            math.sqrt(alpha * (1.0 - alpha) / n0)),
        power_se_at_half=float(0.5 / math.sqrt(n1)),
        phase_pairs=phase_pairs,
        total_pairs=int(sum(phase_pairs.values())))


def _print_gkm_budget_diagnostics(result, alpha):
    print("Direct GKM D.3.2 simulation preflight:")
    print(f"  common null grid H={result['common_grid_size']}; "
          f"N0=n_fit={result['n_fit']:,} per null "
          f"({result['pooled_observations']:,} pooled observations)")
    print(f"  Step 5 uses mu0=-2, omega=2, and exactly "
          f"O=n_iter={result['n_iter']} cached-table updates")
    print(f"  N1=n_power={result['n_power']:,} fresh alternative draws per "
          f"non-null beta; largest Bernoulli MC SE is about "
          f"{100 * result['power_se_at_half']:.3f} pp")
    print(f"  nominal direct-null tail SE reference at alpha={alpha:g} is "
          f"{100 * result['null_tail_se_reference']:.3f} pp; actual ordinary-"
          "IS precision depends on the saved mass and ESS diagnostics")
    total = result["total_pairs"]
    phase_text = ", ".join(
        f"{name}={count:,} ({100.0 * count / total:.1f}%)"
        for name, count in result["phase_pairs"].items())
    print(f"  density-pair cost by phase: {phase_text}")
    print("  The null table is computed once and reused across beta values; "
          "Step 6 and Step 8 reuse it. Adaptive-M retries add raw C "
          "evaluations but not logical pairs.\n", flush=True)




def _benchmark_adaptive_mhg(ncp_table, betas, total_logical_pairs, k_eff,
                            M_start, M_step, M_max, mhg_tol, n_workers,
                            n_samples, fit_grids,
                            benchmark_seed=MHG_BENCHMARK_SEED):
    """Time a small deterministic batch of representative real p=4 pairs.

    The benchmark owns its ``Generator`` and exits before a production run is
    initialized, so it neither consumes production random draws nor writes an
    artifact.  Roughly two thirds of the samples span the common null grid and
    one third are genuine Xi'Xi draws from the median-trace non-null
    alternative.  This approximates the fresh production pair mix and exposes
    costly boundary/stress draws that the former alternative-only benchmark
    missed.  Every observation is evaluated against every fitted-null density
    plus the alternative, matching production row batching.  Alternative trace
    quantiles from a larger candidate pool reduce small-batch noise.
    """
    n_samples = _validated_integer("benchmark_samples", n_samples)
    n_workers = _validated_integer("n_workers", n_workers)
    k_eff = _validated_integer("k_eff", k_eff)
    total_logical_pairs = _validated_integer(
        "total_logical_pairs", total_logical_pairs)
    if n_samples > MHG_MAX_BENCHMARK_SAMPLES:
        raise ValueError(
            f"benchmark_samples must be <= {MHG_MAX_BENCHMARK_SAMPLES}")

    betas = np.asarray(betas, dtype=float)
    ncp_table = np.asarray(ncp_table, dtype=float)
    if (betas.ndim != 1 or ncp_table.ndim != 2
            or ncp_table.shape != (betas.size, 4)
            or not np.all(np.isfinite(betas))
            or not np.all(np.isfinite(ncp_table))
            or np.any(ncp_table < 0.0)):
        raise ValueError("benchmark requires aligned finite nonnegative p=4 NCPs")
    nonnull_indices = np.flatnonzero(betas != 0.0)
    if nonnull_indices.size == 0:
        raise ValueError("benchmark beta grid has no non-null alternative")

    ncp_traces = ncp_table[nonnull_indices].sum(axis=1)
    trace_order = np.argsort(ncp_traces, kind="stable")
    representative_index = int(
        nonnull_indices[trace_order[len(trace_order) // 2]])
    representative_omega = ncp_table[representative_index].copy()
    if len(fit_grids) != betas.size:
        raise ValueError("fit_grids must align with the benchmark beta grid")
    fit_grid = _validated_null_grid(
        fit_grids[representative_index], 3,
        "benchmark fit null grid")
    benchmark_omegas = np.vstack([
        np.asarray([list(row) + [0.0] for row in fit_grid], dtype=float),
        representative_omega[None],
    ])
    benchmark_scope = "all_fitted_null_rows_plus_alternative"

    # This generator is deliberately unrelated to all four production phase
    # SeedSequences.  The constant seed makes target-machine comparisons
    # repeatable even when the requested production seed changes.
    rng = np.random.default_rng(int(benchmark_seed))
    n_null_samples = min(
        n_samples, max(1, int(round(2.0 * n_samples / 3.0))))
    n_alt_samples = n_samples - n_null_samples
    grid_locations = np.rint(np.linspace(
        0, len(fit_grid) - 1, n_null_samples)).astype(int)
    null_samples = []
    for location in grid_locations:
        null_omega = list(fit_grid[int(location)]) + [0.0]
        null_samples.append(eigenvalues_descending(simulate_Xi(
            build_M(null_omega, k_eff), 1, rng))[0])
    if n_alt_samples:
        pool_size = max(32, 4 * n_alt_samples)
        candidates = eigenvalues_descending(simulate_Xi(
            build_M(representative_omega, k_eff), pool_size, rng))
        ordered = np.argsort(candidates.sum(axis=1), kind="stable")
        quantiles = (np.linspace(0.25, 0.75, n_alt_samples)
                     if n_alt_samples > 1 else np.array([0.5]))
        locations = np.rint(quantiles * (pool_size - 1)).astype(int)
        samples = np.vstack([
            np.asarray(null_samples), candidates[ordered[locations]]])
    else:
        samples = np.asarray(null_samples)

    benchmark_pairs = len(benchmark_omegas) * n_samples
    # chunked_mhg_batch parallelizes over samples; all Omega rows for one
    # sample stay in the same worker, matching the production batching path.
    benchmark_workers = min(n_workers, n_samples)
    benchmark_chunk_size = max(
        1, int(n_samples // benchmark_workers))
    started = time.perf_counter()
    values, diagnostics = chunked_mhg_batch(
        k_eff / 2.0, benchmark_omegas, samples,
        M_trunc=M_start, chunk_size=benchmark_chunk_size,
        progress_label="benchmark",
        progress_interval_sec=float("inf"), n_workers=benchmark_workers,
        M_step=M_step, M_max=M_max, mhg_tol=mhg_tol,
        return_diagnostics=True)
    elapsed = time.perf_counter() - started
    if (elapsed <= 0.0 or values.shape != (len(benchmark_omegas), n_samples)
            or not np.all(np.isfinite(values)) or np.any(values < 1.0 - 1e-12)
            or diagnostics["pairs"] != benchmark_pairs):
        raise RuntimeError("invalid result from target-machine mhg benchmark")

    pairs_per_second = benchmark_pairs / elapsed
    measured_extrapolation = total_logical_pairs / pairs_per_second
    optimistic_configured_rate = (
        pairs_per_second * n_workers / benchmark_workers)
    return dict(
        benchmark_seed=int(benchmark_seed),
        machine=platform.platform(),
        processor=platform.processor() or platform.machine(),
        logical_cpu_count=int(os.cpu_count() or 1),
        representative_beta=float(betas[representative_index]),
        representative_omega=representative_omega,
        benchmark_scope=benchmark_scope,
        null_samples=int(n_null_samples),
        alternative_samples=int(n_alt_samples),
        omega_rows=int(len(benchmark_omegas)),
        fit_null_rows=int(len(benchmark_omegas) - 1),
        sample_trace_min=float(np.min(samples.sum(axis=1))),
        sample_trace_max=float(np.max(samples.sum(axis=1))),
        samples=int(n_samples), pairs=int(benchmark_pairs),
        workers=int(benchmark_workers),
        configured_workers=int(n_workers), elapsed_seconds=float(elapsed),
        pairs_per_second=float(pairs_per_second),
        measured_extrapolated_seconds=float(measured_extrapolation),
        optimistic_configured_seconds=float(
            total_logical_pairs / optimistic_configured_rate),
        raw_evaluations=int(diagnostics["raw_evaluations"]),
        order_counts=dict(diagnostics["order_counts"]),
        max_order=int(diagnostics["max_order"]),
        max_remainder_ratio=float(diagnostics["max_remainder_ratio"]),
    )


def _print_mhg_benchmark(result):
    print("Target-machine adaptive p=4 benchmark:")
    print(f"  machine: {result['machine']}; processor: "
          f"{result['processor']}; logical CPUs: "
          f"{result['logical_cpu_count']}")
    print(f"  deterministic benchmark seed: {result['benchmark_seed']}")
    print(f"  representative beta={result['representative_beta']:+.3f}, "
          f"Omega={np.round(result['representative_omega'], 4).tolist()}")
    print(f"  sampled Xi'Xi trace range: {result['sample_trace_min']:.2f}"
          f"--{result['sample_trace_max']:.2f}")
    print(f"  sample mix: {result['null_samples']} common-grid null + "
          f"{result['alternative_samples']} representative alternative")
    print(f"  density-row scope: {result['benchmark_scope']} "
          f"({result['fit_null_rows']} fitted null + 1 alternative)")
    print(f"  {result['samples']} samples x {result['omega_rows']} Omega rows = "
          f"{result['pairs']} density pairs on {result['workers']} worker(s) "
          f"in {result['elapsed_seconds']:.2f} s: "
          f"{result['pairs_per_second']:.4f} measured pairs/s")
    print(f"  raw C evaluations={result['raw_evaluations']} "
          f"({result['raw_evaluations'] / result['pairs']:.2f}/pair); "
          "selected orders="
          f"{result['order_counts']}; max remainder ratio="
          f"{result['max_remainder_ratio']:.2e}")
    print("  current-run extrapolation at measured concurrency: "
          f"{_format_duration(result['measured_extrapolated_seconds'])}")
    if result['workers'] < result['configured_workers']:
        print(f"  optimistic perfect-{result['configured_workers']}-way "
              "extrapolation from this small batch: "
              f"{_format_duration(result['optimistic_configured_seconds'])}")
        print("  Increase --benchmark-samples to at least --workers to measure "
              "all configured workers directly.")
    print("  Timing includes worker-pool startup, so tiny/easy batches are "
          "conservative and noisy.")
    print("  Benchmark used a private deterministic RNG stream and wrote no "
          "artifact. Rare high-order draws can still be materially slower.\n",
          flush=True)






def main():
    """Run the direct m_W=3 extension of GKM Supplement D.3.2."""
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Direct GKM D.3.2 / EMW eigenvalue power-bound calculation for "
            "m_W=3, with per-density adaptive 0F1 truncation."))
    parser.add_argument(
        "--version", required=True, choices=list(VERSION_LABELS),
        help="configuration label (" + ", ".join(VERSION_LABELS) + ")")
    parser.add_argument(
        "--profile", choices=("production", "reference"),
        default="production",
        help=("simulation scale; reference uses GKM's N0=10,000, "
              "N1=100,000, O=600"))
    parser.add_argument(
        "--force", action="store_true",
        help="replace an incompatible completed direct-GKM artifact")
    preflight = parser.add_mutually_exclusive_group()
    preflight.add_argument(
        "--preflight-only", action="store_true",
        help="print exact direct-GKM density-pair counts and exit")
    preflight.add_argument(
        "--benchmark-preflight", action="store_true",
        help="benchmark a representative adaptive p=4 density batch and exit")
    parser.add_argument("--benchmark-samples", type=int, default=None)
    parser.add_argument(
        "--acknowledge-expensive", action="store_true",
        help="required before starting a production/reference calculation")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--m-start", type=int, default=None,
                        help="minimum 0F1 degree; each density pair adapts upward")
    parser.add_argument("--m-step", type=int, default=MHG_DEFAULT_STEP)
    parser.add_argument("--m-max", type=int, default=MHG_DEFAULT_MAX)
    parser.add_argument("--mhg-rtol", type=float, default=MHG_CONV_TOL)
    parser.add_argument(
        "--n-fit", type=int, default=None,
        help="GKM N0 draws per common-grid null")
    parser.add_argument(
        "--n-power", type=int, default=None,
        help="GKM N1 fresh alternative draws per non-null beta")
    parser.add_argument(
        "--n-iter", type=int, default=None,
        help="GKM O fixed weight updates (paper value: 600)")
    parser.add_argument(
        "--grid-shapes", type=int, default=9,
        help="common three-dimensional nuisance-shape directions")
    parser.add_argument(
        "--grid-strengths", type=int, default=7,
        help="log-spaced strengths per direction")
    parser.add_argument(
        "--grid-max-strength", type=float, default=100.0)
    parser.add_argument(
        "--beta-count", type=int, default=9,
        help="odd number of saved beta values on [-2,2] (default: 9)")
    args = parser.parse_args()

    k = 7
    n = 250
    alpha = 0.05
    key = VERSION_LABELS[args.version]
    cfg = ALLOWED_CONFIGS[key]
    kappas = np.asarray(key, dtype=float)
    standard_points = cfg["standard"]
    M_start = cfg["M_start"] if args.m_start is None else args.m_start
    profiles = {
        "production": dict(n_fit=2000, n_power=50000, n_iter=600),
        "reference": dict(n_fit=10000, n_power=100000, n_iter=600),
    }
    budget = profiles[args.profile].copy()
    for name, value in (
            ("n_fit", args.n_fit), ("n_power", args.n_power),
            ("n_iter", args.n_iter)):
        if value is not None:
            budget[name] = value
    for name in ("n_fit", "n_power"):
        if (not isinstance(budget[name], (int, np.integer))
                or isinstance(budget[name], (bool, np.bool_))
                or int(budget[name]) < 2):
            parser.error(f"--{name.replace('_', '-')} must be an integer >= 2")
    if (not isinstance(budget["n_iter"], (int, np.integer))
            or isinstance(budget["n_iter"], (bool, np.bool_))
            or int(budget["n_iter"]) < 1):
        parser.error("--n-iter must be a positive integer")
    if args.beta_count < 3 or args.beta_count % 2 != 1:
        parser.error("--beta-count must be an odd integer >= 3 so beta=0 is saved")
    if args.grid_shapes < 1 or args.grid_strengths < 1:
        parser.error("--grid-shapes and --grid-strengths must be positive")
    if (not np.isfinite(args.grid_max_strength)
            or args.grid_max_strength <= 0.1):
        parser.error("--grid-max-strength must be finite and greater than 0.1")
    if args.seed < 0:
        parser.error("--seed must be nonnegative")
    if not (1 <= M_start <= args.m_max) or args.m_step < 1:
        parser.error("require 1 <= --m-start <= --m-max and --m-step >= 1")
    if not np.isfinite(args.mhg_rtol) or not (1e-13 <= args.mhg_rtol < 1.0):
        parser.error("--mhg-rtol must be finite and lie in [1e-13, 1)")
    if args.benchmark_samples is not None:
        if not args.benchmark_preflight:
            parser.error("--benchmark-samples requires --benchmark-preflight")
        if not (1 <= args.benchmark_samples <= MHG_MAX_BENCHMARK_SAMPLES):
            parser.error(
                "--benchmark-samples must lie in "
                f"[1, {MHG_MAX_BENCHMARK_SAMPLES}]")
    n_workers = (args.workers if args.workers is not None
                 else min(os.cpu_count() or 1, 16))
    if n_workers < 1:
        parser.error("--workers must be positive")

    betas = np.linspace(-2.0, 2.0, int(args.beta_count))
    ncp_table = np.asarray([
        np.maximum(asymptotic_ncp_eigenvalues(b, kappas, k, n), 0.0)
        for b in betas])
    nonnull = betas != 0.0
    # The nuisance grid is part of the statistical design, not a side effect
    # of how densely the displayed beta curve is sampled.  Derive its two
    # path-dependent shape directions from one fixed dense design path so
    # changing --beta-count changes curve resolution only.
    grid_design_betas = np.linspace(
        -2.0, 2.0, GRID_DESIGN_BETA_COUNT)
    grid_design_nuisance = np.asarray([
        np.maximum(
            asymptotic_ncp_eigenvalues(b, kappas, k, n), 0.0)[:-1]
        for b in grid_design_betas])
    common_grid = common_null_grid_3d(
        grid_design_nuisance, kappas, standard_points=standard_points,
        n_shapes=args.grid_shapes, n_strengths=args.grid_strengths,
        max_strength=args.grid_max_strength)
    H = len(common_grid)
    grid_anchor_count = H - (1 + args.grid_shapes * args.grid_strengths)
    if not (0 <= grid_anchor_count <= len(standard_points)):
        raise AssertionError("unexpected common-grid anchor count")
    B = int(np.count_nonzero(nonnull))
    try:
        budget_diagnostics = _gkm_budget_diagnostics(
            alpha, H, B, budget)
    except ValueError as exc:
        parser.error(f"invalid direct-GKM simulation budget: {exc}")
    total_logical_pairs = budget_diagnostics["total_pairs"]
    _print_gkm_budget_diagnostics(budget_diagnostics, alpha)

    representative_pair_seconds = {
        (35, 25, 15): 0.123,
        (100, 30, 15): 0.827,
        (100, 95, 90): 31.1,
    }[key]
    serial_estimate = total_logical_pairs * representative_pair_seconds
    parallel_lower = serial_estimate / n_workers
    print("Preflight computational scale:")
    print(f"  logical density pairs: {total_logical_pairs:,} over {B} "
          "non-null betas")
    print(f"  prior developer-machine representative pair: "
          f"~{representative_pair_seconds:g} s; indicative serial "
          f"extrapolation: {_format_duration(serial_estimate)}")
    print(f"  optimistic perfect-{n_workers}-way lower bound: "
          f"{_format_duration(parallel_lower)}")
    if key == (100, 95, 90):
        print("  WARNING: strong-grid pairs can be materially slower than "
              "this representative extrapolation.")
    print("  Adaptive retries add raw C evaluations and multiprocessing "
          "scaling is sublinear. Use --benchmark-preflight on the target "
          "machine.\n", flush=True)
    if args.preflight_only:
        print("Preflight only; no simulation or artifact write was performed.")
        return
    if args.benchmark_preflight:
        sample_count = (args.benchmark_samples
                        if args.benchmark_samples is not None
                        else MHG_DEFAULT_BENCHMARK_SAMPLES[key])
        result = _benchmark_adaptive_mhg(
            ncp_table, betas, total_logical_pairs, k,
            M_start, args.m_step, args.m_max, args.mhg_rtol,
            n_workers, sample_count,
            fit_grids=[common_grid for _ in range(len(betas))])
        _print_mhg_benchmark(result)
        return
    if not args.acknowledge_expensive:
        parser.error(
            "production/reference runs require --acknowledge-expensive; "
            "inspect the preflight estimate first")

    out_dir = os.path.join(args.version, "gkm_direct")
    out_npz = os.path.join(out_dir, f"gkm_eigval_{args.version}.npz")
    partial_npz = os.path.join(
        out_dir, f"gkm_eigval_{args.version}.partial.npz")
    source_path = os.path.abspath(__file__)
    lib_name = "libmhg.dylib" if sys.platform == "darwin" else "libmhg.so"
    provenance = dict(
        schema_version=RESULT_SCHEMA_VERSION,
        algorithm=ALGORITHM_VERSION,
        producer=os.path.basename(source_path),
        calibration_method=CALIBRATION_METHOD,
        source_sha256=_sha256_file(source_path),
        mhg_core_sha256=_sha256_file(os.path.join(MHG_DIR, "mhg_core.c")),
        mhg_library_sha256=_sha256_file(os.path.join(MHG_DIR, lib_name)),
        mhg_build_source_sha256=_verify_mhg_build_provenance(
            MHG_DIR, lib_name),
        python_version=sys.version,
        numpy_version=np.__version__, scipy_version=scipy.__version__,
        platform=platform.platform())
    bank_seed = int(np.random.SeedSequence(
        [args.seed, 0x474B4D34]).generate_state(1, dtype=np.uint32)[0])
    run_settings = dict(
        version_label=args.version, kappas=kappas.tolist(), k=k, n=n,
        alpha=alpha, profile=args.profile, seed=args.seed,
        M_start=M_start, M_step=args.m_step, M_max=args.m_max,
        mhg_rtol=args.mhg_rtol, beta_count=int(args.beta_count),
        fit_grid_strategy=COMMON_GRID_METHOD,
        pooled_importance_method=POOLED_IS_METHOD,
        common_grid=common_grid,
        grid_design_beta_count=GRID_DESIGN_BETA_COUNT,
        grid_shapes=args.grid_shapes,
        grid_strengths=args.grid_strengths,
        grid_max_strength=args.grid_max_strength,
        grid_anchor_count=grid_anchor_count,
        bank_seed=bank_seed,
        n_fit=int(budget["n_fit"]), n_power=int(budget["n_power"]),
        n_iter=int(budget["n_iter"]),
        gkm_initial_mu=-2.0, gkm_step_size=2.0,
        **provenance)
    run_signature = hashlib.sha256(json.dumps(
        run_settings, sort_keys=True,
        separators=(",", ":")).encode()).hexdigest()

    if os.path.isfile(out_npz) and not args.force:
        try:
            with np.load(out_npz, allow_pickle=False) as existing:
                saved_signature = str(existing["run_signature"].item())
        except (OSError, KeyError, ValueError) as exc:
            raise RuntimeError(
                f"Existing artifact {out_npz} lacks trustworthy provenance "
                f"({exc}). Move it aside or use --force explicitly.") from exc
        if saved_signature == run_signature:
            print(f"Compatible result already present: {out_npz}\nNothing to do.")
            return
        raise RuntimeError(
            f"Existing artifact {out_npz} has different code/settings. "
            "Move it aside or use --force explicitly.")

    os.makedirs(out_dir, exist_ok=True)
    log_handle = open(
        os.path.join(out_dir, "bound_run.log"), "a", buffering=1)
    sys.stdout = _Tee(sys.stdout, log_handle)
    sys.stderr = _Tee(sys.stderr, log_handle)
    print(f"Logging to {os.path.join(out_dir, 'bound_run.log')}")
    verify_mhg()
    print(f"Version {args.version}: k={k}, kappas={kappas.tolist()}, "
          f"n={n}, alpha={alpha}")
    print(f"  direct GKM D.3.2: H={H}, N0={budget['n_fit']:,}, "
          f"N1={budget['n_power']:,}, O={budget['n_iter']}")
    print(f"  full H-point mixture retained; no compression, audit split, "
          "or confidence correction")
    print(f"  adaptive M: start={M_start}, step={args.m_step}, "
          f"max={args.m_max}, rtol={args.mhg_rtol:.1e}; "
          f"workers={n_workers}")
    print(f"  beta grid: {len(betas)} points in [{betas[0]:+.2f}, "
          f"{betas[-1]:+.2f}]; run signature={run_signature}\n", flush=True)

    bounds = np.full(len(betas), np.nan)
    bounds_se = np.full(len(betas), np.nan)
    mixture_power = np.full(len(betas), np.nan)
    mixture_power_se = np.full(len(betas), np.nan)
    epsilon_grid = np.full(len(betas), np.nan)
    fitted_weights = np.full((len(betas), H), np.nan)
    fitted_log_weights = np.full((len(betas), H), np.nan)
    fit_rejection_probabilities = np.full((len(betas), H), np.nan)
    grid_rejection_probabilities = np.full((len(betas), H), np.nan)
    fit_iterations = np.zeros(len(betas), dtype=int)
    max_m_used = np.zeros(len(betas), dtype=int)
    diagnostics_json = np.full(len(betas), "", dtype="<U100000")
    checkpoint_arrays = dict(
        bounds=bounds, bounds_se=bounds_se,
        mixture_power=mixture_power,
        mixture_power_se=mixture_power_se,
        epsilon_grid=epsilon_grid,
        fitted_weights=fitted_weights,
        fitted_log_weights=fitted_log_weights,
        fit_rejection_probabilities=fit_rejection_probabilities,
        grid_rejection_probabilities=grid_rejection_probabilities,
        fit_iterations=fit_iterations, max_m_used=max_m_used,
        diagnostics_json=diagnostics_json)
    checkpoint_metadata = dict(
        schema_version=np.array(RESULT_SCHEMA_VERSION),
        algorithm=np.array(ALGORITHM_VERSION),
        producer=np.array("alfd_eigval.py"),
        calibration_method=np.array(CALIBRATION_METHOD),
        bound_kind=np.array(BOUND_KIND),
        version_label=np.array(args.version),
        run_signature=np.array(run_signature),
        settings_json=np.array(json.dumps(run_settings, sort_keys=True)),
        kappas=kappas, k=np.array(k), n=np.array(n), alpha=np.array(alpha))

    if os.path.isfile(partial_npz) and not args.force:
        try:
            with np.load(partial_npz, allow_pickle=False) as checkpoint:
                if str(checkpoint["run_signature"].item()) != run_signature:
                    raise ValueError("run signature differs")
                for name, destination in checkpoint_arrays.items():
                    saved = checkpoint[name]
                    if saved.shape != destination.shape:
                        raise ValueError(f"shape mismatch for {name}")
                    destination[...] = saved
        except (OSError, KeyError, ValueError) as exc:
            raise RuntimeError(
                f"Cannot resume checkpoint {partial_npz}: {exc}. Move it "
                "aside or use --force to replace this run.") from exc
        print(f"  resumed {np.count_nonzero(np.isfinite(bounds))}/"
              f"{len(betas)} beta points from {partial_npz}")
    _atomic_savez(
        partial_npz, **checkpoint_metadata, betas=betas, ncp=ncp_table,
        **checkpoint_arrays)

    bank = build_or_load_pooled_is_bank(
        common_grid, k, budget["n_fit"], bank_seed,
        M_start=M_start, M_step=args.m_step, M_max=args.m_max,
        mhg_tol=args.mhg_rtol, n_workers=n_workers, role="gkm",
        cache_dir=out_dir, cache_metadata=provenance)

    start_total = time.time()
    beta_times = []
    for i, beta_value in enumerate(betas):
        if np.isfinite(bounds[i]):
            print(f"========== beta {i + 1}/{len(betas)} = "
                  f"{beta_value:+.2f}: loaded from checkpoint ==========",
                  flush=True)
            continue
        start_beta = time.time()
        print(f"========== beta {i + 1}/{len(betas)} = {beta_value:+.2f} "
              f"(total elapsed {(start_beta - start_total) / 60:.1f} min) "
              "==========", flush=True)
        ncp = ncp_table[i]
        exact_null = not nonnull[i]
        beta_seed = None
        if exact_null:
            print("  exact rank-deficient alternative: randomized power "
                  f"equals alpha={alpha}", flush=True)
            result = _exact_gkm_result(alpha, H)
        else:
            beta_seed = int(np.random.SeedSequence(
                [args.seed, 0x42455441, i]).generate_state(
                    1, dtype=np.uint32)[0])
            if beta_seed == bank_seed:
                beta_seed = (beta_seed + 1) % (2 ** 32)
            print("  NCP=[" + ", ".join(f"{x:.4f}" for x in ncp)
                  + f"]; full GKM support H={H}", flush=True)
            result = gkm_eigval_bound_from_pooled_bank(
                kappas_alt=ncp, bank=bank, k_eff=k, alpha=alpha,
                n_sim_power=budget["n_power"], n_iter=budget["n_iter"],
                seed=beta_seed, verbose=True, n_workers=n_workers,
                M_trunc=M_start, M_step=args.m_step, M_max=args.m_max,
                mhg_tol=args.mhg_rtol)

        bounds[i] = result.bound
        bounds_se[i] = result.bound_se
        mixture_power[i] = result.mixture_power
        mixture_power_se[i] = result.mixture_power_se
        epsilon_grid[i] = result.epsilon_grid
        fitted_weights[i] = result.weights
        fitted_log_weights[i] = result.log_weights
        fit_rejection_probabilities[i] = result.fit_rejection_probabilities
        grid_rejection_probabilities[i] = result.grid_rejection_probabilities
        fit_iterations[i] = result.fit_iterations
        max_m_used[i] = int(result.mhg_diagnostics["max_order"])
        diagnostic_record = json.dumps(_json_safe(dict(
            power_seed=beta_seed,
            mixture_rule=result.mixture_rule,
            grid_rule=result.grid_rule,
            importance=result.importance_diagnostics,
            mhg=result.mhg_diagnostics)), sort_keys=True)
        if len(diagnostic_record) > diagnostics_json.dtype.itemsize // 4:
            raise RuntimeError(
                "per-beta diagnostic JSON exceeds its checkpoint field")
        diagnostics_json[i] = diagnostic_record
        _atomic_savez(
            partial_npz, **checkpoint_metadata, betas=betas, ncp=ncp_table,
            **checkpoint_arrays)

        elapsed = time.time() - start_beta
        beta_times.append(elapsed)
        remaining = int(np.count_nonzero(~np.isfinite(bounds)))
        eta = float(np.mean(beta_times) * remaining) if beta_times else 0.0
        print(f"  -> GKM Figure-3 bound tilde(pi)={bounds[i]:.5f} "
              f"+/- {bounds_se[i]:.5f} MC SE; bar(pi)="
              f"{mixture_power[i]:.5f}; epsilon={epsilon_grid[i]:.5f}; "
              f"max M={max_m_used[i]}")
        print(f"     beta runtime={elapsed / 60:.1f} min; "
              f"ETA={eta / 60:.1f} min\n", flush=True)

    total_elapsed = time.time() - start_total
    print(f"\n=== Direct GKM total loop runtime: {total_elapsed / 60:.1f} min ===")
    for beta_value, bound, se, bar, epsilon in zip(
            betas, bounds, bounds_se, mixture_power, epsilon_grid):
        print(f"  beta={beta_value:+.2f}: tilde(pi)={bound:.5f} +/- "
              f"{se:.5f}; bar(pi)={bar:.5f}; epsilon={epsilon:.5f}")

    save_payload = dict(
        **checkpoint_metadata,
        density_accuracy_scope=np.array("adaptive_empirical_tail_criterion"),
        betas=betas, ncp=ncp_table,
        **checkpoint_arrays,
        common_null_grid=np.asarray(common_grid, dtype=float),
        common_grid_size=np.array(H),
        grid_shapes=np.array(args.grid_shapes),
        grid_strengths=np.array(args.grid_strengths),
        grid_max_strength=np.array(args.grid_max_strength),
        grid_anchor_count=np.array(grid_anchor_count),
        grid_design_beta_count=np.array(GRID_DESIGN_BETA_COUNT),
        bank_id=np.array(bank.bank_id),
        bank_content_signature=np.array(bank.content_signature),
        bank_mhg_diagnostics_json=np.array(json.dumps(
            _json_safe(bank.mhg_diagnostics), sort_keys=True)),
        M_start=np.array(M_start), M_step=np.array(args.m_step),
        M_max=np.array(args.m_max), mhg_rtol=np.array(args.mhg_rtol),
        seed=np.array(args.seed), n_fit=np.array(budget["n_fit"]),
        n_power=np.array(budget["n_power"]),
        n_iter=np.array(budget["n_iter"]))
    _atomic_savez(out_npz, **save_payload)
    if os.path.isfile(partial_npz):
        os.unlink(partial_npz)
    print(f"\nSaved {out_npz}", flush=True)


if __name__ == "__main__":
    # On macOS Python defaults to 'spawn'; calling freeze_support() is the
    # standard guard so the worker processes import this module cleanly
    # instead of re-executing main().
    import multiprocessing as mp
    mp.freeze_support()
    main()
