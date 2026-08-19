"""
Eigenvalue-density-based ALFD power bound for m_W = 3 (p = 4).

Uses the noncentral-Wishart joint density of (real) ordered eigenvalues of a
noncentral Wishart matrix. The matrix hypergeometric 0F1^(2)((1/2)k; (1/4)Ω, S)
is evaluated via Plamen Koev's mhg algorithm (Koev & Edelman 2006). We call a
standalone C port of Koev's mhg.c directly from Python (ctypes) — no MATLAB or
Octave required.

The calculation bounds tests measurable with respect to the ordered eigenvalue
vector (the invariant class studied in the reference application).  The code
implements EMW's mandatory post-weight mixture calibration, an independent GKM
finite-grid tightening check, Monte Carlo confidence limits, and per-density
adaptive hypergeometric truncation.

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
import gc
import time
import json
import math
import hashlib
import platform
from collections import Counter
from dataclasses import asdict, dataclass

import numpy as np
import scipy
import matplotlib.pyplot as plt
from scipy.linalg import inv, sqrtm
from scipy.special import logsumexp
from scipy.stats import beta as beta_distribution
from scipy.stats import chi2


# ============================================================
# Allowed kappa configurations (m_W = 3)
# Each provides a *starting* 0F1 order.  Every density evaluation now checks
# its coefficient tail and increases the order locally when needed, so this is
# no longer a numerical result that users have to tune by rerunning a curve.
# ============================================================

RESULT_SCHEMA_VERSION = 3
ALGORITHM_VERSION = "emw_eigval_gkm_is_adaptive_v3"
CALIBRATION_METHOD = "independent_mixture_quantile"
BOUND_KIND = "simultaneous_mc_confidence_upper_conditional_on_density_accuracy"
COMMON_GRID_METHOD = "strength_shape_3d_v1"
POOLED_IS_METHOD = "gkm_stratified_equal_null_mixture_v1"
CONFIDENCE_ALLOCATION_METHOD = "two_event_upper_separate_two_event_grid_v1"

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

def lean_null_grid(alt_nuisance, standard_points, n_perturb=4,
                  perturb_sd=1.5, seed=0):
    """
    Compact null grid: alt-nuisance + config-specific standard points + random
    perturbations of alt-nuisance. The perturbations improve tightness when the
    least-favorable support lies near the alternative nuisance values. A sparse
    support does not invalidate the separately calibrated EMW upper endpoint;
    it can only make that endpoint loose.

    standard_points : list of (k1, k2, k3) tuples appropriate to the kappa
                      configuration (so coverage is at the right scale).
    """
    alt_arr = np.asarray(alt_nuisance, dtype=float)
    alt = tuple(sorted(alt_arr, reverse=True))
    grid = [alt] + [tuple(float(x) for x in sp) for sp in standard_points]
    rng = np.random.default_rng(seed)
    for _ in range(n_perturb):
        eps = rng.normal(0.0, perturb_sd, size=len(alt_arr))
        pert = tuple(sorted([max(0.5, a + e) for a, e in zip(alt_arr, eps)],
                            reverse=True))
        grid.append(pert)
    # De-duplicate (round-to-4dp keys)
    seen, out = set(), []
    for g in grid:
        key = tuple(round(x, 4) for x in g)
        if key not in seen:
            seen.add(key)
            out.append(g)
    return out


def validation_null_grid(fit_grid, alt_nuisance, standard_points,
                         n_points=32, max_strength=None, seed=81723):
    """Build a deterministic, broader grid for the GKM size check.

    For ``m_W=3`` the null nuisance space is an unbounded ordered cone, so no
    finite grid alone proves global size control.  This grid covers the fit
    support, boundary faces, rays and log-spread interior points.  The result
    is explicitly reported as a *finite-grid* certificate.
    """
    dim = len(alt_nuisance)
    base = [tuple(float(x) for x in row) for row in fit_grid]
    base += [tuple(float(x) for x in row) for row in standard_points]
    if max_strength is None:
        largest = max([1.0] + [max(row) for row in base]
                      + [float(np.max(alt_nuisance))])
        max_strength = max(100.0, 2.0 * largest)
    max_strength = float(max_strength)

    target_points = max(int(n_points), len(fit_grid))
    candidates = list(base)
    candidates.append(tuple(0.0 for _ in range(dim)))
    levels = np.expm1(np.linspace(0.0, np.log1p(max_strength), 7))[1:]
    for level in levels:
        for rank in range(1, dim + 1):
            candidates.append(tuple([float(level)] * rank
                                    + [0.0] * (dim - rank)))

    rng = np.random.default_rng(seed)
    while len(candidates) < max(target_points * 2, target_points + 8):
        draw = np.expm1(rng.uniform(0.0, np.log1p(max_strength), size=dim))
        candidates.append(tuple(np.sort(draw)[::-1]))

    seen = set()
    result = []
    for row in candidates:
        ordered = tuple(sorted((max(0.0, float(x)) for x in row), reverse=True))
        key = tuple(round(x, 8) for x in ordered)
        if key not in seen:
            seen.add(key)
            result.append(ordered)
        if len(result) >= target_points:
            break
    return result


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
class ALFDBoundResult:
    upper_point: float
    upper_point_se: float
    upper_confidence: float
    lower_grid_point: float
    lower_grid_confidence: float
    epsilon_grid_point: float
    epsilon_grid_confidence: float
    confidence_level: float
    weights: np.ndarray
    fit_rejection_probabilities: np.ndarray
    fit_complementarity_residual: float
    fit_converged: bool
    fit_iterations: int
    point_rule: TailRule
    upper_confidence_rule: TailRule
    lower_grid_rule: TailRule
    lower_grid_confidence_rule: TailRule
    calibration_component_counts: np.ndarray
    validation_rejection_probabilities: np.ndarray
    invariant_benchmark_power: float
    invariant_benchmark_se: float
    invariant_benchmark_lower_confidence: float
    mhg_diagnostics: dict
    full_weights: np.ndarray = None
    active_indices: np.ndarray = None
    active_null_grid: np.ndarray = None
    discarded_weight_mass: float = 0.0
    gkm_point_upper: float = np.nan
    gkm_grid_lower: float = np.nan
    gkm_epsilon: float = np.nan
    importance_diagnostics: dict = None
    grid_bracket_confidence_level: float = np.nan


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


def calibrate_weighted_tail(scores, sample_weights, alpha,
                            method="weighted_empirical_exact"):
    """Calibrate ``1{S>c}+rho*1{S=c}`` to exact weighted empirical size."""
    scores = np.asarray(scores, dtype=float).ravel()
    weights = np.asarray(sample_weights, dtype=float).ravel()
    if scores.size == 0 or scores.shape != weights.shape:
        raise ValueError("scores and sample_weights must be nonempty and aligned")
    if not np.all(np.isfinite(scores)) or not np.all(np.isfinite(weights)):
        raise ValueError("calibration scores and weights must be finite")
    if np.any(weights < 0.0) or not (0.0 < alpha < 1.0):
        raise ValueError("weights must be nonnegative and 0 < alpha < 1")
    total_weight = float(weights.sum())
    if total_weight <= 0.0:
        raise ValueError("sample weights must have positive mass")
    weights = weights / total_weight
    target = float(alpha)

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
        if above <= target + 1e-15 and target <= above + tie_mass + 1e-15:
            rho = float(np.clip((target - above) / tie_mass, 0.0, 1.0))
            empirical = above + rho * tie_mass
            return TailRule(float(sorted_scores[i]), rho, float(empirical), method)
        above += tie_mass
        i = j
    raise RuntimeError("failed to locate weighted tail quantile")


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
                raise RuntimeError("zero-mass tie cannot attain raw alpha")
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
            or bank.role not in ("training", "audit")
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


def calibrate_empirical_tail(scores, alpha, method="empirical_exact"):
    scores = np.asarray(scores, dtype=float).ravel()
    if scores.size == 0:
        raise ValueError("calibration scores must be nonempty")
    return calibrate_weighted_tail(
        scores, np.full(scores.size, 1.0 / scores.size), alpha, method=method)


def _liberal_tail_rejection_count(n, alpha, delta):
    """Calibration observations rejected by the confidence-liberal rule.

    This is the smallest ``q`` for which
    ``P{Binomial(n, alpha) >= q} <= delta``.  Keeping the rank calculation in
    one helper ensures that preflight reports exactly the rule used later.
    """
    n = _validated_integer("n", n, minimum=2)
    if not (0.0 < alpha < 1.0) or not (0.0 < delta < 1.0):
        raise ValueError("alpha and delta must lie in (0,1)")
    r = np.arange(n, dtype=int)
    cdf_at_alpha = beta_distribution.cdf(alpha, r + 1, n - r)
    valid = np.flatnonzero(cdf_at_alpha <= delta)
    if valid.size == 0:
        raise ValueError(
            f"n_calibration={n} is too small for alpha={alpha} and "
            f"delta={delta}; increase n_calibration or relax the confidence "
            "level")
    return int(valid[0] + 1)


def _conservative_tail_rejection_count(n, alpha, delta):
    """Validation observations rejected by the confidence-conservative rule.

    This is the largest ``r`` for which
    ``P{Binomial(n, alpha) <= r} <= delta``.
    """
    n = _validated_integer("n", n, minimum=2)
    if not (0.0 < alpha < 1.0) or not (0.0 < delta < 1.0):
        raise ValueError("alpha and delta must lie in (0,1)")
    candidate = np.arange(n, dtype=int)
    cdf_at_alpha = beta_distribution.cdf(
        alpha, candidate + 1, n - candidate)
    valid = np.flatnonzero(cdf_at_alpha >= 1.0 - delta)
    if valid.size == 0:
        raise ValueError(
            f"n_validation={n} is too small for alpha={alpha} and "
            f"delta={delta}; increase n_validation or relax the confidence "
            "level")
    return int(valid[-1])


def confidence_liberal_tail_rule(scores, alpha, delta):
    """Distribution-free order-statistic rule with true tail >= alpha.

    Conditional on a fitted mixture and independent iid calibration scores,
    this deliberately chooses a slightly liberal LR cutoff.  With probability
    at least ``1-delta`` its population mixture size is at least ``alpha``;
    its alternative power therefore remains above the level-alpha NP envelope.
    """
    scores = np.asarray(scores, dtype=float).ravel()
    if not np.all(np.isfinite(scores)) or not (0.0 < alpha < 1.0):
        raise ValueError("scores must be finite and 0 < alpha < 1")
    scores = np.sort(scores)
    N = scores.size
    if N < 2 or not (0.0 < delta < 1.0):
        raise ValueError("need at least two scores and 0 < delta < 1")
    # q includes the selected order statistic because the cutoff is moved one
    # representable float downward.  Before that move, q-1 observations are
    # strictly above it.
    q = _liberal_tail_rejection_count(N, alpha, delta)
    n_above = q - 1
    index = N - n_above - 1
    # Lower by one representable float so every observation tied at the order
    # statistic is included; this is conservative if numerical ties occur.
    threshold = float(np.nextafter(scores[index], -np.inf))
    empirical = float(np.mean(scores > threshold))
    return TailRule(threshold, 0.0, empirical,
                    "order_statistic_mixture_size_at_least_alpha")


def confidence_conservative_tail_rule(scores, alpha, delta):
    """Order-statistic rule with true tail <= alpha with confidence 1-delta."""
    scores = np.asarray(scores, dtype=float).ravel()
    if not np.all(np.isfinite(scores)) or not (0.0 < alpha < 1.0):
        raise ValueError("scores must be finite and 0 < alpha < 1")
    scores = np.sort(scores)
    N = scores.size
    if N < 2 or not (0.0 < delta < 1.0):
        raise ValueError("need at least two scores and 0 < delta < 1")
    n_above = _conservative_tail_rejection_count(N, alpha, delta)
    index = N - n_above - 1
    # Raise by one representable float so ties are excluded conservatively.
    threshold = float(np.nextafter(scores[index], np.inf))
    empirical = float(np.mean(scores > threshold))
    return TailRule(threshold, 0.0, empirical,
                    "order_statistic_null_size_at_most_alpha")


def clopper_pearson(count, n, delta, side):
    """One-sided exact binomial confidence limit."""
    count, n = int(count), int(n)
    if not (0 <= count <= n) or not (0.0 < delta < 1.0):
        raise ValueError("invalid binomial confidence-limit arguments")
    if side == "upper":
        return 1.0 if count == n else float(
            beta_distribution.ppf(1.0 - delta, count + 1, n - count))
    if side == "lower":
        return 0.0 if count == 0 else float(
            beta_distribution.ppf(delta, count, n - count + 1))
    raise ValueError("side must be 'lower' or 'upper'")


def common_grid_tail_rule(score_rows, alpha, minimum_rule=None,
                          method="grid_empirical_size_at_most_alpha"):
    """Smallest common empirical rule controlling every supplied null row."""
    rows = np.asarray(score_rows, dtype=float)
    if (rows.ndim != 2 or rows.shape[0] == 0 or rows.shape[1] == 0
            or not np.all(np.isfinite(rows)) or not (0.0 < alpha < 1.0)):
        raise ValueError(
            "score_rows must be a nonempty finite matrix and 0 < alpha < 1")
    individual = [calibrate_empirical_tail(row, alpha) for row in rows]
    threshold = max(rule.threshold for rule in individual)
    if minimum_rule is not None:
        threshold = max(threshold, minimum_rule.threshold)

    allowed_rho = 1.0
    for row in rows:
        above = float(np.mean(row > threshold))
        tied = float(np.mean(row == threshold))
        if above > alpha + 1e-12:
            raise RuntimeError("common grid threshold failed empirical size control")
        if tied > 0.0:
            allowed_rho = min(allowed_rho, max(0.0, (alpha - above) / tied))
    if minimum_rule is not None and threshold == minimum_rule.threshold:
        allowed_rho = min(allowed_rho, minimum_rule.tie_probability)
    rule = TailRule(float(threshold), float(np.clip(allowed_rho, 0.0, 1.0)),
                    np.nan, method)
    max_size = max(float(np.mean(tail_rejection_probabilities(row, rule)))
                   for row in rows)
    return TailRule(rule.threshold, rule.tie_probability, max_size, rule.method)


def fit_emw_weights(log_f, log_g, alpha=0.05, n_iter=600,
                    initial_step=2.0, min_step=0.01,
                    active_weight_tol=1e-6, convergence_tol=None,
                    convergence_patience=10):
    """Fit finite-grid EMW weights, with sign-switch step damping.

    The update direction and initial factor are EMW equation (10).  Damping a
    coordinate after a sign switch prevents the deterministic empirical
    staircase from cycling forever.  The returned weights are always followed
    by a separate independent critical-value calibration.
    """
    log_f = np.asarray(log_f, dtype=float)
    log_g = np.asarray(log_g, dtype=float)
    if log_f.ndim != 3 or log_g.shape != log_f.shape[1:]:
        raise ValueError("expected log_f (G,G,N) and log_g (G,N)")
    G, G_samples, n_per_null = log_f.shape
    if G != G_samples or not np.all(np.isfinite(log_f)) or not np.all(np.isfinite(log_g)):
        raise ValueError("training density arrays are invalid")
    if convergence_tol is None:
        convergence_tol = max(2.0 / n_per_null, 5e-4)

    mu = np.full(G, -2.0)
    steps = np.full(G, float(initial_step))
    previous_sign = np.zeros(G)
    best = None
    stable = 0

    for iteration in range(1, int(n_iter) + 1):
        log_threshold = logsumexp(mu[:, None, None] + log_f, axis=0)
        rejection = (log_g > log_threshold).astype(float)
        rp = rejection.mean(axis=1)
        error = rp - alpha
        weights = _softmax(mu)
        active = weights > active_weight_tol
        active_residual = (float(np.max(np.abs(error[active])))
                           if np.any(active) else 0.0)
        slack_residual = float(np.max(np.maximum(error[~active], 0.0))) \
            if np.any(~active) else 0.0
        residual = max(active_residual, slack_residual)
        if best is None or residual < best[0]:
            best = (residual, mu.copy(), rp.copy(), iteration)

        if residual <= convergence_tol:
            stable += 1
            if stable >= convergence_patience:
                break
        else:
            stable = 0

        sign = np.sign(error)
        switched = (previous_sign != 0.0) & (sign != 0.0) & (sign != previous_sign)
        steps[switched] = np.maximum(min_step, 0.5 * steps[switched])
        mu = mu + steps * error
        previous_sign = sign

    _, best_mu, _, best_iteration = best
    weights = _softmax(best_mu)

    # Diagnose complementarity after the mandatory normalized-mixture
    # calibration (the raw pre-update RP from the old code was meaningless).
    log_mix = logsumexp(
        _log_probability_weights(weights)[:, None, None] + log_f, axis=0)
    scores = log_g - log_mix
    observation_weights = np.repeat(weights / n_per_null, n_per_null)
    training_rule = calibrate_weighted_tail(
        scores.reshape(-1), observation_weights, alpha,
        method="training_weighted_mixture_exact")
    rp = np.mean(tail_rejection_probabilities(scores, training_rule), axis=1)
    active = weights > active_weight_tol
    active_residual = float(np.max(np.abs(rp[active] - alpha))) \
        if np.any(active) else 0.0
    slack_residual = float(np.max(np.maximum(rp[~active] - alpha, 0.0))) \
        if np.any(~active) else 0.0
    residual = max(active_residual, slack_residual)
    return EMWFitResult(
        weights=weights, mu=best_mu, rejection_probabilities=rp,
        training_rule=training_rule, iterations=best_iteration,
        converged=bool(residual <= convergence_tol),
        complementarity_residual=float(residual))


def fit_emw_weights_is(bank, log_g, alpha=0.05, n_iter=600,
                       initial_step=2.0, min_step=0.01,
                       active_weight_tol=1e-6, convergence_tol=None,
                       convergence_patience=10):
    """Fit EMW weights with GKM's pooled ordinary importance sampler."""
    H, N = _validate_pooled_is_bank(bank)
    log_g = np.asarray(log_g, dtype=float).ravel()
    if log_g.shape != (N,) or not np.all(np.isfinite(log_g)):
        raise ValueError("log_g must be finite and aligned with the pooled bank")
    n_iter = _validated_integer("n_iter", n_iter)
    convergence_patience = _validated_integer(
        "convergence_patience", convergence_patience)
    numeric_parameters = np.asarray(
        [alpha, initial_step, min_step, active_weight_tol], dtype=float)
    if (not np.all(np.isfinite(numeric_parameters))
            or not (0.0 < alpha < 1.0)
            or initial_step <= 0.0 or min_step <= 0.0
            or min_step > initial_step
            or not (0.0 <= active_weight_tol < 1.0)):
        raise ValueError("invalid EMW fit parameters")
    ratios = _pooled_is_ratios(bank)
    max_contribution = float(np.max(ratios * bank.base_weights[None, :]))
    if convergence_tol is None:
        convergence_tol = max(2.0 * max_contribution, 5e-4)
    elif (not np.isfinite(convergence_tol)) or convergence_tol <= 0.0:
        raise ValueError("convergence_tol must be finite and positive")

    mu = np.full(H, -2.0)
    steps = np.full(H, float(initial_step))
    previous_sign = np.zeros(H)
    best = None
    stable = 0
    for iteration in range(1, n_iter + 1):
        log_threshold = logsumexp(mu[:, None] + bank.log_f, axis=0)
        rejection = (log_g > log_threshold).astype(float)
        rp = ratios @ (bank.base_weights * rejection)
        error = rp - alpha
        weights = _softmax(mu)
        active = weights > active_weight_tol
        active_residual = (float(np.max(np.abs(error[active])))
                           if np.any(active) else 0.0)
        slack_residual = (float(np.max(np.maximum(error[~active], 0.0)))
                          if np.any(~active) else 0.0)
        residual = max(active_residual, slack_residual)
        if best is None or residual < best[0]:
            best = (residual, mu.copy(), iteration)
        if residual <= convergence_tol:
            stable += 1
            if stable >= convergence_patience:
                break
        else:
            stable = 0
        sign = np.sign(error)
        switched = ((previous_sign != 0.0) & (sign != 0.0)
                    & (sign != previous_sign))
        steps[switched] = np.maximum(min_step, 0.5 * steps[switched])
        mu = mu + steps * error
        previous_sign = sign

    _, best_mu, best_iteration = best
    weights = _softmax(best_mu)
    log_mix = logsumexp(
        _log_probability_weights(weights)[:, None] + bank.log_f, axis=0)
    scores = log_g - log_mix
    mix_contributions = bank.base_weights * np.exp(log_mix - bank.log_q)
    training_rule = calibrate_raw_weighted_tail(
        scores, mix_contributions, alpha,
        method="gkm_training_mixture_ordinary_is")
    rejection = tail_rejection_probabilities(scores, training_rule)
    rp = ratios @ (bank.base_weights * rejection)
    active = weights > active_weight_tol
    active_residual = (float(np.max(np.abs(rp[active] - alpha)))
                       if np.any(active) else 0.0)
    slack_residual = (float(np.max(np.maximum(rp[~active] - alpha, 0.0)))
                      if np.any(~active) else 0.0)
    residual = max(active_residual, slack_residual)
    return EMWFitResult(
        weights=weights, mu=best_mu, rejection_probabilities=rp,
        training_rule=training_rule, iterations=best_iteration,
        converged=bool(residual <= convergence_tol),
        complementarity_residual=float(residual))


def compress_mixture(weights, max_active=None):
    """Deterministically retain the largest weights and renormalize exactly.

    Support selection uses training output only.  Every cutoff is recomputed
    later on independent data, so the compressed distribution is simply a new
    valid null mixture rather than an approximation used inside certification.
    """
    weights = np.asarray(weights, dtype=float).ravel()
    if (weights.size == 0 or not np.all(np.isfinite(weights))
            or np.any(weights < 0.0) or weights.sum() <= 0.0):
        raise ValueError("mixture weights must be finite and nonnegative")
    weights = weights / weights.sum()
    if max_active is None:
        keep_count = weights.size
    else:
        keep_count = min(_validated_integer("max_active", max_active), weights.size)
        if keep_count < 1:
            raise ValueError("max_active must be positive")
    order = np.argsort(-weights, kind="mergesort")
    active = np.sort(order[:keep_count])
    dropped = float(1.0 - weights[active].sum())
    retained = weights[active]
    retained = retained / retained.sum()
    return active, retained, max(0.0, dropped)


def _pooled_experiment_settings(grid, k_eff, M_start, M_step, M_max,
                                mhg_tol, metadata=None):
    """Canonical density-experiment identity shared by train/audit banks."""
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
    """Validate and canonicalize the numerical audit attached to a bank."""
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
    if (pairs != int(expected_pairs) or raw < pairs or max_order < 0
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
                                 n_workers=1, role="training",
                                 cache_dir=None, cache_metadata=None):
    """Build/cache one beta-invariant GKM stratified null bank."""
    grid = _validated_null_grid(grid, 3, "common pooled null grid")
    n_per_stratum = _validated_integer(
        "n_per_stratum", n_per_stratum, minimum=2)
    k_eff = _validated_integer("k_eff", k_eff)
    if role not in ("training", "audit"):
        raise ValueError("pooled bank role must be 'training' or 'audit'")
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

def _sample_mixture_eigenvalues(M_nulls, weights, n_sim, rng):
    labels = rng.choice(len(M_nulls), size=int(n_sim), p=weights)
    p = M_nulls[0].shape[1]
    eigs = np.empty((int(n_sim), p))
    for j, M_null in enumerate(M_nulls):
        locations = np.flatnonzero(labels == j)
        if locations.size:
            Xi = simulate_Xi(M_null, locations.size, rng)
            eigs[locations] = eigenvalues_descending(Xi)
    return eigs, labels


def _score_from_log_densities(log_densities, weights):
    G = len(weights)
    log_mix = logsumexp(
        _log_probability_weights(weights)[:, None] + log_densities[:G], axis=0)
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


def _exact_null_result(alpha, G, null_grid=None):
    rule = TailRule(0.0, float(alpha), float(alpha),
                    "exact_null_randomization")
    exact_weights = np.full(G, 1.0 / G)
    active_grid = (None if null_grid is None
                   else np.asarray(null_grid, dtype=float).copy())
    return ALFDBoundResult(
        upper_point=float(alpha), upper_point_se=0.0,
        upper_confidence=float(alpha), lower_grid_point=float(alpha),
        lower_grid_confidence=float(alpha), epsilon_grid_point=0.0,
        epsilon_grid_confidence=0.0, confidence_level=1.0,
        weights=exact_weights,
        fit_rejection_probabilities=np.full(G, alpha),
        fit_complementarity_residual=0.0, fit_converged=True,
        fit_iterations=0,
        point_rule=rule, upper_confidence_rule=rule,
        lower_grid_rule=rule, lower_grid_confidence_rule=rule,
        calibration_component_counts=np.zeros(G, dtype=int),
        validation_rejection_probabilities=np.full(G, alpha),
        invariant_benchmark_power=float(alpha), invariant_benchmark_se=0.0,
        invariant_benchmark_lower_confidence=float(alpha),
        mhg_diagnostics=dict(pairs=0, raw_evaluations=0, max_order=0,
                             max_remainder_ratio=0.0,
                             order_counts={}, phases={}),
        full_weights=exact_weights.copy(), active_indices=np.arange(G),
        active_null_grid=active_grid, discarded_weight_mass=0.0,
        gkm_point_upper=float(alpha), gkm_grid_lower=float(alpha),
        gkm_epsilon=0.0,
        importance_diagnostics=dict(exact_null=True),
        grid_bracket_confidence_level=1.0)


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


def _union_null_grids(first, second):
    """Stable exact-value union used to retain every fitted support point."""
    seen = set()
    result = []
    for row in list(first) + list(second):
        key = tuple(row)
        if key not in seen:
            seen.add(key)
            result.append(key)
    return result


def alfd_eigval_bound(kappas_alt, grid_kappas_null, k_eff,
                     alpha=0.05, n_sim=1000, n_sim_power=50000,
                     n_iter=600, M_trunc=20, seed=42, verbose=True,
                     n_workers=1, n_sim_calibration=20000,
                     n_sim_validation=2000,
                     validation_grid_kappas_null=None,
                     confidence_delta=0.01,
                     M_step=MHG_DEFAULT_STEP, M_max=MHG_DEFAULT_MAX,
                     mhg_tol=MHG_CONV_TOL, return_result=False,
                     alternative_is_exact_null=False):
    """Compute an EMW upper bound and a GKM finite-grid epsilon bracket.

    The critical distinction from the previous implementation is that the EMW
    weights are normalized and then calibrated on a fresh iid sample from the
    fitted null mixture.  ``upper_point`` is the paper-style Monte Carlo point
    estimate.  ``upper_confidence`` additionally uses a liberal order-statistic
    cutoff and a one-sided binomial power limit, making it a confidence-valid
    upper bound conditional on the density accuracy.

    ``lower_grid_*`` and ``epsilon_grid_*`` implement GKM's tightened critical
    value on an independent finite null grid.  They are not advertised as
    global certificates for the unbounded three-dimensional null cone.

    The default scalar return is ``upper_confidence`` so callers cannot mistake
    the paper-style finite-Monte-Carlo point estimate for a guaranteed endpoint.
    Set ``return_result=True`` to retain both endpoints and all diagnostics.
    """
    kappas_alt = np.asarray(kappas_alt, dtype=float)
    if kappas_alt.ndim != 1 or kappas_alt.size == 0:
        raise ValueError("alternative NCP eigenvalues must be a nonempty vector")
    p = kappas_alt.size
    grid_kappas_null = _validated_null_grid(
        grid_kappas_null, p - 1, "fit null grid")
    G = len(grid_kappas_null)
    k_eff = _validated_integer("k_eff", k_eff)
    if G == 0 or k_eff < p:
        raise ValueError("need a nonempty null grid and k_eff >= p")
    if (not np.all(np.isfinite(kappas_alt)) or np.any(kappas_alt < 0.0)
            or np.any(np.diff(kappas_alt) > 1e-10)):
        raise ValueError("alternative NCP eigenvalues must be finite, nonnegative and ordered")
    if not (0.0 < alpha < 1.0) or not (0.0 < confidence_delta < 1.0):
        raise ValueError("alpha and confidence_delta must lie in (0,1)")
    if not isinstance(alternative_is_exact_null, (bool, np.bool_)):
        raise ValueError("alternative_is_exact_null must be boolean")

    # Rank deficiency must be known from the model, not inferred by thresholding
    # a small positive eigenvalue.  Otherwise a genuine nearby alternative could
    # be incorrectly replaced by alpha, which is not an upper bound there.
    if alternative_is_exact_null:
        numerical_zero_tol = (100.0 * np.finfo(float).eps
                              * max(1.0, kappas_alt[0]))
        if kappas_alt[-1] > numerical_zero_tol:
            raise ValueError(
                "alternative_is_exact_null=True but the smallest NCP is "
                "materially positive")
        result = _exact_null_result(alpha, G)
        return result if return_result else result.upper_confidence

    n_sim = _validated_integer("n_sim", n_sim, minimum=2)
    n_sim_calibration = _validated_integer(
        "n_sim_calibration", n_sim_calibration, minimum=2)
    n_sim_validation = _validated_integer(
        "n_sim_validation", n_sim_validation, minimum=2)
    n_sim_power = _validated_integer(
        "n_sim_power", n_sim_power, minimum=2)
    n_iter = _validated_integer("n_iter", n_iter)
    n_workers = _validated_integer("n_workers", n_workers)

    requested_validation_grid = (
        grid_kappas_null if validation_grid_kappas_null is None
        else _validated_null_grid(
            validation_grid_kappas_null, p - 1, "validation null grid"))
    # A finite-grid GKM bracket must constrain every support point used by the
    # fitted mixture.  Custom validation grids are therefore augmented rather
    # than trusted to contain the fit support themselves.
    validation_grid = _union_null_grids(
        grid_kappas_null, requested_validation_grid)
    V = len(validation_grid)

    seed_sequences = np.random.SeedSequence(seed).spawn(4)
    rng_fit, rng_cal, rng_validation, rng_power = [
        np.random.default_rng(child) for child in seed_sequences]
    c_0F1 = k_eff / 2.0
    M_alt = build_M(kappas_alt, k_eff)
    M_nulls = [build_M(list(row) + [0.0], k_eff)
               for row in grid_kappas_null]
    Omega_alt = kappas_alt
    Omegas_null = np.array([list(row) + [0.0]
                            for row in grid_kappas_null], dtype=float)
    Omegas_all = np.vstack([Omegas_null, Omega_alt[None]])
    phase_diagnostics = {}

    if verbose:
        print(f"    G_fit={G}, G_validate={V}, n_fit/null={n_sim}, "
              f"n_cal_mix={n_sim_calibration}, n_validate/null={n_sim_validation}, "
              f"n_power={n_sim_power}")
        print(f"    adaptive 0F1: start={M_trunc}, step={M_step}, "
              f"max={M_max}, tol={mhg_tol:.1e}", flush=True)

    # 1. Independent direct null samples for the EMW weight iteration.
    eigs_train = np.empty((G, int(n_sim), p))
    for j, M_null in enumerate(M_nulls):
        eigs_train[j] = eigenvalues_descending(
            simulate_Xi(M_null, int(n_sim), rng_fit))
    log_dens, phase_diagnostics["fit"] = log_eigval_density_partial(
        eigs_train.reshape(G * int(n_sim), p), Omegas_all, c_0F1,
        M_trunc=M_trunc, chunk_size=100, progress_label="fit",
        n_workers=n_workers, M_step=M_step, M_max=M_max,
        mhg_tol=mhg_tol, return_diagnostics=True)
    log_f = log_dens[:G].reshape(G, G, int(n_sim))
    log_g = log_dens[G].reshape(G, int(n_sim))
    fit = fit_emw_weights(log_f, log_g, alpha=alpha, n_iter=n_iter)
    del log_dens, log_f, log_g, eigs_train
    gc.collect()
    if verbose:
        print(f"    EMW weights: {np.round(fit.weights, 6).tolist()}")
        print(f"    calibrated training RP: "
              f"{np.round(fit.rejection_probabilities, 5).tolist()}  "
              f"complementarity residual={fit.complementarity_residual:.5f}  "
              f"converged={fit.converged}", flush=True)

    # 2. EMW/GKM Step 5/6: freeze lambda and independently solve the scalar
    # mixture critical value.  Sampling component labels produces iid draws
    # from f_lambda without needing N observations under every support point.
    eigs_cal, cal_labels = _sample_mixture_eigenvalues(
        M_nulls, fit.weights, int(n_sim_calibration), rng_cal)
    log_cal, phase_diagnostics["mixture_calibration"] = log_eigval_density_partial(
        eigs_cal, Omegas_all, c_0F1, M_trunc=M_trunc, chunk_size=100,
        progress_label="mixture-cal", n_workers=n_workers, M_step=M_step,
        M_max=M_max, mhg_tol=mhg_tol, return_diagnostics=True)
    scores_cal = _score_from_log_densities(log_cal, fit.weights)
    point_rule = calibrate_empirical_tail(
        scores_cal, alpha, method="independent_mixture_quantile")

    # The primary upper is a two-event confidence family: liberal mixture
    # calibration plus the alternative-power CP limit.  The optional finite-
    # grid lower is a separate two-event family.  This avoids making the upper
    # unnecessarily loose merely because the diagnostic lower is requested.
    upper_delta_each = confidence_delta / 2.0
    grid_delta_each = confidence_delta / 2.0
    upper_conf_rule = confidence_liberal_tail_rule(
        scores_cal, alpha, upper_delta_each)
    component_counts = np.bincount(cal_labels, minlength=G)
    del log_cal, eigs_cal, scores_cal
    gc.collect()

    # 3. GKM Step 8: independent null-grid validation and critical-value
    # tightening.  The LR still uses the fitted support mixture in its denominator.
    M_validation = [build_M(list(row) + [0.0], k_eff)
                    for row in validation_grid]
    eigs_validation = np.empty((V, int(n_sim_validation), p))
    for j, M_null in enumerate(M_validation):
        eigs_validation[j] = eigenvalues_descending(
            simulate_Xi(M_null, int(n_sim_validation), rng_validation))
    log_validation, phase_diagnostics["null_validation"] = \
        log_eigval_density_partial(
            eigs_validation.reshape(V * int(n_sim_validation), p),
            Omegas_all, c_0F1, M_trunc=M_trunc, chunk_size=100,
            progress_label="null-validation", n_workers=n_workers,
            M_step=M_step, M_max=M_max, mhg_tol=mhg_tol,
            return_diagnostics=True)
    scores_validation = _score_from_log_densities(
        log_validation, fit.weights).reshape(V, int(n_sim_validation))
    validation_rp = np.mean(
        tail_rejection_probabilities(scores_validation, point_rule), axis=1)
    lower_grid_rule = common_grid_tail_rule(
        scores_validation, alpha, minimum_rule=point_rule)

    per_null_delta = grid_delta_each / V
    conservative_rules = [confidence_conservative_tail_rule(
        row, alpha, per_null_delta) for row in scores_validation]
    conservative_threshold = max(
        [upper_conf_rule.threshold]
        + [rule.threshold for rule in conservative_rules])
    lower_grid_conf_rule = TailRule(
        float(conservative_threshold), 0.0,
        float(np.max(np.mean(scores_validation > conservative_threshold,
                             axis=1))),
        "simultaneous_grid_order_statistic_size_at_most_alpha")
    del log_validation, eigs_validation
    gc.collect()

    # 4. Fresh alternative sample for all reported power endpoints.
    eigs_power = eigenvalues_descending(
        simulate_Xi(M_alt, int(n_sim_power), rng_power))
    log_power, phase_diagnostics["alternative_power"] = log_eigval_density_partial(
        eigs_power, Omegas_all, c_0F1, M_trunc=M_trunc, chunk_size=100,
        progress_label="power", n_workers=n_workers, M_step=M_step,
        M_max=M_max, mhg_tol=mhg_tol, return_diagnostics=True)
    scores_power = _score_from_log_densities(log_power, fit.weights)

    benchmark_rejection = (eigs_power[:, -1]
                           > chi2.ppf(1.0 - alpha, df=k_eff - p + 1))
    benchmark_count = int(np.count_nonzero(benchmark_rejection))
    benchmark_power = float(benchmark_count / int(n_sim_power))
    benchmark_se = float(np.sqrt(
        benchmark_power * (1.0 - benchmark_power) / int(n_sim_power)))
    benchmark_lower_confidence = clopper_pearson(
        benchmark_count, int(n_sim_power), upper_delta_each, "lower")

    point_rejection = tail_rejection_probabilities(scores_power, point_rule)
    upper_point = float(np.mean(point_rejection))
    upper_point_se = float(np.std(point_rejection, ddof=1)
                           / np.sqrt(int(n_sim_power)))
    upper_conf_count = int(np.count_nonzero(
        scores_power > upper_conf_rule.threshold))
    upper_confidence = clopper_pearson(
        upper_conf_count, int(n_sim_power), upper_delta_each, "upper")

    lower_point = float(np.mean(
        tail_rejection_probabilities(scores_power, lower_grid_rule)))
    lower_conf_count = int(np.count_nonzero(
        scores_power > lower_grid_conf_rule.threshold))
    lower_confidence = clopper_pearson(
        lower_conf_count, int(n_sim_power), grid_delta_each, "lower")

    # The rules are nested by construction; these inequalities catch any future
    # regression in calibration/tie handling rather than clipping bad output.
    if lower_point > upper_point + 1e-12:
        raise AssertionError("GKM tightened rule has power above the EMW upper rule")
    if lower_confidence > upper_confidence + 1e-12:
        raise AssertionError("confidence bracket endpoints are reversed")
    if not np.isclose(point_rule.empirical_size, alpha, atol=5e-13):
        raise AssertionError("mixture point calibration did not attain alpha")
    if not np.isclose(fit.weights.sum(), 1.0, atol=1e-12) \
            or np.any(fit.weights < 0.0):
        raise AssertionError("invalid normalized EMW weights")
    if upper_confidence + 1e-12 < alpha:
        raise AssertionError("confidence upper bound fell below alpha")
    if upper_confidence + 1e-12 < benchmark_lower_confidence:
        raise AssertionError(
            "EMW confidence upper bound is below the same-experiment "
            "smallest-eigenvalue benchmark's one-sided confidence lower "
            "limit; increase simulation sizes")

    result = ALFDBoundResult(
        upper_point=upper_point, upper_point_se=upper_point_se,
        upper_confidence=upper_confidence,
        lower_grid_point=lower_point,
        lower_grid_confidence=lower_confidence,
        epsilon_grid_point=upper_point - lower_point,
        epsilon_grid_confidence=upper_confidence - lower_confidence,
        confidence_level=1.0 - confidence_delta,
        weights=fit.weights,
        fit_rejection_probabilities=fit.rejection_probabilities,
        fit_complementarity_residual=fit.complementarity_residual,
        fit_converged=fit.converged, fit_iterations=fit.iterations,
        point_rule=point_rule, upper_confidence_rule=upper_conf_rule,
        lower_grid_rule=lower_grid_rule,
        lower_grid_confidence_rule=lower_grid_conf_rule,
        calibration_component_counts=component_counts,
        validation_rejection_probabilities=validation_rp,
        invariant_benchmark_power=benchmark_power,
        invariant_benchmark_se=benchmark_se,
        invariant_benchmark_lower_confidence=benchmark_lower_confidence,
        mhg_diagnostics=_combine_phase_mhg_diagnostics(phase_diagnostics),
        grid_bracket_confidence_level=max(
            0.0, 1.0 - 2.0 * confidence_delta))
    if verbose:
        print(f"    EMW point upper={upper_point:.5f} (SE {upper_point_se:.5f}); "
              f"confidence upper={upper_confidence:.5f}")
        print(f"    grid lower={lower_point:.5f}; "
              f"grid epsilon={upper_point-lower_point:.5f}; "
              f"max validation RP before tightening={validation_rp.max():.5f}")
        print(f"    same-experiment chi-square benchmark={benchmark_power:.5f} "
              f"(SE {benchmark_se:.5f}, one-sided lower "
              f"{benchmark_lower_confidence:.5f})")
        print(f"    adaptive 0F1 selected orders: "
              f"{result.mhg_diagnostics['order_counts']}", flush=True)

    return result if return_result else result.upper_confidence


def alfd_eigval_bound_from_pooled_banks(
        kappas_alt, training_bank, audit_bank, k_eff, alpha=0.05,
        n_sim_calibration=20000, n_sim_power=50000, n_iter=600,
        max_active_support=8, seed=42, verbose=True, n_workers=1,
        confidence_delta=0.01, M_trunc=20, M_step=MHG_DEFAULT_STEP,
        M_max=MHG_DEFAULT_MAX, mhg_tol=MHG_CONV_TOL):
    """Hybrid GKM/EMW bound using shared IS discovery and iid certification.

    The pooled banks may be reused across alternatives.  Only the training bank
    selects weights/support.  The audit bank is independent, while mixture
    calibration and alternative power use fresh iid streams.  Consequently the
    saved confidence upper retains the same conditional finite-MC guarantee as
    :func:`alfd_eigval_bound`.
    """
    k_eff = _validated_integer("k_eff", k_eff)
    M_trunc = _validated_integer("M_trunc", M_trunc)
    M_step = _validated_integer("M_step", M_step)
    M_max = _validated_integer("M_max", M_max)
    seed = _validated_integer("seed", seed, minimum=0)
    if (M_trunc > M_max or not np.isfinite(mhg_tol)
            or not (1e-13 <= mhg_tol < 1.0)):
        raise ValueError("invalid adaptive-M settings")
    H, N_train = _validate_pooled_is_bank(training_bank)
    H_audit, N_audit = _validate_pooled_is_bank(audit_bank)
    if (training_bank.role != "training" or audit_bank.role != "audit"
            or training_bank.bank_id == audit_bank.bank_id):
        raise ValueError("training and audit banks must be distinct and correctly typed")
    if H != H_audit or not np.array_equal(training_bank.grid, audit_bank.grid):
        raise ValueError("training and audit banks must use the same ordered grid")
    if (training_bank.sampling_seed is None
            or audit_bank.sampling_seed is None
            or int(training_bank.sampling_seed) == int(audit_bank.sampling_seed)):
        raise ValueError(
            "training and audit banks must record distinct sampling seeds")
    if seed in (int(training_bank.sampling_seed),
                int(audit_bank.sampling_seed)):
        raise ValueError(
            "iid calibration/power seed must differ from pooled-bank seeds")
    training_settings = _authenticated_pooled_bank_settings(training_bank)
    audit_settings = _authenticated_pooled_bank_settings(audit_bank)
    if (training_bank.experiment_signature is None
            or training_bank.experiment_signature
            != audit_bank.experiment_signature):
        raise ValueError(
            "training and audit banks must share one authenticated density "
            "experiment")
    if training_bank.content_signature == audit_bank.content_signature:
        raise ValueError(
            "training and audit banks have identical numeric content and "
            "cannot support an independence claim")
    if (training_bank.eigs.shape == audit_bank.eigs.shape
            and np.array_equal(training_bank.eigs, audit_bank.eigs)):
        raise ValueError(
            "training and audit banks contain identical sampled eigenvalues "
            "and cannot support an independence claim")
    if (training_bank.k_eff is None or audit_bank.k_eff is None
            or int(training_bank.k_eff) != int(k_eff)
            or int(audit_bank.k_eff) != int(k_eff)):
        raise ValueError("pooled banks were built for a different k_eff")
    for settings in (training_settings, audit_settings):
        if (int(settings["M_start"]) != M_trunc
                or int(settings["M_step"]) != M_step
                or int(settings["M_max"]) != M_max
                or float(settings["mhg_tol"]) != float(mhg_tol)):
            raise ValueError(
                "call-time adaptive-M settings differ from pooled-bank "
                "density settings")
    if (training_bank.mhg_diagnostics is None
            or audit_bank.mhg_diagnostics is None):
        raise ValueError("certification banks must retain MHG diagnostics")

    kappas_alt = np.asarray(kappas_alt, dtype=float)
    p = kappas_alt.size
    if (p != 4 or not np.all(np.isfinite(kappas_alt))
            or np.any(kappas_alt < 0.0)
            or np.any(np.diff(kappas_alt) > 1e-10)):
        raise ValueError("pooled p=4 alternative eigenvalues are invalid")
    if not (0.0 < alpha < 1.0) or not (0.0 < confidence_delta < 1.0):
        raise ValueError("alpha and confidence_delta must lie in (0,1)")
    n_sim_calibration = _validated_integer(
        "n_sim_calibration", n_sim_calibration, minimum=2)
    n_sim_power = _validated_integer("n_sim_power", n_sim_power, minimum=2)
    if k_eff < p:
        raise ValueError("k_eff must be at least p")

    omega_alt = kappas_alt[None, :]
    phase_diagnostics = {}
    log_g_train_matrix, phase_diagnostics["training_alternative"] = \
        log_eigval_density_partial(
            training_bank.eigs, omega_alt, k_eff / 2.0,
            M_trunc=M_trunc, chunk_size=100,
            progress_label="shared-train-g", n_workers=n_workers,
            M_step=M_step, M_max=M_max, mhg_tol=mhg_tol,
            return_diagnostics=True)
    log_g_train = log_g_train_matrix[0]
    fit_full = fit_emw_weights_is(
        training_bank, log_g_train, alpha=alpha, n_iter=n_iter)
    active_indices, active_weights, dropped_mass = compress_mixture(
        fit_full.weights, max_active=max_active_support)
    active_grid = np.asarray(training_bank.grid)[active_indices]
    active_log_f_train = training_bank.log_f[active_indices]

    # GKM-comparable point calibration/tightening on the reusable discovery
    # bank.  These are diagnostics, not the confidence-certified endpoint.
    log_mix_train = logsumexp(
        _log_probability_weights(active_weights)[:, None]
        + active_log_f_train, axis=0)
    scores_train = log_g_train - log_mix_train
    mix_contributions = (training_bank.base_weights
                         * np.exp(log_mix_train - training_bank.log_q))
    gkm_point_rule = calibrate_raw_weighted_tail(
        scores_train, mix_contributions, alpha,
        method="gkm_reused_bank_mixture_ordinary_is")
    target_contributions = (_pooled_is_ratios(training_bank)
                            * training_bank.base_weights[None, :])
    gkm_grid_rule = common_grid_raw_is_tail_rule(
        scores_train, target_contributions, alpha,
        minimum_rule=gkm_point_rule)
    compressed_train_rp = gkm_importance_rejection_probabilities(
        training_bank,
        tail_rejection_probabilities(scores_train, gkm_point_rule))
    active_mask = np.zeros(H, dtype=bool)
    active_mask[active_indices] = True
    active_residual = (float(np.max(np.abs(
        compressed_train_rp[active_mask] - alpha)))
        if np.any(active_mask) else 0.0)
    slack_residual = (float(np.max(np.maximum(
        compressed_train_rp[~active_mask] - alpha, 0.0)))
        if np.any(~active_mask) else 0.0)
    compressed_residual = max(active_residual, slack_residual)
    max_contribution = float(np.max(
        _pooled_is_ratios(training_bank)
        * training_bank.base_weights[None, :]))
    compressed_tolerance = max(2.0 * max_contribution, 5e-4)
    compressed_converged = bool(compressed_residual <= compressed_tolerance)
    is_diagnostics = pooled_is_diagnostics(
        training_bank,
        tail_rejection_probabilities(scores_train, gkm_point_rule))
    is_diagnostics.update(
        full_fit_converged=bool(fit_full.converged),
        full_fit_complementarity_residual=float(
            fit_full.complementarity_residual),
        compressed_fit_converged=compressed_converged,
        compressed_fit_complementarity_residual=float(compressed_residual),
        compressed_fit_tolerance=float(compressed_tolerance))

    if verbose:
        print(f"    common GKM grid H={H}; pooled observations={N_train:,}; "
              f"active support={len(active_indices)}; discarded weight="
              f"{dropped_mass:.3e}")
        print(f"    active grid indices={active_indices.tolist()}; weights="
              f"{np.round(active_weights, 6).tolist()}")
        print(f"    ordinary-IS mass range="
              f"[{np.min(is_diagnostics['raw_mass']):.4f}, "
              f"{np.max(is_diagnostics['raw_mass']):.4f}]; min ESS fraction="
              f"{np.min(is_diagnostics['kish_ess_fraction']):.3f}", flush=True)

    M_active = [build_M(list(row) + [0.0], k_eff) for row in active_grid]
    omegas_active = np.asarray(
        [list(row) + [0.0] for row in active_grid], dtype=float)
    omegas_score = np.vstack([omegas_active, omega_alt])
    rng_cal, rng_power = [np.random.default_rng(child) for child in
                          np.random.SeedSequence(seed).spawn(2)]

    # Fresh iid mixture calibration: this is deliberately not importance
    # sampled, since the order-statistic confidence argument requires iid
    # scores from the frozen mixture.
    eigs_cal, cal_labels = _sample_mixture_eigenvalues(
        M_active, active_weights, n_sim_calibration, rng_cal)
    log_cal, phase_diagnostics["mixture_calibration"] = \
        log_eigval_density_partial(
            eigs_cal, omegas_score, k_eff / 2.0, M_trunc=M_trunc,
            chunk_size=100, progress_label="iid-mixture-cal",
            n_workers=n_workers, M_step=M_step, M_max=M_max,
            mhg_tol=mhg_tol, return_diagnostics=True)
    scores_cal = _score_from_log_densities(log_cal, active_weights)
    point_rule = calibrate_empirical_tail(
        scores_cal, alpha, method="independent_mixture_quantile")
    upper_delta_each = confidence_delta / 2.0
    grid_delta_each = confidence_delta / 2.0
    upper_conf_rule = confidence_liberal_tail_rule(
        scores_cal, alpha, upper_delta_each)
    component_counts = np.bincount(
        cal_labels, minlength=len(active_weights))
    del eigs_cal, log_cal, scores_cal
    gc.collect()

    # Independent common-grid audit.  Samples are direct iid draws within each
    # stratum, so the existing binomial order-statistic rules remain valid.
    log_g_audit_matrix, phase_diagnostics["audit_alternative"] = \
        log_eigval_density_partial(
            audit_bank.eigs, omega_alt, k_eff / 2.0,
            M_trunc=M_trunc, chunk_size=100,
            progress_label="shared-audit-g", n_workers=n_workers,
            M_step=M_step, M_max=M_max, mhg_tol=mhg_tol,
            return_diagnostics=True)
    log_mix_audit = logsumexp(
        _log_probability_weights(active_weights)[:, None]
        + audit_bank.log_f[active_indices], axis=0)
    scores_audit = (log_g_audit_matrix[0] - log_mix_audit).reshape(
        H, audit_bank.n_per_stratum)
    validation_rp = np.mean(
        tail_rejection_probabilities(scores_audit, point_rule), axis=1)
    lower_grid_rule = common_grid_tail_rule(
        scores_audit, alpha, minimum_rule=point_rule)
    per_null_delta = grid_delta_each / H
    conservative_rules = [confidence_conservative_tail_rule(
        row, alpha, per_null_delta) for row in scores_audit]
    conservative_threshold = max(
        [upper_conf_rule.threshold]
        + [rule.threshold for rule in conservative_rules])
    lower_grid_conf_rule = TailRule(
        float(conservative_threshold), 0.0,
        float(np.max(np.mean(scores_audit > conservative_threshold, axis=1))),
        "simultaneous_grid_order_statistic_size_at_most_alpha")

    # Fresh alternative sample shared by all point/confidence endpoints.
    M_alt = build_M(kappas_alt, k_eff)
    eigs_power = eigenvalues_descending(
        simulate_Xi(M_alt, n_sim_power, rng_power))
    log_power, phase_diagnostics["alternative_power"] = \
        log_eigval_density_partial(
            eigs_power, omegas_score, k_eff / 2.0, M_trunc=M_trunc,
            chunk_size=100, progress_label="iid-power", n_workers=n_workers,
            M_step=M_step, M_max=M_max, mhg_tol=mhg_tol,
            return_diagnostics=True)
    scores_power = _score_from_log_densities(log_power, active_weights)

    benchmark_rejection = (
        eigs_power[:, -1] > chi2.ppf(1.0 - alpha, df=k_eff - p + 1))
    benchmark_count = int(np.count_nonzero(benchmark_rejection))
    benchmark_power = float(benchmark_count / n_sim_power)
    benchmark_se = float(np.sqrt(
        benchmark_power * (1.0 - benchmark_power) / n_sim_power))
    benchmark_lower_confidence = clopper_pearson(
        benchmark_count, n_sim_power, upper_delta_each, "lower")

    point_rejection = tail_rejection_probabilities(scores_power, point_rule)
    upper_point = float(np.mean(point_rejection))
    upper_point_se = float(np.std(point_rejection, ddof=1)
                           / np.sqrt(n_sim_power))
    upper_conf_count = int(np.count_nonzero(
        scores_power > upper_conf_rule.threshold))
    upper_confidence = clopper_pearson(
        upper_conf_count, n_sim_power, upper_delta_each, "upper")
    lower_point = float(np.mean(
        tail_rejection_probabilities(scores_power, lower_grid_rule)))
    lower_conf_count = int(np.count_nonzero(
        scores_power > lower_grid_conf_rule.threshold))
    lower_confidence = clopper_pearson(
        lower_conf_count, n_sim_power, grid_delta_each, "lower")
    gkm_point_upper = float(np.mean(
        tail_rejection_probabilities(scores_power, gkm_point_rule)))
    gkm_grid_lower = float(np.mean(
        tail_rejection_probabilities(scores_power, gkm_grid_rule)))

    if lower_point > upper_point + 1e-12:
        raise AssertionError("independent grid-tightened power exceeds point upper")
    if lower_confidence > upper_confidence + 1e-12:
        raise AssertionError("confidence bracket endpoints are reversed")
    if gkm_grid_lower > gkm_point_upper + 1e-12:
        raise AssertionError("GKM grid-tightened power exceeds its point upper")
    if not np.isclose(point_rule.empirical_size, alpha, atol=5e-13):
        raise AssertionError("iid mixture calibration did not attain alpha")
    if upper_confidence + 1e-12 < alpha:
        raise AssertionError("confidence upper bound fell below alpha")
    if upper_confidence + 1e-12 < benchmark_lower_confidence:
        raise AssertionError(
            "EMW confidence upper is below the same-experiment benchmark lower")

    # Shared bank diagnostics are recorded once in the bank cache; include them
    # here for max-order/provenance inspection but callers should not sum these
    # repeated per-beta counts as runtime work.
    all_phases = dict(phase_diagnostics)
    all_phases["shared_training_null_bank"] = training_bank.mhg_diagnostics
    all_phases["shared_audit_null_bank"] = audit_bank.mhg_diagnostics
    mhg_diagnostics = _combine_phase_mhg_diagnostics(all_phases)
    result = ALFDBoundResult(
        upper_point=upper_point, upper_point_se=upper_point_se,
        upper_confidence=upper_confidence,
        lower_grid_point=lower_point,
        lower_grid_confidence=lower_confidence,
        epsilon_grid_point=upper_point - lower_point,
        epsilon_grid_confidence=upper_confidence - lower_confidence,
        confidence_level=1.0 - confidence_delta,
        weights=active_weights,
        fit_rejection_probabilities=compressed_train_rp,
        fit_complementarity_residual=compressed_residual,
        fit_converged=compressed_converged,
        fit_iterations=fit_full.iterations,
        point_rule=point_rule, upper_confidence_rule=upper_conf_rule,
        lower_grid_rule=lower_grid_rule,
        lower_grid_confidence_rule=lower_grid_conf_rule,
        calibration_component_counts=component_counts,
        validation_rejection_probabilities=validation_rp,
        invariant_benchmark_power=benchmark_power,
        invariant_benchmark_se=benchmark_se,
        invariant_benchmark_lower_confidence=benchmark_lower_confidence,
        mhg_diagnostics=mhg_diagnostics,
        full_weights=fit_full.weights, active_indices=active_indices,
        active_null_grid=active_grid,
        discarded_weight_mass=dropped_mass,
        gkm_point_upper=gkm_point_upper,
        gkm_grid_lower=gkm_grid_lower,
        gkm_epsilon=gkm_point_upper - gkm_grid_lower,
        importance_diagnostics=is_diagnostics,
        grid_bracket_confidence_level=max(
            0.0, 1.0 - 2.0 * confidence_delta))
    if verbose:
        print(f"    independent point upper={upper_point:.5f} "
              f"(SE {upper_point_se:.5f}); confidence upper="
              f"{upper_confidence:.5f}")
        print(f"    independent finite-grid lower={lower_point:.5f}; "
              f"epsilon={upper_point-lower_point:.5f}; max validation RP="
              f"{validation_rp.max():.5f}")
        print(f"    GKM reused-bank point/grid power="
              f"{gkm_point_upper:.5f}/{gkm_grid_lower:.5f}; "
              f"epsilon={gkm_point_upper-gkm_grid_lower:.5f}", flush=True)
    return result


# ============================================================
# Driver: common-grid production curve (11 beta points by default)
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


def _simulation_budget_diagnostics(alpha, curve_confidence, fit_grid_sizes,
                                   validation_grid_sizes, budget,
                                   events_per_family=4):
    """Return confidence ranks, precision diagnostics, and exact pair counts.

    ``fit_grid_sizes`` and ``validation_grid_sizes`` contain one entry for each
    non-null beta.  This helper is deliberately pure: preflight and tests can
    inspect exactly what a requested budget buys without evaluating a density.
    """
    if not (0.0 < alpha < 1.0) or not (0.0 < curve_confidence < 1.0):
        raise ValueError("alpha and curve_confidence must lie in (0,1)")
    fit_sizes = np.asarray(fit_grid_sizes, dtype=int)
    validation_sizes = np.asarray(validation_grid_sizes, dtype=int)
    if (fit_sizes.ndim != 1 or validation_sizes.shape != fit_sizes.shape
            or fit_sizes.size == 0 or np.any(fit_sizes < 1)
            or np.any(validation_sizes < fit_sizes)):
        raise ValueError(
            "need aligned positive fit/validation grid sizes for non-null betas")

    n_fit = _validated_integer("n_fit", budget['n_fit'], minimum=2)
    n_cal = _validated_integer(
        "n_calibration", budget['n_calibration'], minimum=2)
    n_validation = _validated_integer(
        "n_validation", budget['n_validation'], minimum=2)
    n_power = _validated_integer("n_power", budget['n_power'], minimum=2)
    n_iter = _validated_integer("n_iter", budget['n_iter'])
    _validated_integer("validation_grid_size", budget['validation_grid_size'])

    n_nonnull = int(fit_sizes.size)
    point_delta = (1.0 - float(curve_confidence)) / n_nonnull
    events_per_family = _validated_integer(
        "events_per_family", events_per_family, minimum=2)
    if events_per_family not in (2, 4):
        raise ValueError("events_per_family must be 2 or 4")
    event_delta = point_delta / float(events_per_family)
    calibration_count = _liberal_tail_rejection_count(
        n_cal, alpha, event_delta)

    validation_rows = []
    for V in sorted(set(int(x) for x in validation_sizes)):
        per_null_delta = event_delta / V
        try:
            rejection_count = _conservative_tail_rejection_count(
                n_validation, alpha, per_null_delta)
        except ValueError as exc:
            raise ValueError(
                f"n_validation={n_validation} cannot support the requested "
                f"confidence after splitting over V={V} validation points") from exc
        validation_rows.append(dict(
            V=V, per_null_delta=float(per_null_delta),
            rejection_count=int(rejection_count),
            empirical_size=float(rejection_count / n_validation),
            deflation=float(alpha - rejection_count / n_validation)))

    cp_margins = {}
    for probability in (alpha, 0.5):
        count = int(round(probability * n_power))
        observed = count / n_power
        upper = clopper_pearson(count, n_power, event_delta, "upper")
        cp_margins[probability] = float(upper - observed)

    phase_pairs = dict(
        fit=int(sum((G + 1) * G * n_fit for G in fit_sizes)),
        calibration=int(sum((G + 1) * n_cal for G in fit_sizes)),
        validation=int(sum((G + 1) * V * n_validation
                           for G, V in zip(fit_sizes, validation_sizes))),
        power=int(sum((G + 1) * n_power for G in fit_sizes)),
    )
    return dict(
        n_nonnull=n_nonnull, point_delta=float(point_delta),
        event_delta=float(event_delta),
        events_per_family=int(events_per_family),
        grid_bracket_curve_confidence=(
            float(curve_confidence) if events_per_family == 4 else
            max(0.0, 2.0 * float(curve_confidence) - 1.0)),
        n_fit=n_fit, fit_grid_min=int(fit_sizes.min()),
        fit_grid_max=int(fit_sizes.max()),
        fit_tail_se=float(math.sqrt(alpha * (1.0 - alpha) / n_fit)),
        n_iter=n_iter, n_calibration=n_cal,
        calibration_rejection_count=int(calibration_count),
        calibration_empirical_size=float(calibration_count / n_cal),
        calibration_inflation=float(calibration_count / n_cal - alpha),
        n_validation=n_validation, validation=validation_rows,
        validation_grid_min=int(validation_sizes.min()),
        validation_grid_max=int(validation_sizes.max()),
        n_power=n_power, cp_upper_margins=cp_margins,
        hoeffding_margin=float(math.sqrt(
            math.log(1.0 / event_delta) / (2.0 * n_power))),
        phase_pairs=phase_pairs, total_pairs=int(sum(phase_pairs.values())))


def _common_is_budget_diagnostics(alpha, curve_confidence, common_grid_size,
                                  n_nonnull, max_active_support, budget):
    """Exact statistical ranks and logical-pair counts for the shared-bank path."""
    H = _validated_integer("common_grid_size", common_grid_size)
    B = _validated_integer("n_nonnull", n_nonnull)
    K = min(_validated_integer("max_active_support", max_active_support), H)
    result = _simulation_budget_diagnostics(
        alpha, curve_confidence, [H] * B, [H] * B, budget,
        events_per_family=2)
    phase_pairs = dict(
        shared_fit_null=int(H * H * result['n_fit']),
        fit_alternative=int(B * H * result['n_fit']),
        calibration=int(B * (K + 1) * result['n_calibration']),
        shared_audit_null=int(H * H * result['n_validation']),
        audit_alternative=int(B * H * result['n_validation']),
        power=int(B * (K + 1) * result['n_power']),
    )
    result.update(
        common_is=True, common_grid_size=H, max_active_support=K,
        pooled_observations=int(H * result['n_fit']),
        phase_pairs=phase_pairs,
        total_pairs=int(sum(phase_pairs.values())))
    return result


def _print_simulation_budget_diagnostics(result, alpha, curve_confidence):
    """Explain the statistical consequences of one requested curve budget."""
    print("Statistical budget and confidence preflight:")
    print(f"  simultaneous confidence {curve_confidence:.3%} covers the "
          f"{result['n_nonnull']} computed non-null beta points only (not the "
          "interpolated continuum)")
    if result.get('events_per_family') == 2:
        print(f"  per-beta failure allowance={result['point_delta']:.3g}; "
              f"each of the 2 primary-upper events gets delta="
              f"{result['event_delta']:.3g}")
        print(f"  the finite-grid lower uses a separate 2-event family; the "
              f"upper and lower jointly form at least a "
              f"{result['grid_bracket_curve_confidence']:.3%} simultaneous "
              "saved-grid bracket")
    else:
        print(f"  per-beta failure allowance={result['point_delta']:.3g}; "
              f"each of 4 bracket events gets delta="
              f"{result['event_delta']:.3g}")
    if result.get('common_is'):
        print(f"  fit: common H={result['common_grid_size']} strength/shape "
              f"points, n_fit={result['n_fit']:,} per proposal "
              f"({result['pooled_observations']:,} pooled observations); "
              f"ordinary GKM importance sampling; active support capped at "
              f"{result['max_active_support']}")
        print(f"       nominal direct-tail SE reference="
              f"{100 * result['fit_tail_se']:.3f} pp; actual IS precision is "
              "reported by mass/ESS diagnostics; n_iter="
              f"{result['n_iter']} reuses the bank")
    else:
        print(f"  fit: n_fit={result['n_fit']:,} per null, G="
              f"{result['fit_grid_min']}--{result['fit_grid_max']}; nominal "
              f"Bernoulli SE at alpha={100 * result['fit_tail_se']:.3f} pp; "
              f"n_iter={result['n_iter']} (fit quality/tightness, not validity)")
    print(f"  calibration: n_calibration={result['n_calibration']:,}; "
          f"confidence-liberal rank rejects "
          f"{result['calibration_rejection_count']:,}/"
          f"{result['n_calibration']:,}="
          f"{100 * result['calibration_empirical_size']:.3f}% "
          f"({100 * result['calibration_inflation']:.3f} pp above alpha)")
    for row in result['validation']:
        print(f"  validation: n_validation={result['n_validation']:,} per null, "
              f"V={row['V']}; confidence-conservative rank rejects "
              f"{row['rejection_count']:,}/{result['n_validation']:,}="
              f"{100 * row['empirical_size']:.3f}% "
              f"({100 * row['deflation']:.3f} pp below alpha; affects only "
              "the finite-grid lower endpoint/epsilon)")
    print(f"  power: n_power={result['n_power']:,}; one-sided exact CP upper "
          f"allowance is about {100 * result['cp_upper_margins'][alpha]:.3f} "
          f"pp near power {alpha:.2f} and "
          f"{100 * result['cp_upper_margins'][0.5]:.3f} pp near power 0.50; "
          f"distribution-free Hoeffding ceiling="
          f"{100 * result['hoeffding_margin']:.3f} pp")

    if result['calibration_inflation'] > 0.01:
        print("  WARNING: calibration confidence costs more than 1 percentage "
              "point of null-size inflation; the upper can be quite loose.")
    if max(row['deflation'] for row in result['validation']) > 0.01:
        print("  WARNING: validation confidence costs more than 1 percentage "
              "point of null-size deflation; the finite-grid lower endpoint "
              "can be quite conservative (the EMW upper remains valid).")
    if result['hoeffding_margin'] > 0.01:
        print("  WARNING: n_power does not guarantee a <=1 percentage-point "
              "one-sided Monte Carlo allowance in the worst case.")

    total = result['total_pairs']
    phase_text = ", ".join(
        f"{name}={count:,} ({100.0 * count / total:.1f}%)"
        for name, count in result['phase_pairs'].items())
    print(f"  density-pair cost by phase: {phase_text}")
    if result.get('common_is'):
        print("  The null-density training/audit tables are each computed once "
              "and reused across every beta. Counts shown are for a fresh run; "
              "a compatible cached table removes its corresponding cost. The "
              "current top-K compression retains exactly the displayed active "
              "support count for every non-null beta.")
    print("  n_iter reuses fitted density tables and therefore adds no density "
          "pairs. Adaptive M retries do add raw C evaluations.\n", flush=True)


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


def _run_scalar_smoke(budget, alpha, curve_confidence, seed,
                      M_start, M_step, M_max, mhg_tol, n_workers):
    """Fast end-to-end regression; deliberately produces no curve artifact."""
    from scipy.stats import ncx2

    print("Smoke profile: scalar p=1 EMW calibration regression")
    print("  This checks the complete fit/calibration/confidence path and does "
          "not write a p=4 bound artifact.")
    verify_mhg()
    result = alfd_eigval_bound(
        (0.2,), [()], 7, alpha=alpha,
        n_sim=budget['n_fit'],
        n_sim_calibration=budget['n_calibration'],
        n_sim_validation=budget['n_validation'],
        n_sim_power=budget['n_power'], n_iter=budget['n_iter'],
        M_trunc=M_start, M_step=M_step, M_max=M_max, mhg_tol=mhg_tol,
        seed=seed, verbose=True, n_workers=n_workers,
        confidence_delta=1.0 - curve_confidence, return_result=True)
    exact = float(ncx2.sf(chi2.ppf(1.0 - alpha, 7), 7, 0.2))
    if not np.isclose(result.point_rule.empirical_size, alpha, atol=5e-13):
        raise RuntimeError("scalar smoke mixture calibration failed")
    if result.upper_confidence + 1e-12 < exact:
        raise RuntimeError(
            "scalar smoke confidence endpoint missed the exact NP benchmark")
    print(f"Smoke passed: exact NP={exact:.6f}, paper point="
          f"{result.upper_point:.6f}, confidence upper="
          f"{result.upper_confidence:.6f}. No artifact was written.")


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description=(
            "Calibrated eigenvalue-density EMW bound with adaptive 0F1 "
            "truncation and a GKM finite-grid epsilon check."))
    parser.add_argument('--version', required=True, choices=list(VERSION_LABELS),
                        help="config version label (" + ", ".join(VERSION_LABELS) + ")")
    parser.add_argument('--profile', choices=('smoke', 'production', 'reference'),
                        default='production',
                        help="simulation budget; reference follows the papers' scale")
    parser.add_argument('--force', action='store_true',
                        help="overwrite an existing adaptive artifact")
    preflight_mode = parser.add_mutually_exclusive_group()
    preflight_mode.add_argument(
        '--preflight-only', action='store_true',
        help='print exact density-pair counts/runtime warning and exit')
    preflight_mode.add_argument(
        '--benchmark-preflight', action='store_true',
        help=('time a short representative adaptive p=4 batch on this machine, '
              'extrapolate the requested run, and exit without an artifact'))
    parser.add_argument(
        '--benchmark-samples', type=int, default=None,
        help=('Xi samples for --benchmark-preflight; each is evaluated under '
              'all fitted-null Omega rows and the full-rank alternative; '
              'defaults to 64/16/2 '
              'for easy/medium/strong configurations'))
    parser.add_argument(
        '--acknowledge-expensive', action='store_true',
        help='required before a production/reference p=4 curve is started')
    parser.add_argument('--workers', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--m-start', type=int, default=None,
                        help="minimum 0F1 degree; each call adapts upward")
    parser.add_argument('--m-step', type=int, default=MHG_DEFAULT_STEP)
    parser.add_argument('--m-max', type=int, default=MHG_DEFAULT_MAX)
    parser.add_argument('--mhg-rtol', type=float, default=MHG_CONV_TOL)
    parser.add_argument('--curve-confidence', type=float, default=0.99,
                        help="simultaneous Monte Carlo confidence for the saved curve")
    parser.add_argument('--n-fit', type=int, default=None)
    parser.add_argument('--n-calibration', type=int, default=None)
    parser.add_argument('--n-validation', type=int, default=None)
    parser.add_argument('--n-power', type=int, default=None)
    parser.add_argument('--n-iter', type=int, default=None)
    parser.add_argument(
        '--validation-grid-size', type=int, default=None,
        help=('deprecated for p=4; the common grid is controlled by '
              '--grid-shapes and --grid-strengths'))
    parser.add_argument('--grid-shapes', type=int, default=9,
                        help=('common 3D nuisance-shape directions (default: 9; '
                              'includes configured stress and rank-boundary shapes)'))
    parser.add_argument('--grid-strengths', type=int, default=7,
                        help='log-spaced strengths per direction (default: 7)')
    parser.add_argument('--grid-max-strength', type=float, default=100.0,
                        help='largest nuisance eigenvalue on common grid')
    parser.add_argument('--max-active-support', type=int, default=8,
                        help=('largest retained mixture support after fitting; '
                              'the compressed mixture is independently recalibrated'))
    parser.add_argument(
        '--beta-count', type=int, default=None,
        help='saved beta-grid size (default: 11; ignored by scalar smoke)')
    args = parser.parse_args()

    k = 7
    n = 250
    alpha = 0.05

    # ---- resolve config from the version label ----
    key = VERSION_LABELS[args.version]
    cfg = ALLOWED_CONFIGS[key]
    M_start = cfg['M_start'] if args.m_start is None else args.m_start
    standard_points = cfg['standard']
    kappas = np.array(key, dtype=float)

    profiles = {
        # Smoke proves plumbing only; it is deliberately not a publication run.
        # Fast p=1 regression only; it never writes a p=4 bound artifact.
        'smoke': dict(n_fit=100, n_calibration=2000, n_validation=500,
                      n_power=5000, n_iter=100, validation_grid_size=1),
        # One well-calibrated run is statistically more useful than eight tiny,
        # cycling 150-draw fits.  Confidence limits expose remaining MC error.
        'production': dict(n_fit=2000, n_calibration=20000,
                           n_validation=2000, n_power=50000, n_iter=600,
                           validation_grid_size=32),
        # GKM uses N0=10,000, N1=100,000 and 600 iterations.  Mixture
        # calibration is separate here, so it also receives 100,000 draws.
        'reference': dict(n_fit=10000, n_calibration=100000,
                          n_validation=10000, n_power=100000, n_iter=600,
                          validation_grid_size=42),
    }
    budget = profiles[args.profile].copy()
    for name, cli_value in (
            ('n_fit', args.n_fit), ('n_calibration', args.n_calibration),
            ('n_validation', args.n_validation), ('n_power', args.n_power),
            ('n_iter', args.n_iter),
            ('validation_grid_size', args.validation_grid_size)):
        if cli_value is not None:
            budget[name] = cli_value
    for name in ('n_fit', 'n_calibration', 'n_validation', 'n_power'):
        if (not isinstance(budget[name], (int, np.integer))
                or int(budget[name]) < 2):
            parser.error(f"--{name.replace('_', '-')} must be an integer >= 2")
    for name in ('n_iter', 'validation_grid_size'):
        if (not isinstance(budget[name], (int, np.integer))
                or int(budget[name]) < 1):
            parser.error(f"--{name.replace('_', '-')} must be a positive integer")
    beta_count = args.beta_count if args.beta_count is not None else 11
    if beta_count < 1:
        parser.error("--beta-count must be positive")
    if args.validation_grid_size is not None:
        parser.error(
            "--validation-grid-size was replaced by the common-grid controls "
            "--grid-shapes and --grid-strengths")
    if args.grid_shapes < 1 or args.grid_strengths < 1:
        parser.error("--grid-shapes and --grid-strengths must be positive")
    if (not np.isfinite(args.grid_max_strength)
            or args.grid_max_strength <= 0.1):
        parser.error("--grid-max-strength must be finite and greater than 0.1")
    if args.max_active_support < 1:
        parser.error("--max-active-support must be positive")
    if args.seed < 0:
        parser.error("--seed must be nonnegative")
    if not (0.0 < args.curve_confidence < 1.0):
        parser.error("--curve-confidence must lie in (0,1)")
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
    # High-order p=4 work arrays are large.  Keep automatic parallelism
    # conservative; explicit --workers remains available after the user checks
    # the machine's memory budget.
    n_workers = (args.workers if args.workers is not None
                 else min(os.cpu_count() or 1, 16))
    if n_workers < 1:
        parser.error("--workers must be positive")

    if args.profile == 'smoke':
        if args.benchmark_preflight:
            parser.error(
                "--benchmark-preflight requires profile=production or "
                "reference; smoke is a scalar-only regression")
        if args.preflight_only:
            print("Smoke preflight: the exact scalar EMW regression uses the "
                  f"fixed smoke budgets {budget} and writes no p=4 artifact.")
            return
        _run_scalar_smoke(
            budget, alpha, args.curve_confidence, args.seed,
            M_start, args.m_step, args.m_max, args.mhg_rtol, 1)
        return

    betas_alfd = np.linspace(-2.0, 2.0, int(beta_count))
    n_betas = len(betas_alfd)
    ncp_table = np.empty((n_betas, len(kappas) + 1))
    for i, beta_value in enumerate(betas_alfd):
        ncp_table[i] = np.maximum(
            asymptotic_ncp_eigenvalues(beta_value, kappas, k, n), 0.0)
    # In this DGP beta=0 is analytically rank deficient.  Do not classify a
    # nearby alternative as null using a floating-point eigenvalue tolerance.
    nonnull = betas_alfd != 0.0
    point_delta = ((1.0 - args.curve_confidence)
                   / max(1, int(np.count_nonzero(nonnull))))

    common_grid = common_null_grid_3d(
        ncp_table[:, :-1], kappas, standard_points=standard_points,
        n_shapes=args.grid_shapes,
        n_strengths=args.grid_strengths,
        max_strength=args.grid_max_strength)
    H_common = len(common_grid)
    grid_anchor_count = H_common - (1 + args.grid_shapes * args.grid_strengths)
    if not (0 <= grid_anchor_count <= len(standard_points)):
        raise AssertionError("unexpected common-grid anchor count")
    active_cap = min(args.max_active_support, H_common)
    logical_pairs_by_beta = np.zeros(n_betas, dtype=np.int64)
    per_beta_variable = (
        H_common * (budget['n_fit'] + budget['n_validation'])
        + (active_cap + 1)
        * (budget['n_calibration'] + budget['n_power']))
    logical_pairs_by_beta[nonnull] = per_beta_variable
    nonnull_indices = np.flatnonzero(nonnull)
    if nonnull_indices.size:
        try:
            budget_diagnostics = _common_is_budget_diagnostics(
                alpha, args.curve_confidence, H_common,
                int(nonnull_indices.size), active_cap, budget)
        except ValueError as exc:
            parser.error(f"invalid statistical budget: {exc}")
        total_logical_pairs = budget_diagnostics['total_pairs']
        _print_simulation_budget_diagnostics(
            budget_diagnostics, alpha, args.curve_confidence)
    else:
        budget_diagnostics = None
        total_logical_pairs = 0
        print("Statistical budget preflight: beta grid contains only the exact "
              "null, so no Monte Carlo density evaluations are needed.\n")

    representative_pair_seconds = {
        (35, 25, 15): 0.123,
        (100, 30, 15): 0.827,
        (100, 95, 90): 31.1,
    }[key]
    optimistic_serial = total_logical_pairs * representative_pair_seconds
    optimistic_parallel = optimistic_serial / n_workers
    print("Preflight computational scale:")
    print(f"  logical density pairs: {total_logical_pairs:,} over "
          f"{int(np.count_nonzero(nonnull))} non-null betas")
    print(f"  prior developer-machine representative pair: "
          f"~{representative_pair_seconds:g} s; indicative serial "
          f"extrapolation: {_format_duration(optimistic_serial)}")
    print(f"  optimistic perfect-{n_workers}-way lower bound: "
          f"{_format_duration(optimistic_parallel)}")
    if key == (100, 95, 90):
        print("  WARNING: strong validation-grid pairs have exceeded 150 s each; "
              "the extrapolation above is materially optimistic.")
    print("  Adaptive retries add raw C evaluations; actual multiprocessing "
          "scaling is sublinear. Use --benchmark-preflight for a measurement "
          "on the machine that will run the job.\n", flush=True)
    if args.preflight_only:
        print("Preflight only; no simulation or artifact write was performed.")
        return
    if args.benchmark_preflight:
        if not nonnull_indices.size:
            parser.error("--benchmark-preflight needs at least one non-null beta")
        benchmark_samples = (args.benchmark_samples
                             if args.benchmark_samples is not None
                             else MHG_DEFAULT_BENCHMARK_SAMPLES[key])
        benchmark = _benchmark_adaptive_mhg(
            ncp_table, betas_alfd, total_logical_pairs, k,
            M_start, args.m_step, args.m_max, args.mhg_rtol,
            n_workers, benchmark_samples,
            fit_grids=[common_grid for _ in range(n_betas)])
        _print_mhg_benchmark(benchmark)
        return

    # Corrected artifacts live in a new namespace and cannot collide with the
    # legacy fixed-M/pre-calibration files.
    out_dir = os.path.join(args.version, "adaptive")
    out_npz = os.path.join(out_dir, f"alfd_eigval_{args.version}.npz")
    partial_npz = os.path.join(
        out_dir, f"alfd_eigval_{args.version}.partial.npz")
    out_png = os.path.join(out_dir, f"alfd_eigval_{args.version}.png")
    source_path = os.path.abspath(__file__)
    lib_name = 'libmhg.dylib' if sys.platform == 'darwin' else 'libmhg.so'
    provenance = dict(
        schema_version=RESULT_SCHEMA_VERSION,
        algorithm=ALGORITHM_VERSION,
        producer=os.path.basename(source_path),
        calibration_method=CALIBRATION_METHOD,
        source_sha256=_sha256_file(source_path),
        mhg_core_sha256=_sha256_file(os.path.join(MHG_DIR, 'mhg_core.c')),
        mhg_library_sha256=_sha256_file(os.path.join(MHG_DIR, lib_name)),
        mhg_build_source_sha256=_verify_mhg_build_provenance(
            MHG_DIR, lib_name),
        python_version=sys.version,
        numpy_version=np.__version__,
        scipy_version=scipy.__version__,
        platform=platform.platform(),
    )
    training_bank_seed = int(np.random.SeedSequence(
        [args.seed, 0x47534B4D]).generate_state(1, dtype=np.uint32)[0])
    audit_bank_seed = int(np.random.SeedSequence(
        [args.seed, 0x41554449]).generate_state(1, dtype=np.uint32)[0])
    run_settings = dict(
        version_label=args.version, kappas=kappas.tolist(), k=k, n=n,
        alpha=alpha, profile=args.profile, seed=args.seed,
        M_start=M_start, M_step=args.m_step, M_max=args.m_max,
        mhg_rtol=args.mhg_rtol, curve_confidence=args.curve_confidence,
        beta_count=beta_count,
        fit_grid_strategy=COMMON_GRID_METHOD,
        pooled_importance_method=POOLED_IS_METHOD,
        confidence_allocation_method=CONFIDENCE_ALLOCATION_METHOD,
        common_grid=common_grid,
        grid_shapes=args.grid_shapes,
        grid_strengths=args.grid_strengths,
        grid_max_strength=args.grid_max_strength,
        grid_anchor_count=grid_anchor_count,
        max_active_support=active_cap,
        training_bank_seed=training_bank_seed,
        audit_bank_seed=audit_bank_seed,
        n_fit=budget['n_fit'],
        n_calibration=budget['n_calibration'],
        n_validation=budget['n_validation'],
        n_power=budget['n_power'], n_iter=budget['n_iter'],
        **provenance)
    run_signature = hashlib.sha256(json.dumps(
        run_settings, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

    if os.path.isfile(out_npz) and not args.force:
        try:
            with np.load(out_npz, allow_pickle=False) as existing:
                saved_signature = str(existing['run_signature'].item())
        except (OSError, KeyError, ValueError) as exc:
            raise RuntimeError(
                f"Existing artifact {out_npz} has no trustworthy provenance "
                f"({exc}). Use --force to replace it explicitly.") from exc
        if saved_signature == run_signature:
            print(f"Compatible result already present: {out_npz}\nNothing to do.")
            return
        raise RuntimeError(
            f"Existing artifact {out_npz} was produced with different code or "
            "settings. It will not be loaded or overwritten silently. Re-run "
            "with --force to replace it, or move it aside.")

    if not args.acknowledge_expensive:
        parser.error(
            "production/reference p=4 runs require --acknowledge-expensive; "
            "inspect the preflight estimate above first")
    os.makedirs(out_dir, exist_ok=True)

    # Tee all console output into <out_dir>/run.log so the log lives with the
    # results. Workers don't print (the parent does), so this is fork/spawn safe.
    _log_fh = open(os.path.join(out_dir, "bound_run.log"), "a", buffering=1)
    sys.stdout = _Tee(sys.stdout, _log_fh)
    sys.stderr = _Tee(sys.stderr, _log_fh)
    print(f"Logging to {os.path.join(out_dir, 'bound_run.log')}")

    verify_mhg()

    print(f"Version {args.version}: k={k}, kappas={kappas.tolist()}, n={n}, alpha={alpha}")
    print(f"  output dir: {out_dir}/")
    print(f"  betas: {len(betas_alfd)} in [{betas_alfd[0]:+.2f}, {betas_alfd[-1]:+.2f}]")
    print(f"  adaptive M: start={M_start}, step={args.m_step}, max={args.m_max}, "
          f"rtol={args.mhg_rtol:.1e}; parallelism={n_workers}")
    print(f"  common null grid: H={H_common} = origin + "
          f"{args.grid_shapes} shapes x {args.grid_strengths} strengths; "
          f"{grid_anchor_count} additional exact stress anchors; maximum "
          f"strength={args.grid_max_strength:g}")
    print(f"  active mixture support cap={active_cap}; pooled method="
          f"{POOLED_IS_METHOD}")
    print(f"  profile={args.profile}, seed={args.seed}, budgets={budget}")
    print(f"  simultaneous MC confidence={args.curve_confidence:.3%}")
    print(f"  acknowledged preflight: {total_logical_pairs:,} logical pairs; "
          f"optimistic perfect-{n_workers}-way estimate "
          f"{_format_duration(optimistic_parallel)}")
    print(f"  run signature={run_signature}\n")

    t_total = time.time()

    # These beta-invariant null tables are the principal GKM computational
    # reuse.  Their cache keys authenticate the grid, RNG stream, adaptive-M
    # settings and current density implementation.
    training_bank = build_or_load_pooled_is_bank(
        common_grid, k, budget['n_fit'], training_bank_seed,
        M_start=M_start, M_step=args.m_step, M_max=args.m_max,
        mhg_tol=args.mhg_rtol, n_workers=n_workers, role="training",
        cache_dir=out_dir, cache_metadata=provenance)
    audit_bank = build_or_load_pooled_is_bank(
        common_grid, k, budget['n_validation'], audit_bank_seed,
        M_start=M_start, M_step=args.m_step, M_max=args.m_max,
        mhg_tol=args.mhg_rtol, n_workers=n_workers, role="audit",
        cache_dir=out_dir, cache_metadata=provenance)

    bounds_confidence = np.full(n_betas, np.nan)
    bounds_point = np.full(n_betas, np.nan)
    bounds_point_se = np.full(n_betas, np.nan)
    lower_grid_point = np.full(n_betas, np.nan)
    lower_grid_confidence = np.full(n_betas, np.nan)
    epsilon_grid_point = np.full(n_betas, np.nan)
    epsilon_grid_confidence = np.full(n_betas, np.nan)
    benchmark_power = np.full(n_betas, np.nan)
    benchmark_se = np.full(n_betas, np.nan)
    benchmark_lower_confidence = np.full(n_betas, np.nan)
    fit_residual = np.full(n_betas, np.nan)
    fit_converged = np.zeros(n_betas, dtype=bool)
    fit_iterations = np.zeros(n_betas, dtype=int)
    max_validation_rp = np.full(n_betas, np.nan)
    max_m_used = np.zeros(n_betas, dtype=int)
    active_support_count = np.zeros(n_betas, dtype=int)
    discarded_weight_mass = np.full(n_betas, np.nan)
    gkm_point_upper = np.full(n_betas, np.nan)
    gkm_grid_lower = np.full(n_betas, np.nan)
    gkm_epsilon = np.full(n_betas, np.nan)
    fitted_weights_full = np.full((n_betas, H_common), np.nan)
    retained_weights = np.full((n_betas, active_cap), np.nan)
    retained_indices = np.full((n_betas, active_cap), -1, dtype=int)
    diagnostics_json = np.full(n_betas, '', dtype='<U100000')

    checkpoint_arrays = dict(
        bounds_confidence=bounds_confidence, bounds_point=bounds_point,
        bounds_point_se=bounds_point_se, lower_grid_point=lower_grid_point,
        lower_grid_confidence=lower_grid_confidence,
        epsilon_grid_point=epsilon_grid_point,
        epsilon_grid_confidence=epsilon_grid_confidence,
        benchmark_power=benchmark_power, benchmark_se=benchmark_se,
        benchmark_lower_confidence=benchmark_lower_confidence,
        fit_residual=fit_residual, fit_converged=fit_converged,
        fit_iterations=fit_iterations,
        max_validation_rp=max_validation_rp, max_m_used=max_m_used,
        active_support_count=active_support_count,
        discarded_weight_mass=discarded_weight_mass,
        gkm_point_upper=gkm_point_upper,
        gkm_grid_lower=gkm_grid_lower, gkm_epsilon=gkm_epsilon,
        fitted_weights_full=fitted_weights_full,
        retained_weights=retained_weights,
        retained_indices=retained_indices,
        diagnostics_json=diagnostics_json)
    if os.path.isfile(partial_npz) and not args.force:
        try:
            with np.load(partial_npz, allow_pickle=False) as checkpoint:
                if str(checkpoint['run_signature'].item()) != run_signature:
                    raise ValueError("run signature differs")
                for name, destination in checkpoint_arrays.items():
                    saved = checkpoint[name]
                    if saved.shape != destination.shape:
                        raise ValueError(f"shape mismatch for {name}")
                    destination[...] = saved
        except (OSError, KeyError, ValueError) as exc:
            raise RuntimeError(
                f"Cannot resume checkpoint {partial_npz}: {exc}. Use --force "
                "to start this adaptive run again.") from exc
        print(f"  resumed {np.count_nonzero(np.isfinite(bounds_confidence))}/"
              f"{n_betas} beta points from {partial_npz}")
    beta_times = []

    for i, b in enumerate(betas_alfd):
        if np.isfinite(bounds_confidence[i]):
            print(f"========== beta {i+1}/{n_betas} = {b:+.2f}: "
                  "loaded from compatible checkpoint ==========", flush=True)
            continue
        t0 = time.time()
        header = (f"========== beta {i+1}/{n_betas} = {b:+.2f}  "
                  f"(elapsed total: {(time.time()-t_total)/60:.1f} min)")
        if beta_times:
            avg = np.mean(beta_times)
            eta = avg * (n_betas - i)
            header += f"  ETA for remainder: {eta/60:.1f} min"
        header += " =========="
        print(header, flush=True)

        ncp = ncp_table[i]
        ncp_str = "[" + ", ".join(f"{x:5.2f}" for x in ncp) + "]"
        exact_null = not nonnull[i]
        beta_seed = None
        if exact_null:
            print(f"  NCP={ncp_str}; exact rank-deficient alternative -> "
                  f"randomized bound alpha={alpha}", flush=True)
            result = _exact_null_result(alpha, H_common, common_grid)
        else:
            print(f"  NCP={ncp_str}; common grid H={H_common}; "
                  f"active support cap={active_cap}; beta-specific logical "
                  f"density pairs={logical_pairs_by_beta[i]:,}",
                  flush=True)
            beta_seed = int(np.random.SeedSequence(
                [args.seed, 0x42455441, i]).generate_state(
                    1, dtype=np.uint32)[0])
            result = alfd_eigval_bound_from_pooled_banks(
                kappas_alt=tuple(ncp), training_bank=training_bank,
                audit_bank=audit_bank, k_eff=k, alpha=alpha,
                n_sim_calibration=budget['n_calibration'],
                n_sim_power=budget['n_power'], n_iter=budget['n_iter'],
                max_active_support=active_cap,
                M_trunc=M_start, M_step=args.m_step, M_max=args.m_max,
                mhg_tol=args.mhg_rtol, seed=beta_seed,
                verbose=True, n_workers=n_workers,
                confidence_delta=point_delta)

        bounds_confidence[i] = result.upper_confidence
        bounds_point[i] = result.upper_point
        bounds_point_se[i] = result.upper_point_se
        lower_grid_point[i] = result.lower_grid_point
        lower_grid_confidence[i] = result.lower_grid_confidence
        epsilon_grid_point[i] = result.epsilon_grid_point
        epsilon_grid_confidence[i] = result.epsilon_grid_confidence
        benchmark_power[i] = result.invariant_benchmark_power
        benchmark_se[i] = result.invariant_benchmark_se
        benchmark_lower_confidence[i] = (
            result.invariant_benchmark_lower_confidence)
        fit_residual[i] = result.fit_complementarity_residual
        fit_converged[i] = result.fit_converged
        fit_iterations[i] = result.fit_iterations
        max_validation_rp[i] = float(np.max(result.validation_rejection_probabilities))
        max_m_used[i] = int(result.mhg_diagnostics['max_order'])
        active_support_count[i] = (0 if exact_null else len(result.weights))
        discarded_weight_mass[i] = result.discarded_weight_mass
        gkm_point_upper[i] = result.gkm_point_upper
        gkm_grid_lower[i] = result.gkm_grid_lower
        gkm_epsilon[i] = result.gkm_epsilon
        if result.full_weights is not None:
            if np.asarray(result.full_weights).shape != (H_common,):
                raise RuntimeError("fitted full-weight vector has wrong shape")
            fitted_weights_full[i] = result.full_weights
        if not exact_null:
            count = active_support_count[i]
            retained_weights[i, :count] = result.weights
            retained_indices[i, :count] = result.active_indices
        diagnostic_record = json.dumps(_json_safe(dict(
            weights=result.weights,
            certification_seed=beta_seed,
            full_weights=result.full_weights,
            active_indices=result.active_indices,
            active_null_grid=result.active_null_grid,
            discarded_weight_mass=result.discarded_weight_mass,
            fit_rejection_probabilities=result.fit_rejection_probabilities,
            point_rule=result.point_rule,
            upper_confidence_rule=result.upper_confidence_rule,
            lower_grid_rule=result.lower_grid_rule,
            lower_grid_confidence_rule=result.lower_grid_confidence_rule,
            calibration_component_counts=result.calibration_component_counts,
            validation_rejection_probabilities=result.validation_rejection_probabilities,
            gkm_point_upper=result.gkm_point_upper,
            gkm_grid_lower=result.gkm_grid_lower,
            gkm_epsilon=result.gkm_epsilon,
            importance=result.importance_diagnostics,
            grid_bracket_confidence_level=(
                result.grid_bracket_confidence_level),
            mhg=result.mhg_diagnostics)), sort_keys=True)
        if len(diagnostic_record) > diagnostics_json.dtype.itemsize // 4:
            raise RuntimeError(
                "per-beta diagnostic JSON exceeds the checkpoint field; "
                "increase its declared width rather than truncating it")
        diagnostics_json[i] = diagnostic_record

        _atomic_savez(
            partial_npz, run_signature=np.array(run_signature),
            betas=betas_alfd, ncp=ncp_table,
            **{name: value for name, value in checkpoint_arrays.items()})

        elapsed = time.time() - t0
        beta_times.append(elapsed)
        avg = np.mean(beta_times)
        eta = avg * (n_betas - (i + 1))
        print(f"  -> paper point={bounds_point[i]:.5f} +/- {bounds_point_se[i]:.5f}; "
              f"simultaneous-MC upper conditional on density accuracy="
              f"{bounds_confidence[i]:.5f}")
        print(f"     finite-grid lower={lower_grid_point[i]:.5f}; "
              f"epsilon={epsilon_grid_point[i]:.5f}; max M={max_m_used[i]}")
        if not exact_null:
            print(f"     active support={active_support_count[i]}/{H_common}; "
                  f"discarded fitted mass={discarded_weight_mass[i]:.3e}; "
                  f"GKM reused-bank epsilon={gkm_epsilon[i]:.5f}")
        print(f"     (this beta: {elapsed:.0f}s = {elapsed/60:.1f} min; "
              f"avg {avg:.0f}s/beta; ETA remainder: {eta/60:.1f} min)\n",
              flush=True)

    total_elapsed = time.time() - t_total
    print(f"\n=== Total runtime: {total_elapsed/60:.1f} min ===")
    print("Results (paper point; simultaneous-MC upper conditional on density accuracy):")
    for i, b in enumerate(betas_alfd):
        print(f"  beta={b:+.2f}: point={bounds_point[i]:.5f} +/- "
              f"{bounds_point_se[i]:.5f}; upper={bounds_confidence[i]:.5f}; "
              f"grid epsilon={epsilon_grid_point[i]:.5f}")

    # `bounds` is intentionally the confidence-valid curve.  `bounds_se` is
    # zero because this endpoint already includes its one-sided MC allowance;
    # the paper-style estimate and its SE are saved under explicit names.
    save_payload = dict(
        schema_version=np.array(RESULT_SCHEMA_VERSION),
        algorithm=np.array(ALGORITHM_VERSION), producer=np.array('alfd_eigval.py'),
        calibration_method=np.array(CALIBRATION_METHOD),
        confidence_allocation_method=np.array(
            CONFIDENCE_ALLOCATION_METHOD),
        bound_kind=np.array(BOUND_KIND),
        confidence_scope=np.array('saved_beta_grid_only'),
        density_accuracy_scope=np.array('adaptive_empirical_tail_criterion'),
        grid_certificate_scope=np.array('finite_grid_only'),
        version_label=np.array(args.version), run_signature=np.array(run_signature),
        source_sha256=np.array(provenance['source_sha256']),
        mhg_core_sha256=np.array(provenance['mhg_core_sha256']),
        mhg_library_sha256=np.array(provenance['mhg_library_sha256']),
        mhg_build_source_sha256=np.array(
            provenance['mhg_build_source_sha256']),
        settings_json=np.array(json.dumps(run_settings, sort_keys=True)),
        betas=betas_alfd, bounds=bounds_confidence,
        bounds_se=np.zeros_like(bounds_confidence),
        bounds_point=bounds_point, bounds_point_se=bounds_point_se,
        bounds_grid_lower=lower_grid_point,
        bounds_grid_lower_confidence=lower_grid_confidence,
        epsilon_grid=epsilon_grid_point,
        epsilon_grid_confidence=epsilon_grid_confidence,
        invariant_benchmark_power=benchmark_power,
        invariant_benchmark_se=benchmark_se,
        invariant_benchmark_lower_confidence=benchmark_lower_confidence,
        fit_complementarity_residual=fit_residual,
        fit_converged=fit_converged,
        fit_iterations=fit_iterations,
        max_validation_rejection_probability=max_validation_rp,
        max_m_used=max_m_used,
        common_null_grid=np.asarray(common_grid, dtype=float),
        common_grid_size=np.array(H_common),
        active_support_count=active_support_count,
        discarded_weight_mass=discarded_weight_mass,
        fitted_weights_full=fitted_weights_full,
        retained_weights=retained_weights,
        retained_indices=retained_indices,
        gkm_point_upper=gkm_point_upper,
        gkm_grid_lower=gkm_grid_lower, gkm_epsilon=gkm_epsilon,
        training_bank_id=np.array(training_bank.bank_id),
        audit_bank_id=np.array(audit_bank.bank_id),
        training_bank_content_signature=np.array(
            training_bank.content_signature),
        audit_bank_content_signature=np.array(audit_bank.content_signature),
        pooled_experiment_signature=np.array(
            training_bank.experiment_signature),
        training_bank_mhg_diagnostics_json=np.array(json.dumps(
            _json_safe(training_bank.mhg_diagnostics), sort_keys=True)),
        audit_bank_mhg_diagnostics_json=np.array(json.dumps(
            _json_safe(audit_bank.mhg_diagnostics), sort_keys=True)),
        diagnostics_json=diagnostics_json,
        ncp=ncp_table, kappas=kappas, k=np.array(k), n=np.array(n),
        alpha=np.array(alpha), curve_confidence=np.array(args.curve_confidence),
        grid_lower_curve_confidence=np.array(args.curve_confidence),
        grid_bracket_curve_confidence=np.array(
            max(0.0, 2.0 * args.curve_confidence - 1.0)),
        point_confidence_delta=np.array(point_delta), M_start=np.array(M_start),
        M_step=np.array(args.m_step), M_max=np.array(args.m_max),
        mhg_rtol=np.array(args.mhg_rtol), seed=np.array(args.seed),
        n_fit=np.array(budget['n_fit']),
        n_calibration=np.array(budget['n_calibration']),
        n_validation=np.array(budget['n_validation']),
        n_power=np.array(budget['n_power']), n_iter=np.array(budget['n_iter']),
        grid_shapes=np.array(args.grid_shapes),
        grid_strengths=np.array(args.grid_strengths),
        grid_max_strength=np.array(args.grid_max_strength),
        grid_anchor_count=np.array(grid_anchor_count),
        max_active_support=np.array(active_cap))
    _atomic_savez(out_npz, **save_payload)
    if os.path.isfile(partial_npz):
        os.unlink(partial_npz)

    plt.figure(figsize=(8, 5))
    plt.plot(betas_alfd, bounds_confidence, 'go-', linewidth=2, markersize=5,
             label=(rf'EMW upper ({args.curve_confidence:.0%} simultaneous MC; '
                    'conditional on density accuracy)'))
    plt.plot(betas_alfd, bounds_point, 'g--', linewidth=1.2,
             label='EMW paper-style point estimate')
    plt.plot(betas_alfd, lower_grid_point, color='darkorange', linestyle=':',
             linewidth=1.5, label='GKM finite-grid lower endpoint')
    plt.axhline(alpha, color='gray', linestyle=':', label=rf'$\alpha={alpha}$')
    plt.xlabel(r'True $\beta$')
    plt.ylabel('Upper bound on power')
    plt.title(rf'Calibrated EMW bound, $\kappa$ = {kappas.tolist()}, '
              rf'adaptive $M\leq {args.m_max}$')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=120)
    print(f"\nSaved {out_npz} and {out_png}")


if __name__ == "__main__":
    # On macOS Python defaults to 'spawn'; calling freeze_support() is the
    # standard guard so the worker processes import this module cleanly
    # instead of re-executing main().
    import multiprocessing as mp
    mp.freeze_support()
    main()
