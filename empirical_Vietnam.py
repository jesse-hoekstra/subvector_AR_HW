#!/usr/bin/env python3
"""Replicate TCM (2010) Table 5 & GKM (2019) Table 2, Specification 2, and add HW intervals.

This is a standalone empirical replication for the Tanaka-Camerer-Nguyen
Vietnam data.  It first validates the published 2SLS, Wald,
conditional subvector AR (C_{GKM}), and unconditional subvector AR (U) results.  Subseqquently it
computes the Hoekstra-Windmeijer conditional interval (C_{HW}) after those
GKM benchmarks pass.

This supports the empirical results of the paper (https://arxiv.org/pdf/2601.17843).

Run from the repository root with an environment containing pandas, NumPy,
SciPy, and Matplotlib:

    python3 empirical_Vietnam.py

Only files below results/Vietnam are written.

Critical-value data are taken from the official GKM replication archive: 
https://pure.uva.nl/ws/files/46416176/666_3195_1_SP.zip
"""

from __future__ import annotations

import math
import re
import sys
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from scipy import linalg
    from scipy.optimize import brentq
    from scipy.stats import chi2, norm
except ImportError as exc:  
    raise SystemExit(
        "Missing a required package. Install pandas, numpy, scipy, and "
        f"matplotlib before running this script. Original error: {exc}"
    ) from exc


# =============================================================================
# EDITABLE SPECIFICATION AND REPLICATION CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "Vietnam"
RESULTS_DIR = PROJECT_ROOT / "results" / "Vietnam"
GRID_DIR = RESULTS_DIR / "grids"

DATA_FILE = "20060431_risk.dta"
README_FILES = ("20060431_ReadMeFile.doc", "20060431_ReadMeFile.pdf")
DO_FILE = "20060431_dofile.do"

DEPENDENT = "vfctnc"
INCLUDED_EXOGENOUS = (
    "chinese",
    "age",
    "gender",
    "edu",
    "market",
    "south",
    "constant",
)
ENDOGENOUS = ("nmlrlincome", "mnincome")
EXCLUDED_INSTRUMENTS = ("headnowork", "rainfall")
SPEC1_ENDOGENOUS = "income"

ROW_ORDER = INCLUDED_EXOGENOUS + ENDOGENOUS
DISPLAY_NAMES = {
    "chinese": "Chinese",
    "age": "Age",
    "gender": "Gender",
    "edu": "Edu",
    "market": "Market",
    "south": "South",
    "constant": "Constant",
    "nmlrlincome": "Relative Income (IV)",
    "mnincome": "Mean income (IV)",
}

ALPHA = 0.05
EXPECTED_N = 181
EXPECTED_POINT_TOLERANCE = 5.01e-4
EXPECTED_WALD_TOLERANCE = 5.01e-4
# The published AR intervals were inverted on a finite grid.  Root-refined
# endpoints can differ from the printed inward grid endpoint by almost 0.001.
EXPECTED_AR_ENDPOINT_TOLERANCE = 1.01e-3
EIGEN_ABS_TOLERANCE = 1e-9
WHITENING_TOLERANCE = 5e-9
ROOT_TOLERANCE = 2e-12
AUDIT_GRID_SIZE = 4001

REQUESTED_PLOT_VARIABLES = ("gender", "south")
EXOGENOUS_CONDITIONING_PLOT_VARIABLES = (
    "chinese",
    "age",
    "gender",
    "edu",
    "market",
    "south",
)
EXOGENOUS_CONDITIONING_BETA_MIN = -0.125
EXOGENOUS_CONDITIONING_BETA_MAX = 0.125
EXOGENOUS_CONDITIONING_BETA_STEP = 0.001
AGE_CONDITIONING_BETA_MIN = -0.0312
AGE_CONDITIONING_BETA_MAX = 0.0312
AGE_CONDITIONING_BETA_STEP = 0.0001
GENDER_FIGURE6_BETA_MIN = -0.5
GENDER_FIGURE6_BETA_MAX = 0.75
GENDER_FIGURE6_STEP = 0.001
SOUTH_PLOT_BETA_MIN = -0.75
SOUTH_PLOT_BETA_MAX = 0.5
SOUTH_PLOT_BETA_STEP = 0.001
SOUTH_PLOT_Y_MIN = 0.0
SOUTH_PLOT_Y_MAX = 7.0

GKM_GENDER_SPEC1_REPORTED = {
    "estimate": 0.022,
    "C": (-0.135, 0.302),
    "U": (-0.140, 0.307),
}

# GKM Table 2, Specification 2, printed benchmarks.  These values are used only
# as validation targets and for the publication-style display; all calculations
# retain their independently replicated full-precision values.
GKM_REPORTED = {
    "chinese": {
        "estimate": -0.096,
        "W": (-0.361, 0.169),
        "C": (-0.394, 0.165),
        "U": (-0.396, 0.166),
    },
    "age": {
        "estimate": -0.006,
        "W": (-0.011, -0.002),
        "C": (-0.011, -0.002),
        "U": (-0.011, -0.002),
    },
    "gender": {
        "estimate": -0.006,
        "W": (-0.120, 0.108),
        "C": (-0.120, 0.117),
        "U": (-0.121, 0.118),
    },
    "edu": {
        "estimate": -0.028,
        "W": (-0.046, -0.009),
        "C": (-0.055, -0.008),
        "U": (-0.055, -0.008),
    },
    "market": {
        "estimate": -0.013,
        "W": (-0.042, 0.017),
        "C": (-0.044, 0.016),
        "U": (-0.045, 0.016),
    },
    "south": {
        "estimate": -0.148,
        "W": (-0.301, 0.005),
        "C": (-0.313, 0.008),
        "U": (-0.314, 0.008),
    },
    "constant": {
        "estimate": 0.992,
        "W": (0.684, 1.299),
        "C": (0.676, 1.363),
        "U": (0.675, 1.366),
    },
    "nmlrlincome": {
        "estimate": 0.049,
        "W": (-0.235, 0.333),
        "C": (-0.334, 0.628),
        "U": (-0.339, 0.638),
    },
    "mnincome": {
        "estimate": 0.010,
        "W": (-0.000, 0.021),
        "C": (0.001, 0.020),
        "U": (-0.001, 0.022),
    },
}


# Exact official GKM 5% table for k-m_W=1.  Interior conditional quantiles were
# rounded *up* to one decimal by Critical_Value_Tables.ox.  The values at 1000
# and infinity retain the full precision stored in the official .bn7 file.
GKM_CV_KAPPA_DF1_5PCT = np.array(
    [
        0.0,
        0.5,
        0.6,
        0.7,
        0.9,
        1.0,
        1.2,
        1.3,
        1.5,
        1.6,
        1.8,
        2.0,
        2.1,
        2.3,
        2.5,
        2.7,
        2.9,
        3.1,
        3.4,
        3.6,
        3.9,
        4.1,
        4.4,
        4.8,
        5.1,
        5.5,
        6.0,
        6.5,
        7.0,
        7.8,
        8.6,
        9.8,
        11.4,
        13.9,
        18.5,
        29.7,
        1000.0,
        np.inf,
    ],
    dtype=float,
)

GKM_CV_95_DF1 = np.array(
    [
        0.0,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1.0,
        1.1,
        1.2,
        1.3,
        1.4,
        1.5,
        1.6,
        1.7,
        1.8,
        1.9,
        2.0,
        2.1,
        2.2,
        2.3,
        2.4,
        2.5,
        2.6,
        2.7,
        2.8,
        2.9,
        3.0,
        3.1,
        3.2,
        3.3,
        3.4,
        3.5,
        3.6,
        3.7,
        3.8,
        3.83761994673704,
        3.841458820731395,
    ],
    dtype=float,
)


# Exact official GKM tables for k-m_W=2, used by the Gender Specification 1
# Figure 6 replication.  The archive filename convention uses k_(df+1), so
# these pairs come from k_3_level_{0.9,0.95,0.99}.{in7,bn7}.  As in the Ox
# implementation, finite knots are linearly interpolated and the last segment
# approaches the stored infinity knot with the exponential tail rule.
GKM_CV_TABLES_DF2 = {
    0.10: np.array(
        [
            (0.0, 0.0),
            (0.5, 0.4), (0.6, 0.5), (0.7, 0.6), (0.8, 0.7),
            (1.0, 0.8), (1.1, 0.9), (1.3, 1.0), (1.4, 1.1),
            (1.6, 1.2), (1.7, 1.3), (1.9, 1.4), (2.0, 1.5),
            (2.2, 1.6), (2.4, 1.7), (2.5, 1.8), (2.7, 1.9),
            (2.9, 2.0), (3.1, 2.1), (3.3, 2.2), (3.5, 2.3),
            (3.7, 2.4), (3.9, 2.5), (4.1, 2.6), (4.3, 2.7),
            (4.6, 2.8), (4.8, 2.9), (5.1, 3.0), (5.4, 3.1),
            (5.7, 3.2), (6.1, 3.3), (6.5, 3.4), (6.9, 3.5),
            (7.3, 3.6), (7.9, 3.7), (8.5, 3.8), (9.2, 3.9),
            (10.1, 4.0), (11.2, 4.1), (12.7, 4.2), (15.0, 4.3),
            (18.6, 4.4), (25.9, 4.5), (47.2, 4.6),
            (1000.0, 4.60056769559785),
            (np.inf, 4.60517018598809),
        ],
        dtype=float,
    ),
    0.05: np.array(
        [
            (0.0, 0.0),
            (0.7, 0.6), (0.8, 0.7), (0.9, 0.8), (1.0, 0.9),
            (1.1, 1.0), (1.3, 1.1), (1.4, 1.2), (1.5, 1.3),
            (1.6, 1.4), (1.8, 1.5), (1.9, 1.6), (2.0, 1.7),
            (2.2, 1.8), (2.3, 1.9), (2.4, 2.0), (2.6, 2.1),
            (2.7, 2.2), (2.9, 2.3), (3.0, 2.4), (3.2, 2.5),
            (3.3, 2.6), (3.5, 2.7), (3.6, 2.8), (3.8, 2.9),
            (4.0, 3.0), (4.2, 3.1), (4.3, 3.2), (4.5, 3.3),
            (4.7, 3.4), (4.9, 3.5), (5.1, 3.6), (5.3, 3.7),
            (5.5, 3.8), (5.8, 3.9), (6.0, 4.0), (6.3, 4.1),
            (6.5, 4.2), (6.8, 4.3), (7.1, 4.4), (7.5, 4.5),
            (7.8, 4.6), (8.2, 4.7), (8.6, 4.8), (9.1, 4.9),
            (9.7, 5.0), (10.3, 5.1), (11.0, 5.2), (11.9, 5.3),
            (13.0, 5.4), (14.5, 5.5), (16.5, 5.6), (19.5, 5.7),
            (24.7, 5.8), (35.4, 5.9),
            (1000.0, 5.98548692022524),
            (np.inf, 5.99146454710798),
        ],
        dtype=float,
    ),
    0.01: np.array(
        [
            (0.0, 0.0),
            (1.6, 1.5), (1.7, 1.6), (1.8, 1.7), (2.0, 1.9),
            (2.2, 2.1), (2.4, 2.3), (2.7, 2.5), (2.9, 2.7),
            (3.1, 2.9), (3.3, 3.1), (3.6, 3.3), (3.8, 3.5),
            (4.1, 3.7), (4.3, 3.9), (4.6, 4.1), (4.8, 4.3),
            (5.1, 4.5), (5.4, 4.7), (5.6, 4.9), (5.9, 5.1),
            (6.2, 5.3), (6.5, 5.5), (6.9, 5.7), (7.2, 5.9),
            (7.5, 6.1), (7.9, 6.3), (8.3, 6.5), (8.7, 6.7),
            (9.2, 6.9), (9.7, 7.1), (10.3, 7.3), (11.0, 7.5),
            (11.7, 7.7), (12.6, 7.9), (13.8, 8.1), (15.3, 8.3),
            (17.5, 8.5), (21.1, 8.7), (28.3, 8.9), (49.5, 9.1),
            (89.0, 9.2), (1000.0, 9.20127548715497),
            (np.inf, 9.21034037197618),
        ],
        dtype=float,
    ),
}


class ReplicationError(RuntimeError):
    """Raised when a published benchmark or mathematical check fails."""


class Tee:
    """Write console output to both the terminal and the replication log."""

    def __init__(self, *streams: object) -> None:
        self.streams = streams

    def write(self, text: str) -> int:
        for stream in self.streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


@dataclass(frozen=True)
class Interval:
    lower: float
    upper: float

    @property
    def length(self) -> float:
        return self.upper - self.lower

    def contains(self, value: float, tolerance: float = 1e-12) -> bool:
        return self.lower - tolerance <= value <= self.upper + tolerance


@dataclass
class TwoSLSResult:
    coefficients: dict[str, float]
    standard_errors: dict[str, float]
    wald_intervals: dict[str, Interval]
    residuals: np.ndarray
    x: np.ndarray
    z: np.ndarray
    x_hat: np.ndarray
    gkm_wald_variance_df: int
    stata_small_sample_df: int


@dataclass
class TestProblem:
    variable: str
    display_name: str
    case: str
    y: np.ndarray
    tested_regressor: np.ndarray
    nuisance_w: np.ndarray
    residualized_z: np.ndarray
    qz: np.ndarray
    original_n: int
    rank_c: int
    effective_n: int
    k: int
    m_w: int
    denominator_df: int
    controls: tuple[str, ...]
    instruments: tuple[str, ...]
    first_stage_rank: int


@dataclass
class RowResult:
    variable: str
    display_name: str
    estimate: float
    wald: Interval
    conditional_gkm: Interval
    unconditional: Interval
    conditional_hw: Interval | None = None


def matrix_rank(a: np.ndarray) -> int:
    """Matrix rank with an explicit LAPACK-style relative tolerance."""

    singular_values = linalg.svdvals(a)
    if singular_values.size == 0:
        return 0
    tolerance = max(a.shape) * np.finfo(float).eps * singular_values[0]
    return int(np.sum(singular_values > tolerance))


def orthonormal_basis(a: np.ndarray, name: str) -> tuple[np.ndarray, int]:
    """Return a pivoted-QR basis for the column space of ``a``."""

    if a.ndim != 2:
        raise ReplicationError(f"{name} must be a two-dimensional matrix")
    rank = matrix_rank(a)
    q, _, _ = linalg.qr(a, mode="economic", pivoting=True)
    return q[:, :rank], rank


def residualize(
    a: np.ndarray, controls: np.ndarray
) -> tuple[np.ndarray, int]:
    """Residualize columns of ``a`` against controls using pivoted QR."""

    qc, rank_c = orthonormal_basis(controls, "control matrix C")
    return a - qc @ (qc.T @ a), rank_c


def normalize_do_file(text: str) -> str:
    return " ".join(text.lower().replace("\r", " ").split())


def inspect_vietnam_directory() -> None:
    """Print the local file inventory and verify the original TCN command."""

    if not DATA_DIR.is_dir():
        raise FileNotFoundError(f"Vietnam data directory not found: {DATA_DIR}")

    print("VIETNAM DATA INSPECTION")
    print("=======================")
    print(f"Directory: {DATA_DIR}")
    format_descriptions = {
        ".dta": "Stata data file",
        ".do": "Stata command file",
        ".doc": "legacy Microsoft Word document",
        ".pdf": "PDF document",
    }
    files = sorted(path for path in DATA_DIR.iterdir() if path.is_file())
    for path in files:
        description = format_descriptions.get(path.suffix.lower(), "unknown")
        print(f"  {path.name:<30} {description:<32} {path.stat().st_size:>9,d} bytes")

    expected_files = {DATA_FILE, DO_FILE, *README_FILES, "20060431_time.dta"}
    missing_files = sorted(expected_files - {path.name for path in files})
    if missing_files:
        raise FileNotFoundError(f"Required Vietnam files missing: {missing_files}")

    do_text = (DATA_DIR / DO_FILE).read_text(encoding="ascii", errors="replace")
    normalized = normalize_do_file(do_text)
    exact_command_spec1 = (
        "ivreg vfctnc chinese age gender edu "
        "(income =headnowork rainfall) market south, first"
    )
    exact_command = (
        "ivreg vfctnc chinese age gender edu "
        "(nmlrlincome mnincome = headnowork rainfall) market south, first"
    )
    if exact_command_spec1 not in normalized:
        raise ReplicationError(
            "The supplied do-file does not contain the expected TCN Table 5 "
            f"Specification 1 command: {exact_command_spec1}"
        )
    if exact_command not in normalized:
        raise ReplicationError(
            "The supplied do-file does not contain the expected TCN Table 5 "
            f"Specification 2 command: {exact_command}"
        )
    print("\nVerified original TCN Table 5 commands:")
    print(f"  Specification 1: {exact_command_spec1}")
    print("  Specification 2:")
    print(f"  {exact_command}")
    print("README documentation was inspected from both supplied DOC and PDF files.")


def load_vietnam_data() -> pd.DataFrame:
    """Load the risk data and apply exactly the Specification 2 complete cases."""

    risk_path = DATA_DIR / DATA_FILE
    time_path = DATA_DIR / "20060431_time.dta"
    risk_all = pd.read_stata(risk_path, convert_categoricals=False)
    time_all = pd.read_stata(time_path, convert_categoricals=False)

    print("\nDataset inventory:")
    print(f"  {risk_path.name}: shape={risk_all.shape}")
    print(f"    columns={list(risk_all.columns)}")
    print(f"  {time_path.name}: shape={time_all.shape}")
    print(f"    columns={list(time_all.columns)}")

    required = (
        DEPENDENT,
        "chinese",
        "age",
        "gender",
        "edu",
        "nmlrlincome",
        "mnincome",
        "market",
        "south",
        "headnowork",
        "rainfall",
    )
    spec1_required = (
        DEPENDENT,
        "chinese",
        "age",
        "gender",
        "edu",
        SPEC1_ENDOGENOUS,
        "market",
        "south",
        "headnowork",
        "rainfall",
    )
    missing_columns = sorted(
        (set(required) | set(spec1_required)) - set(risk_all.columns)
    )
    if missing_columns:
        raise ReplicationError(f"Required variables missing: {missing_columns}")

    missingness = risk_all.loc[:, required].isna().sum()
    print("\nSpecification 2 missingness:")
    for variable, count in missingness.items():
        print(f"  {variable:<15} {int(count):>3}")
    print(
        "  Note: complete cases are defined only over Specification 2 variables; "
        "unrelated lambda variables contain missing values."
    )

    data = risk_all.dropna(subset=list(required)).copy()
    if len(data) != EXPECTED_N:
        raise ReplicationError(
            f"Expected N={EXPECTED_N} complete cases but obtained N={len(data)}. "
            "Check listwise deletion and the supplied risk dataset."
        )
    spec1_index = risk_all.dropna(subset=list(spec1_required)).index
    if len(spec1_index) != EXPECTED_N or not spec1_index.equals(data.index):
        raise ReplicationError(
            "Specification 1 and Specification 2 do not use the same expected "
            "181 observations; diagnose sample construction before Figure 6"
        )
    print("Specification 1 complete-case sample N = 181 (same observations)")
    for variable in set(required) | {SPEC1_ENDOGENOUS}:
        data[variable] = pd.to_numeric(data[variable], errors="raise").astype(float)
    data["constant"] = 1.0

    print(f"\nComplete-case estimation sample N = {len(data)} (expected 181)")
    return data.reset_index(drop=True)


def build_tcn_specification() -> None:
    print("\nTCN / GKM SPECIFICATION 2")
    print("=========================")
    print(f"Dependent variable:       {DEPENDENT}")
    print(f"Included exogenous:       {list(INCLUDED_EXOGENOUS)}")
    print(f"Endogenous regressors:    {list(ENDOGENOUS)}")
    print(f"Excluded instruments:     {list(EXCLUDED_INSTRUMENTS)}")
    print("Weights / robust / cluster: none")
    print("Sample qualifier:         none (exact complete cases above)")


def two_stage_least_squares(data: pd.DataFrame) -> TwoSLSResult:
    """Conventional exactly-identified 2SLS and GKM Table 2 Wald intervals."""

    y = data[DEPENDENT].to_numpy(dtype=float)
    x_names = list(INCLUDED_EXOGENOUS + ENDOGENOUS)
    z_names = list(INCLUDED_EXOGENOUS + EXCLUDED_INSTRUMENTS)
    x = data[x_names].to_numpy(dtype=float)
    z = data[z_names].to_numpy(dtype=float)

    if matrix_rank(x) != x.shape[1]:
        raise ReplicationError("Structural regressor matrix is not full column rank")
    qz, rank_z = orthonormal_basis(z, "full 2SLS instrument matrix")
    if rank_z != z.shape[1]:
        raise ReplicationError("Full 2SLS instrument matrix is not full column rank")

    x_hat = qz @ (qz.T @ x)
    coefficients, _, estimated_rank, _ = linalg.lstsq(
        x_hat, y, cond=None, lapack_driver="gelsy"
    )
    if estimated_rank != x.shape[1]:
        raise ReplicationError("Projected 2SLS regressor matrix is rank deficient")

    # Independent exactly-identified cross-check, used only as a diagnostic.
    exactly_identified = linalg.solve(z.T @ x, z.T @ y)
    coefficient_check = float(np.max(np.abs(coefficients - exactly_identified)))
    if coefficient_check > 1e-9:
        raise ReplicationError(
            "QR-projection and exactly-identified 2SLS estimates disagree: "
            f"{coefficient_check:.3e}"
        )

    residuals = y - x @ coefficients
    cross = x_hat.T @ x_hat
    try:
        factor = linalg.cho_factor(cross, lower=True, check_finite=True)
        bread = linalg.cho_solve(factor, np.eye(cross.shape[0]))
    except linalg.LinAlgError as exc:
        raise ReplicationError("2SLS covariance bread is not positive definite") from exc

    # IMPORTANT GKM TABLE 2 REPLICATION CONVENTION
    # ------------------------------------------------------------
    # The official EmpiricalApplication.ox computes the W endpoints with the
    # homoskedastic variance scale u'u/(N-columns(mZ)), where mZ contains the
    # excluded instruments, and the asymptotic N(0,1) critical value.  Thus the
    # Specification 2 divisor is N-2=179.  This is the GKM's Table 2 Ox-code
    # convention, not a general 2SLS degrees-of-freedom rule; legacy Stata 9
    # ivreg instead used its small-sample t convention u'u/(N-K), K=9.  The
    # implementation below is checked against all nine GKM Table 2 W intervals.
    n_obs = len(data)
    gkm_variance_df = n_obs - len(EXCLUDED_INSTRUMENTS)
    sigma2_gkm = float(residuals.T @ residuals) / gkm_variance_df
    covariance = sigma2_gkm * bread
    standard_errors_array = np.sqrt(np.diag(covariance))
    z_critical = float(norm.ppf(1.0 - ALPHA / 2.0))

    coefficient_dict = dict(zip(x_names, coefficients, strict=True))
    standard_error_dict = dict(zip(x_names, standard_errors_array, strict=True))
    wald_intervals = {
        name: Interval(
            float(coefficient_dict[name] - z_critical * standard_error_dict[name]),
            float(coefficient_dict[name] + z_critical * standard_error_dict[name]),
        )
        for name in x_names
    }

    print("\n2SLS numerical diagnostics:")
    print(f"  rank(X)                         = {matrix_rank(x)} / {x.shape[1]}")
    print(f"  rank(full instrument matrix)    = {rank_z} / {z.shape[1]}")
    print(f"  QR versus exact-ID max |diff|   = {coefficient_check:.3e}")
    print(f"  structural residual sum squares = {residuals.T @ residuals:.12g}")
    print(
        "  official Ox W variance divisor   = "
        f"N - #excluded IVs = {gkm_variance_df}"
    )
    print(f"  legacy Stata small-sample df     = N - 9 = {n_obs - x.shape[1]}")

    return TwoSLSResult(
        coefficients={name: float(value) for name, value in coefficient_dict.items()},
        standard_errors={
            name: float(value) for name, value in standard_error_dict.items()
        },
        wald_intervals=wald_intervals,
        residuals=residuals,
        x=x,
        z=z,
        x_hat=x_hat,
        gkm_wald_variance_df=gkm_variance_df,
        stata_small_sample_df=n_obs - x.shape[1],
    )


def first_stage_diagnostics(data: pd.DataFrame) -> pd.DataFrame:
    """Report the original overall first-stage Fs and excluded-IV partial Fs."""

    unrestricted_names = list(INCLUDED_EXOGENOUS + EXCLUDED_INSTRUMENTS)
    restricted_names = list(INCLUDED_EXOGENOUS)
    xu = data[unrestricted_names].to_numpy(dtype=float)
    xr = data[restricted_names].to_numpy(dtype=float)
    n_obs = len(data)
    q_excluded = len(EXCLUDED_INSTRUMENTS)
    df_u = n_obs - xu.shape[1]
    rows: list[dict[str, float | str]] = []

    for variable in ENDOGENOUS:
        outcome = data[variable].to_numpy(dtype=float)
        coef_u, _, _, _ = linalg.lstsq(xu, outcome, lapack_driver="gelsy")
        coef_r, _, _, _ = linalg.lstsq(xr, outcome, lapack_driver="gelsy")
        resid_u = outcome - xu @ coef_u
        resid_r = outcome - xr @ coef_r
        rss_u = float(resid_u.T @ resid_u)
        rss_r = float(resid_r.T @ resid_r)
        tss = float(((outcome - outcome.mean()) ** 2).sum())
        r_squared = 1.0 - rss_u / tss
        df_model = xu.shape[1] - 1
        overall_f = (r_squared / df_model) / ((1.0 - r_squared) / df_u)
        partial_f = ((rss_r - rss_u) / q_excluded) / (rss_u / df_u)
        rows.append(
            {
                "variable": variable,
                "r_squared": r_squared,
                "overall_first_stage_f": overall_f,
                "excluded_iv_partial_f": partial_f,
            }
        )

    result = pd.DataFrame(rows)
    print("\nDescriptive first-stage diagnostics:")
    print(result.to_string(index=False, float_format=lambda value: f"{value:.8f}"))
    return result


def validate_2sls_and_wald(two_sls: TwoSLSResult) -> None:
    """Stop unless estimates and W endpoints reproduce the printed benchmarks."""

    comparison_rows: list[dict[str, float | str]] = []
    max_estimate_difference = 0.0
    max_wald_difference = 0.0
    for variable in ROW_ORDER:
        benchmark = GKM_REPORTED[variable]
        estimate = two_sls.coefficients[variable]
        interval = two_sls.wald_intervals[variable]
        estimate_difference = abs(estimate - float(benchmark["estimate"]))
        lower_difference = abs(interval.lower - benchmark["W"][0])
        upper_difference = abs(interval.upper - benchmark["W"][1])
        max_estimate_difference = max(max_estimate_difference, estimate_difference)
        max_wald_difference = max(
            max_wald_difference, lower_difference, upper_difference
        )
        comparison_rows.append(
            {
                "variable": DISPLAY_NAMES[variable],
                "GKM estimate": benchmark["estimate"],
                "Python estimate": estimate,
                "|estimate diff|": estimate_difference,
                "GKM W lower": benchmark["W"][0],
                "Python W lower": interval.lower,
                "|lower diff|": lower_difference,
                "GKM W upper": benchmark["W"][1],
                "Python W upper": interval.upper,
                "|upper diff|": upper_difference,
            }
        )

    comparison = pd.DataFrame(comparison_rows)
    print("\n2SLS AND WALD BENCHMARK COMPARISON")
    print("==================================")
    print(comparison.to_string(index=False, float_format=lambda value: f"{value:.9f}"))
    print(f"Maximum estimate discrepancy = {max_estimate_difference:.6g}")
    print(f"Maximum W endpoint discrepancy = {max_wald_difference:.6g}")

    if max_estimate_difference > EXPECTED_POINT_TOLERANCE:
        raise ReplicationError(
            "2SLS estimates fail to reproduce GKM Table 2 at printed precision"
        )
    if max_wald_difference > EXPECTED_WALD_TOLERANCE:
        raise ReplicationError(
            "Wald intervals fail to reproduce GKM Table 2 at printed precision"
        )
    print("2SLS point estimates and W intervals replicate the GKM printed values.")


def build_test_problem(data: pd.DataFrame, variable: str) -> TestProblem:
    """Build GKM Algorithm 1 or Algorithm 2 for one tested coefficient."""

    if variable in INCLUDED_EXOGENOUS:
        case = "exogenous"
        controls = tuple(name for name in INCLUDED_EXOGENOUS if name != variable)
        nuisance = ENDOGENOUS
        instruments = (variable,) + EXCLUDED_INSTRUMENTS
        expected_rank_c = 6
        expected_n = 175
        expected_k = 3
        expected_m_w = 2
    elif variable in ENDOGENOUS:
        case = "endogenous"
        controls = INCLUDED_EXOGENOUS
        nuisance = tuple(name for name in ENDOGENOUS if name != variable)
        instruments = EXCLUDED_INSTRUMENTS
        expected_rank_c = 7
        expected_n = 174
        expected_k = 2
        expected_m_w = 1
    else:  # pragma: no cover - configuration guard
        raise ValueError(f"Unknown tested coefficient: {variable}")

    stacked_names = (DEPENDENT, variable) + nuisance + instruments
    stacked = data[list(stacked_names)].to_numpy(dtype=float)
    c = data[list(controls)].to_numpy(dtype=float)
    residualized, rank_c = residualize(stacked, c)

    m_w = len(nuisance)
    y = residualized[:, 0]
    tested = residualized[:, 1]
    w = residualized[:, 2 : 2 + m_w]
    z = residualized[:, 2 + m_w :]
    qz, rank_z = orthonormal_basis(z, f"residualized Z for {variable}")
    k = z.shape[1]
    effective_n = len(data) - rank_c
    denominator_df = effective_n - k

    if rank_c != expected_rank_c:
        raise ReplicationError(
            f"{variable}: rank(C)={rank_c}, expected {expected_rank_c}"
        )
    if effective_n != expected_n:
        raise ReplicationError(
            f"{variable}: effective n={effective_n}, expected {expected_n}"
        )
    if k != expected_k or rank_z != k:
        raise ReplicationError(
            f"{variable}: residualized Z rank={rank_z}/{k}, expected k={expected_k}"
        )
    if m_w != expected_m_w or k - m_w != 1:
        raise ReplicationError(
            f"{variable}: m_W={m_w}, k-m_W={k-m_w}; expected df=1"
        )
    if denominator_df <= 0:
        raise ReplicationError(f"{variable}: nonpositive GKM denominator n-k")

    # The square first-stage coefficient matrix is useful here because p=k in
    # both algorithms.  Full rank rules out an exact reduced-rank degeneracy.
    first_stage_rhs = np.column_stack([tested, w])
    first_stage_coefficients, _, _, _ = linalg.lstsq(
        z, first_stage_rhs, lapack_driver="gelsy"
    )
    first_stage_rank = matrix_rank(first_stage_coefficients)
    if first_stage_rank != first_stage_coefficients.shape[1]:
        raise ReplicationError(
            f"{variable}: first-stage coefficient matrix is rank deficient "
            f"({first_stage_rank}/{first_stage_coefficients.shape[1]})"
        )

    return TestProblem(
        variable=variable,
        display_name=DISPLAY_NAMES[variable],
        case=case,
        y=y,
        tested_regressor=tested,
        nuisance_w=w,
        residualized_z=z,
        qz=qz,
        original_n=len(data),
        rank_c=rank_c,
        effective_n=effective_n,
        k=k,
        m_w=m_w,
        denominator_df=denominator_df,
        controls=controls,
        instruments=instruments,
        first_stage_rank=first_stage_rank,
    )


def build_gender_spec1_problem(data: pd.DataFrame) -> TestProblem:
    """Build Algorithm 1 for the Gender panel in GKM Figure 6.

    This is deliberately separate from Specification 2.  Here W=[income], so
    m_W=1, p=2, k=3, and k-m_W=2.  Consequently the feasible AR statistic is
    kappa-hat_2n and both GKM and HW condition on kappa-hat_1n.
    """

    controls = ("chinese", "age", "edu", "market", "south", "constant")
    instruments = ("gender",) + EXCLUDED_INSTRUMENTS
    stacked_names = (DEPENDENT, "gender", SPEC1_ENDOGENOUS) + instruments
    stacked = data[list(stacked_names)].to_numpy(dtype=float)
    c = data[list(controls)].to_numpy(dtype=float)
    residualized, rank_c = residualize(stacked, c)

    y = residualized[:, 0]
    tested = residualized[:, 1]
    w = residualized[:, 2:3]
    z = residualized[:, 3:]
    qz, rank_z = orthonormal_basis(z, "Gender Specification 1 residualized Z")
    effective_n = len(data) - rank_c
    k = z.shape[1]
    denominator_df = effective_n - k

    if rank_c != 6 or effective_n != 175:
        raise ReplicationError(
            "Gender Specification 1 must have rank(C)=6 and effective n=175"
        )
    if k != 3 or rank_z != 3 or w.shape[1] != 1 or k - w.shape[1] != 2:
        raise ReplicationError(
            "Gender Specification 1 must have k=3, m_W=1, and k-m_W=2"
        )
    if denominator_df != 172:
        raise ReplicationError(
            f"Gender Specification 1 has n-k={denominator_df}, expected 172"
        )

    first_stage_rhs = np.column_stack([tested, w])
    first_stage_coefficients, _, _, _ = linalg.lstsq(
        z, first_stage_rhs, lapack_driver="gelsy"
    )
    first_stage_rank = matrix_rank(first_stage_coefficients)
    if first_stage_rank != 2:
        raise ReplicationError(
            "Gender Specification 1 first-stage coefficient matrix is rank deficient"
        )

    return TestProblem(
        variable="gender_spec1",
        display_name="Gender (Specification 1)",
        case="exogenous-specification-1",
        y=y,
        tested_regressor=tested,
        nuisance_w=w,
        residualized_z=z,
        qz=qz,
        original_n=len(data),
        rank_c=rank_c,
        effective_n=effective_n,
        k=k,
        m_w=1,
        denominator_df=denominator_df,
        controls=controls,
        instruments=instruments,
        first_stage_rank=first_stage_rank,
    )


def gender_spec1_two_sls(data: pd.DataFrame) -> tuple[float, float]:
    """Return the Specification 1 Gender 2SLS estimate and GKM-style SE."""

    x_names = list(INCLUDED_EXOGENOUS) + [SPEC1_ENDOGENOUS]
    z_names = list(INCLUDED_EXOGENOUS) + list(EXCLUDED_INSTRUMENTS)
    y = data[DEPENDENT].to_numpy(dtype=float)
    x = data[x_names].to_numpy(dtype=float)
    z = data[z_names].to_numpy(dtype=float)
    qz, rank_z = orthonormal_basis(z, "Specification 1 full instrument matrix")
    if rank_z != z.shape[1]:
        raise ReplicationError("Specification 1 instrument matrix is rank deficient")
    x_hat = qz @ (qz.T @ x)
    coefficients, _, rank_x_hat, _ = linalg.lstsq(
        x_hat, y, lapack_driver="gelsy"
    )
    if rank_x_hat != x.shape[1]:
        raise ReplicationError("Specification 1 projected X is rank deficient")
    residuals = y - x @ coefficients
    bread = linalg.cho_solve(
        linalg.cho_factor(x_hat.T @ x_hat, lower=True),
        np.eye(x.shape[1]),
    )
    sigma2 = float(residuals.T @ residuals) / (
        len(data) - len(EXCLUDED_INSTRUMENTS)
    )
    standard_errors = np.sqrt(np.diag(sigma2 * bread))
    index = x_names.index("gender")
    estimate = float(coefficients[index])
    standard_error = float(standard_errors[index])
    if abs(estimate - GKM_GENDER_SPEC1_REPORTED["estimate"]) > 5.01e-4:
        raise ReplicationError(
            "Specification 1 Gender estimate does not reproduce GKM Table 2"
        )
    return estimate, standard_error


def projection_crossproducts(
    problem: TestProblem, beta0: float
) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute Q'P_ZQ and Q'M_ZQ without forming either projection matrix."""

    null_outcome = problem.y - problem.tested_regressor * float(beta0)
    q = np.column_stack([null_outcome, problem.nuisance_w])
    projected = problem.qz @ (problem.qz.T @ q)
    residual = q - projected
    ess = projected.T @ projected
    rss = residual.T @ residual
    decomposition_error = float(np.max(np.abs(q.T @ q - ess - rss)))
    return ess, rss, decomposition_error


def gkm_eigenvalues(problem: TestProblem, beta0: float) -> np.ndarray:
    """Exact feasible roots of ESS v=lambda [RSS/(n-k)] v."""

    ess, rss, _ = projection_crossproducts(problem, beta0)
    denominator = rss / problem.denominator_df
    try:
        linalg.cholesky(denominator, lower=True, check_finite=True)
    except linalg.LinAlgError as exc:
        eigenvalues = linalg.eigvalsh(denominator)
        raise ReplicationError(
            f"{problem.variable}, beta0={beta0:.12g}: RSS/(n-k) is not "
            f"positive definite; eigenvalues={eigenvalues}"
        ) from exc

    roots = linalg.eigh(
        ess, denominator, eigvals_only=True, check_finite=True
    )[::-1]
    scale = max(1.0, float(np.max(np.abs(roots))))
    if float(np.min(roots)) < -EIGEN_ABS_TOLERANCE * scale:
        raise ReplicationError(
            f"{problem.variable}, beta0={beta0:.12g}: materially negative "
            f"generalized eigenvalue(s): {roots}"
        )
    roots = np.maximum(roots, 0.0)
    if not np.all(np.isfinite(roots)):
        raise ReplicationError(f"Nonfinite roots for {problem.variable}")
    if np.any(np.diff(roots) > EIGEN_ABS_TOLERANCE * scale):
        raise ReplicationError(f"Roots not ordered nonincreasingly: {roots}")
    return roots


def whitened_eigenvalues(problem: TestProblem, beta0: float) -> np.ndarray:
    """Independent Cholesky-whitened cross-check of the generalized roots."""

    ess, rss, _ = projection_crossproducts(problem, beta0)
    denominator = rss / problem.denominator_df
    lower = linalg.cholesky(denominator, lower=True, check_finite=True)
    lower_inverse = linalg.solve_triangular(
        lower, np.eye(lower.shape[0]), lower=True, check_finite=True
    )
    whitened = lower_inverse @ ess @ lower_inverse.T
    whitened = (whitened + whitened.T) / 2.0
    return np.maximum(linalg.eigvalsh(whitened)[::-1], 0.0)


def validate_eigenvalues(
    problem: TestProblem, beta_values: Iterable[float]
) -> tuple[float, float]:
    """Validate generalized roots against whitening and Q'Q decomposition."""

    max_root_discrepancy = 0.0
    max_decomposition_error = 0.0
    for beta0 in beta_values:
        generalized = gkm_eigenvalues(problem, beta0)
        whitened = whitened_eigenvalues(problem, beta0)
        discrepancy = float(np.max(np.abs(generalized - whitened)))
        max_root_discrepancy = max(max_root_discrepancy, discrepancy)
        _, _, decomposition_error = projection_crossproducts(problem, beta0)
        max_decomposition_error = max(
            max_decomposition_error, decomposition_error
        )
    if max_root_discrepancy > WHITENING_TOLERANCE:
        raise ReplicationError(
            f"{problem.variable}: generalized and whitened roots disagree by "
            f"{max_root_discrepancy:.3e}"
        )
    return max_root_discrepancy, max_decomposition_error


def official_gkm_cv_table(df: int, alpha: float) -> np.ndarray:
    """Return one exact official table as [conditioning statistic, CV]."""

    alpha_key = round(float(alpha), 2)
    if df == 1 and alpha_key == 0.05:
        return np.column_stack([GKM_CV_KAPPA_DF1_5PCT, GKM_CV_95_DF1])
    if df == 2 and alpha_key in GKM_CV_TABLES_DF2:
        return GKM_CV_TABLES_DF2[alpha_key]
    raise NotImplementedError(
        f"No embedded official GKM table for df={df}, alpha={alpha_key}"
    )


def gkm_conditional_cv(
    conditioning_statistic: float, df: int = 1, alpha: float = ALPHA
) -> float:
    """Official GKM conditional-CV interpolation and upper-tail rule."""

    table = official_gkm_cv_table(df, alpha)
    value = float(conditioning_statistic)
    if math.isnan(value):
        raise ValueError("Conditioning statistic is NaN")
    if value < -EIGEN_ABS_TOLERANCE:
        raise ValueError(f"Conditioning statistic is negative: {value}")
    value = max(value, 0.0)
    if math.isinf(value):
        return float(table[-1, 1])
    if value <= 1000.0:
        return float(
            np.interp(
                value,
                table[:-1, 0],
                table[:-1, 1],
            )
        )

    # Exact upper-tail rule in the official interpolate.ox: exponential CDF
    # probexp(value-1000, 1), implemented stably with -expm1(-x).
    weight = -math.expm1(-(value - 1000.0))
    return float(
        table[-2, 1]
        + weight * (table[-1, 1] - table[-2, 1])
    )


def validate_critical_value_table() -> None:
    """Check official df=1 and df=2 tables, interpolation, and limits."""

    exact_checks = {
        0.0: 0.0,
        0.5: 0.4,
        2.3: 1.6,
        9.8: 3.4,
        29.7: 3.8,
        1000.0: 3.83761994673704,
    }
    for conditioning, expected in exact_checks.items():
        obtained = gkm_conditional_cv(conditioning, 1)
        if abs(obtained - expected) > 5e-13:
            raise ReplicationError(
                f"Critical-value knot failed at {conditioning}: "
                f"{obtained} versus {expected}"
            )
    interpolation_check = gkm_conditional_cv(0.25, 1)
    if abs(interpolation_check - 0.2) > 5e-13:
        raise ReplicationError("Critical-value interpolation from (0,0) failed")
    chi_square_limit = float(chi2.ppf(0.95, 1))
    if abs(GKM_CV_95_DF1[-1] - chi_square_limit) > 1e-9:
        raise ReplicationError("GKM infinity knot disagrees with chi-square_1")
    evaluation_grid = np.r_[np.linspace(0.0, 100.0, 2001), 1000.0, 1001.0]
    values = np.array([gkm_conditional_cv(value, 1) for value in evaluation_grid])
    if np.any(np.diff(values) < -1e-13):
        raise ReplicationError("GKM critical-value interpolation is not monotone")

    df2_checks = {
        0.10: (47.2, 4.6, 4.60517018598809),
        0.05: (35.4, 5.9, 5.99146454710798),
        0.01: (89.0, 9.2, 9.21034037197618),
    }
    for alpha, (conditioning, expected, expected_limit) in df2_checks.items():
        obtained = gkm_conditional_cv(conditioning, 2, alpha)
        if abs(obtained - expected) > 5e-13:
            raise ReplicationError(
                f"df=2 critical-value knot failed for alpha={alpha}: "
                f"{obtained} versus {expected}"
            )
        obtained_limit = gkm_conditional_cv(math.inf, 2, alpha)
        chi_square_limit = float(chi2.ppf(1.0 - alpha, 2))
        if max(
            abs(obtained_limit - expected_limit),
            abs(obtained_limit - chi_square_limit),
        ) > 1e-9:
            raise ReplicationError(
                f"df=2 infinity knot failed for alpha={alpha}"
            )
        evaluation_grid = np.r_[np.linspace(0.0, 100.0, 2001), 1000.0, 1001.0]
        df2_values = np.array(
            [gkm_conditional_cv(value, 2, alpha) for value in evaluation_grid]
        )
        if np.any(np.diff(df2_values) < -1e-13):
            raise ReplicationError(
                f"df=2 interpolation is not monotone for alpha={alpha}"
            )

    print("\nGKM critical-value table validation:")
    print("  alpha                         = 0.05")
    print("  k - m_W                       = 1")
    print(f"  number of official knots      = {len(GKM_CV_KAPPA_DF1_5PCT)}")
    print(f"  c_0.95(0.25, 1)               = {interpolation_check:.12g}")
    print(f"  c_0.95(1000, 1)               = {GKM_CV_95_DF1[-2]:.15g}")
    print(f"  chi-square_1 95% limit         = {GKM_CV_95_DF1[-1]:.15g}")
    print("  k-m_W=2 tables                = alpha 0.01, 0.05, 0.10")
    print(
        "  chi-square_2 95% limit         = "
        f"{gkm_conditional_cv(math.inf, 2, 0.05):.15g}"
    )
    print("  exact knots / interpolation / tail checks: PASS")


def test_at_beta0(problem: TestProblem, beta0: float) -> dict[str, float | bool]:
    """Evaluate GKM C, HW CHW, and unconditional U tests at one null."""

    roots = gkm_eigenvalues(problem, beta0)
    ar_statistic = float(roots[-1])
    conditioning_gkm = float(roots[0])
    # With descending roots, roots[-2] is the second-smallest.  For p=2 this
    # is roots[0], so the GKM and HW conditioning statistics coincide exactly.
    conditioning_hw = float(roots[-2])
    cv_gkm = gkm_conditional_cv(conditioning_gkm, problem.k - problem.m_w)
    cv_hw = gkm_conditional_cv(conditioning_hw, problem.k - problem.m_w)
    cv_u = float(chi2.ppf(1.0 - ALPHA, problem.k - problem.m_w))

    if cv_hw > cv_gkm + 2e-12:
        raise ReplicationError(
            f"{problem.variable}, beta0={beta0:.12g}: CV_CHW > CV_GKM"
        )
    if problem.m_w == 1:
        if abs(conditioning_gkm - conditioning_hw) > 1e-12:
            raise ReplicationError(
                f"{problem.variable}: p=2 conditioning statistics differ"
            )
        if abs(cv_gkm - cv_hw) > 1e-12:
            raise ReplicationError(f"{problem.variable}: p=2 C and CHW CVs differ")

    return {
        "beta0": float(beta0),
        "kappa_hat_1n": float(roots[0]),
        "kappa_hat_2n": float(roots[1]),
        "kappa_hat_3n": float(roots[2]) if roots.size == 3 else np.nan,
        "ar_statistic": ar_statistic,
        "gkm_conditioning_kappa_hat": conditioning_gkm,
        "hw_conditioning_kappa_hat": conditioning_hw,
        "cv_gkm": cv_gkm,
        "cv_chw": cv_hw,
        "cv_u": cv_u,
        "ar_minus_cv_gkm": ar_statistic - cv_gkm,
        "ar_minus_cv_chw": ar_statistic - cv_hw,
        "ar_minus_cv_u": ar_statistic - cv_u,
        "reject_gkm": bool(ar_statistic > cv_gkm),
        "reject_chw": bool(ar_statistic > cv_hw),
        "reject_u": bool(ar_statistic > cv_u),
        "conditioning_kappa_hat_gap": conditioning_gkm - conditioning_hw,
        "cv_gap": cv_gkm - cv_hw,
    }


def method_margin(evaluation: dict[str, float | bool], method: str) -> float:
    column = {
        "C": "ar_minus_cv_gkm",
        "CHW": "ar_minus_cv_chw",
        "U": "ar_minus_cv_u",
    }[method]
    return float(evaluation[column])


def invert_test(
    problem: TestProblem,
    method: str,
    center: float,
    standard_error: float,
) -> Interval:
    """Invert a test with adaptive bracketing and Brent root refinement."""

    radius = max(0.05, 10.0 * standard_error)
    for _ in range(9):
        lower_search = center - radius
        upper_search = center + radius
        search_grid = np.linspace(lower_search, upper_search, 2401)
        margins = np.array(
            [method_margin(test_at_beta0(problem, beta), method) for beta in search_grid]
        )
        transition_indices = np.flatnonzero(margins[:-1] * margins[1:] < 0.0)
        tails_reject = margins[0] > 0.0 and margins[-1] > 0.0
        if tails_reject and transition_indices.size == 2:
            break
        radius *= 2.0
    else:
        raise ReplicationError(
            f"Could not isolate one bounded {method} confidence interval for "
            f"{problem.variable}; inspect the search range and test topology"
        )

    roots: list[float] = []
    for index in transition_indices:
        left = float(search_grid[index])
        right = float(search_grid[index + 1])
        root = brentq(
            lambda beta: method_margin(test_at_beta0(problem, beta), method),
            left,
            right,
            xtol=ROOT_TOLERANCE,
            rtol=ROOT_TOLERANCE,
            maxiter=200,
        )
        roots.append(float(root))

    interval = Interval(roots[0], roots[1])
    if method_margin(test_at_beta0(problem, center), method) > 1e-9:
        raise ReplicationError(
            f"{problem.variable}: 2SLS estimate is rejected by its {method} test"
        )
    return interval


def replicate_gkm_row(
    problem: TestProblem, two_sls: TwoSLSResult
) -> RowResult:
    """Replicate C and U only; CHW is intentionally deferred."""

    variable = problem.variable
    center = two_sls.coefficients[variable]
    standard_error = two_sls.standard_errors[variable]
    conditional = invert_test(problem, "C", center, standard_error)
    unconditional = invert_test(problem, "U", center, standard_error)
    return RowResult(
        variable=variable,
        display_name=problem.display_name,
        estimate=center,
        wald=two_sls.wald_intervals[variable],
        conditional_gkm=conditional,
        unconditional=unconditional,
    )


def validate_gkm_intervals(rows: list[RowResult]) -> tuple[float, float]:
    """Stop before CHW unless published C and U intervals are replicated."""

    comparison_rows: list[dict[str, float | str]] = []
    max_c_difference = 0.0
    max_u_difference = 0.0
    for row in rows:
        benchmark = GKM_REPORTED[row.variable]
        c_diffs = (
            row.conditional_gkm.lower - benchmark["C"][0],
            row.conditional_gkm.upper - benchmark["C"][1],
        )
        u_diffs = (
            row.unconditional.lower - benchmark["U"][0],
            row.unconditional.upper - benchmark["U"][1],
        )
        max_c_difference = max(max_c_difference, *(abs(value) for value in c_diffs))
        max_u_difference = max(max_u_difference, *(abs(value) for value in u_diffs))
        comparison_rows.append(
            {
                "variable": row.display_name,
                "published C lower": benchmark["C"][0],
                "replicated C lower": row.conditional_gkm.lower,
                "C lower diff": c_diffs[0],
                "published C upper": benchmark["C"][1],
                "replicated C upper": row.conditional_gkm.upper,
                "C upper diff": c_diffs[1],
                "published U lower": benchmark["U"][0],
                "replicated U lower": row.unconditional.lower,
                "U lower diff": u_diffs[0],
                "published U upper": benchmark["U"][1],
                "replicated U upper": row.unconditional.upper,
                "U upper diff": u_diffs[1],
            }
        )

    comparison = pd.DataFrame(comparison_rows)
    print("\nGKM C AND U ENDPOINT COMPARISON")
    print("===============================")
    print(comparison.to_string(index=False, float_format=lambda value: f"{value:.9f}"))
    print(f"Maximum C endpoint discrepancy = {max_c_difference:.9g}")
    print(f"Maximum U endpoint discrepancy = {max_u_difference:.9g}")

    if max_c_difference > EXPECTED_AR_ENDPOINT_TOLERANCE:
        raise ReplicationError(
            "GKM conditional C intervals fail to reproduce Table 2 within "
            "one published 0.001 grid unit. CHW will not be computed."
        )
    if max_u_difference > EXPECTED_AR_ENDPOINT_TOLERANCE:
        raise ReplicationError(
            "GKM unconditional U intervals fail to reproduce Table 2 within "
            "one published 0.001 grid unit. CHW will not be computed."
        )

    print(
        "GKM C and U replication PASS: every root-refined endpoint is within "
        "0.001 of the GKM's printed finite-grid endpoint."
    )
    return max_c_difference, max_u_difference


def compute_chw_row(
    problem: TestProblem, row: RowResult, standard_error: float
) -> None:
    """Compute CHW only after the caller has validated all GKM rows."""

    row.conditional_hw = invert_test(
        problem, "CHW", row.estimate, standard_error
    )
    chw = row.conditional_hw
    c_interval = row.conditional_gkm
    if chw.lower < c_interval.lower - 2e-9 or chw.upper > c_interval.upper + 2e-9:
        raise ReplicationError(f"{row.variable}: CHW is not a subset of GKM C")
    if problem.m_w == 1:
        difference = max(
            abs(chw.lower - c_interval.lower),
            abs(chw.upper - c_interval.upper),
        )
        if difference > 2e-10:
            raise ReplicationError(
                f"{row.variable}: CHW != C despite m_W=1; diff={difference:.3e}"
            )


def safe_slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def run_beta_grid(
    problem: TestProblem, row: RowResult
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Save a focused, auditable grid and collect boundary diagnostics."""

    if row.conditional_hw is None:  # pragma: no cover - workflow guard
        raise ReplicationError("CHW must be computed before the audit grid")
    intervals = (
        row.wald,
        row.conditional_gkm,
        row.conditional_hw,
        row.unconditional,
    )
    lower = min(interval.lower for interval in intervals)
    upper = max(interval.upper for interval in intervals)
    span = upper - lower
    margin = max(0.10 * span, 1e-5)
    base_grid = np.linspace(lower - margin, upper + margin, AUDIT_GRID_SIZE)
    anchors = np.array(
        [
            0.0,
            row.estimate,
            *(endpoint for interval in intervals for endpoint in (interval.lower, interval.upper)),
        ]
    )
    beta_grid = np.unique(np.r_[base_grid, anchors])
    records = [test_at_beta0(problem, beta0) for beta0 in beta_grid]
    frame = pd.DataFrame(records).sort_values("beta0").reset_index(drop=True)

    if np.any(frame["cv_chw"] > frame["cv_gkm"] + 2e-12):
        raise ReplicationError(f"{problem.variable}: grid contains CV_CHW > CV_GKM")
    if problem.m_w == 1:
        if not np.allclose(
            frame["gkm_conditioning_kappa_hat"],
            frame["hw_conditioning_kappa_hat"],
            atol=1e-12,
        ):
            raise ReplicationError(f"{problem.variable}: p=2 conditioning grid mismatch")
        if not np.allclose(frame["cv_gkm"], frame["cv_chw"], atol=1e-12):
            raise ReplicationError(f"{problem.variable}: p=2 critical-value grid mismatch")

    grid_path = GRID_DIR / f"{safe_slug(row.display_name)}_beta_grid.csv"
    frame.to_csv(grid_path, index=False, float_format="%.15g")

    boundary_records: list[dict[str, float | str]] = []
    for method, interval in (
        ("C", row.conditional_gkm),
        ("CHW", row.conditional_hw),
    ):
        for side, beta0 in (("lower", interval.lower), ("upper", interval.upper)):
            evaluation = test_at_beta0(problem, beta0)
            boundary_records.append(
                {
                    "variable": row.display_name,
                    "method": method,
                    "side": side,
                    **evaluation,
                }
            )
    boundary_frame = pd.DataFrame(boundary_records)
    summary = {
        "grid_beta_min": float(frame["beta0"].min()),
        "grid_beta_max": float(frame["beta0"].max()),
        "max_conditioning_kappa_hat_gap": float(
            frame["conditioning_kappa_hat_gap"].max()
        ),
        "median_conditioning_kappa_hat_gap": float(
            frame["conditioning_kappa_hat_gap"].median()
        ),
        "max_cv_gap": float(frame["cv_gap"].max()),
        "median_cv_gap": float(frame["cv_gap"].median()),
    }
    return frame, boundary_frame, summary


def interval_text(interval: Interval, digits: int = 3) -> str:
    return f"[{interval.lower:.{digits}f}, {interval.upper:.{digits}f}]"


def make_table(rows: list[RowResult]) -> pd.DataFrame:
    """Write full-precision machine output and a GKM-style Markdown table."""

    records: list[dict[str, float | str | bool]] = []
    for row in rows:
        if row.conditional_hw is None:  # pragma: no cover - workflow guard
            raise ReplicationError(f"CHW missing for {row.variable}")
        published = GKM_REPORTED[row.variable]
        chw = row.conditional_hw
        records.append(
            {
                "variable": row.variable,
                "display_name": row.display_name,
                "estimate": row.estimate,
                "wald_lower": row.wald.lower,
                "wald_upper": row.wald.upper,
                "gkm_c_lower": row.conditional_gkm.lower,
                "gkm_c_upper": row.conditional_gkm.upper,
                "hw_chw_lower": chw.lower,
                "hw_chw_upper": chw.upper,
                "unconditional_u_lower": row.unconditional.lower,
                "unconditional_u_upper": row.unconditional.upper,
                "c_length": row.conditional_gkm.length,
                "chw_length": chw.length,
                "length_reduction": row.conditional_gkm.length - chw.length,
                "percent_length_reduction": 100.0
                * (row.conditional_gkm.length - chw.length)
                / row.conditional_gkm.length,
                "chw_equals_c": bool(
                    max(
                        abs(chw.lower - row.conditional_gkm.lower),
                        abs(chw.upper - row.conditional_gkm.upper),
                    )
                    <= 2e-10
                ),
                "published_estimate": published["estimate"],
                "published_w_lower": published["W"][0],
                "published_w_upper": published["W"][1],
                "published_c_lower": published["C"][0],
                "published_c_upper": published["C"][1],
                "published_u_lower": published["U"][0],
                "published_u_upper": published["U"][1],
            }
        )
    frame = pd.DataFrame(records)
    frame.to_csv(
        RESULTS_DIR / "gkm_spec2_replication.csv",
        index=False,
        float_format="%.15g",
    )

    markdown_lines = [
        "# GKM Table 2, Specification 2, with Hoekstra-Windmeijer CHW",
        "",
        "| Coefficient | 2SLS estimate | W | C | CHW | U |",
        "|---|---:|:---:|:---:|:---:|:---:|",
    ]
    for row in rows:
        published = GKM_REPORTED[row.variable]
        chw = row.conditional_hw
        assert chw is not None
        # For p=2 the newly computed CHW interval equals the replicated C
        # interval exactly.  Display both with the same published C endpoints
        if row.variable in ENDOGENOUS:
            chw_display = (
                f"[{published['C'][0]:.3f}, {published['C'][1]:.3f}]"
            )
        else:
            chw_display = interval_text(chw, 3)
        markdown_lines.append(
            "| "
            f"{row.display_name} | {published['estimate']:.3f} | "
            f"[{published['W'][0]:.3f}, {published['W'][1]:.3f}] | "
            f"[{published['C'][0]:.3f}, {published['C'][1]:.3f}] | "
            f"{chw_display} | "
            f"[{published['U'][0]:.3f}, {published['U'][1]:.3f}] |"
        )
    markdown_lines.extend(
        [
            "",
            "W, C, and U are the GKM printed entries after successful numerical "
            "replication. CHW is new. Full-precision replicated roots and endpoint "
            "differences are in `gkm_spec2_replication.csv`. For the two endogenous "
            "rows, CHW and C are identical because m_W=1.",
        ]
    )
    (RESULTS_DIR / "gkm_spec2_table.md").write_text(
        "\n".join(markdown_lines) + "\n", encoding="utf-8"
    )
    return frame


def make_diagnostic_plot(
    problem: TestProblem,
    row: RowResult,
    grid: pd.DataFrame,
    output_path: Path,
    show_two_sls_estimate: bool = True,
    use_gkm_paper_style: bool = False,
    monochrome: bool = False,
) -> None:
    """Plot AR, GKM CV, and HW CV with both confidence-interval boundaries."""

    if row.conditional_hw is None:  # pragma: no cover - workflow guard
        raise ReplicationError("Cannot plot before CHW is available")
    figure, axis = plt.subplots(figsize=(9.2, 5.8))
    if monochrome:
        ar_color = "black"
        gkm_color = "black"
        hw_color = "black"
        unconditional_color = "black"
        gkm_endpoint_style = "--"
        hw_endpoint_style = "-."
    elif use_gkm_paper_style:
        ar_color = "#ef4444"
        gkm_color = "#c05ac8"
        hw_color = "#2563eb"
        unconditional_color = "#16a34a"
        gkm_endpoint_style = "--"
        hw_endpoint_style = "-."
    else:
        ar_color = "#1f2937"
        gkm_color = "#c2410c"
        hw_color = "#047857"
        unconditional_color = "#7c3aed"
        gkm_endpoint_style = "-"
        hw_endpoint_style = "-"
    axis.plot(
        grid["beta0"],
        grid["ar_statistic"],
        color=ar_color,
        linewidth=2.1,
        label=(
            r"Subvector AR ($\widehat{\kappa}_{pn}$)"
            if problem.variable == "south"
            else r"Feasible subvector AR ($\widehat{\kappa}_{pn}$)"
        ),
    )
    axis.plot(
        grid["beta0"],
        grid["cv_gkm"],
        color=gkm_color,
        linewidth=2.0,
        linestyle="--",
        label=r"GKM cv $c_{0.95}(\widehat{\kappa}_{1n},1)$",
    )
    axis.plot(
        grid["beta0"],
        grid["cv_chw"],
        color=hw_color,
        linewidth=2.0,
        linestyle="-.",
        label=r"HW cv $c_{0.95}(\widehat{\kappa}_{(p-1)n},1)$",
    )
    unconditional_df = problem.k - problem.m_w
    axis.plot(
        grid["beta0"],
        grid["cv_u"],
        color=unconditional_color,
        linewidth=1.7,
        linestyle=":",
        label=rf"Unconditional cv $\chi^2_{{{unconditional_df},0.95}}$",
    )
    axis.axvline(
        row.conditional_gkm.lower,
        color=gkm_color,
        linewidth=1.2,
        alpha=0.8,
        linestyle=gkm_endpoint_style,
        label=None if use_gkm_paper_style else "GKM C endpoints",
    )
    axis.axvline(
        row.conditional_gkm.upper,
        color=gkm_color,
        linewidth=1.2,
        alpha=0.8,
        linestyle=gkm_endpoint_style,
    )
    axis.axvline(
        row.conditional_hw.lower,
        color=hw_color,
        linewidth=1.2,
        alpha=0.8,
        linestyle=hw_endpoint_style,
        label=None if use_gkm_paper_style else "HW CHW endpoints",
    )
    axis.axvline(
        row.conditional_hw.upper,
        color=hw_color,
        linewidth=1.2,
        alpha=0.8,
        linestyle=hw_endpoint_style,
    )
    if use_gkm_paper_style:
        axis.axvline(
            row.unconditional.lower,
            color=unconditional_color,
            linewidth=1.35,
            linestyle=":",
            alpha=0.9,
        )
        axis.axvline(
            row.unconditional.upper,
            color=unconditional_color,
            linewidth=1.35,
            linestyle=":",
            alpha=0.9,
        )
    if show_two_sls_estimate:
        axis.axvline(
            row.estimate,
            color="#2563eb",
            linewidth=1.0,
            linestyle=":",
            label="2SLS estimate",
        )
    axis.axhline(0.0, color="#9ca3af", linewidth=0.8)
    axis.set_xlabel(r"Null value $\beta_0$")
    axis.set_ylabel("Statistic / critical value")
    if not use_gkm_paper_style:
        title = f"{row.display_name} (Specification 2): GKM C versus HW CHW"
        axis.set_title(title)
    axis.set_xlim(float(grid["beta0"].min()), float(grid["beta0"].max()))
    if problem.variable == "gender":
        axis.set_xticks([-0.5, -0.25, 0.0, 0.25, 0.5, 0.75])
    elif problem.variable == "south":
        axis.set_xlim(SOUTH_PLOT_BETA_MIN, SOUTH_PLOT_BETA_MAX)
        axis.set_ylim(SOUTH_PLOT_Y_MIN, SOUTH_PLOT_Y_MAX)
        axis.set_xticks(
            np.arange(
                SOUTH_PLOT_BETA_MIN,
                SOUTH_PLOT_BETA_MAX + 0.01,
                0.25,
            )
        )
        axis.set_yticks(
            np.arange(SOUTH_PLOT_Y_MIN, SOUTH_PLOT_Y_MAX + 0.01, 1.0)
        )
    axis.grid(True, color="#d1d5db", alpha=0.55, linewidth=0.7)
    handles, labels = axis.get_legend_handles_labels()
    unique: dict[str, object] = {}
    for handle, label in zip(handles, labels, strict=True):
        unique.setdefault(label, handle)
    if use_gkm_paper_style:
        axis.legend(
            unique.values(),
            unique.keys(),
            frameon=True,
            facecolor="white",
            edgecolor="#d1d5db",
            framealpha=0.95,
            fontsize=8.2,
            loc="upper center",
            ncol=4,
            columnspacing=1.25,
            handlelength=2.4,
        )
        figure.tight_layout()
    else:
        axis.legend(unique.values(), unique.keys(), frameon=False, fontsize=9)
        figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def make_exogenous_conditioning_figure(
    problems: dict[str, TestProblem],
) -> None:
    """Plot conditioning roots over one common beta grid."""

    beta_grid = np.arange(
        EXOGENOUS_CONDITIONING_BETA_MIN,
        EXOGENOUS_CONDITIONING_BETA_MAX
        + 0.5 * EXOGENOUS_CONDITIONING_BETA_STEP,
        EXOGENOUS_CONDITIONING_BETA_STEP,
    )
    plot_frames: dict[str, pd.DataFrame] = {}
    output_frames: list[pd.DataFrame] = []
    for variable in EXOGENOUS_CONDITIONING_PLOT_VARIABLES:
        variable_beta_grid = (
            np.arange(
                AGE_CONDITIONING_BETA_MIN,
                AGE_CONDITIONING_BETA_MAX
                + 0.5 * AGE_CONDITIONING_BETA_STEP,
                AGE_CONDITIONING_BETA_STEP,
            )
            if variable == "age"
            else beta_grid
        )
        root_array = np.array(
            [
                gkm_eigenvalues(problems[variable], float(beta0))[:2]
                for beta0 in variable_beta_grid
            ]
        )
        frame = pd.DataFrame(
            {
                "variable": variable,
                "beta0": variable_beta_grid,
                "kappa_hat_1n": root_array[:, 0],
                "kappa_hat_2n": root_array[:, 1],
            }
        )
        plot_frames[variable] = frame
        output_frames.append(frame)
    pd.concat(output_frames, ignore_index=True).to_csv(
        GRID_DIR / "exogenous_conditioning_statistics_wide_grid.csv",
        index=False,
        float_format="%.15g",
    )

    figure, axes = plt.subplots(
        2,
        3,
        figsize=(13.2, 7.6),
        sharex=False,
        sharey=False,
    )
    gkm_color = "#c05ac8"
    hw_color = "#2563eb"

    for axis, variable in zip(
        axes.flat,
        EXOGENOUS_CONDITIONING_PLOT_VARIABLES,
        strict=True,
    ):
        grid = plot_frames[variable]
        axis.plot(
            grid["beta0"],
            grid["kappa_hat_1n"],
            color=gkm_color,
            linewidth=2.0,
            linestyle="--",
            label=r"GKM: $\widehat{\kappa}_{1n}$",
        )
        axis.plot(
            grid["beta0"],
            grid["kappa_hat_2n"],
            color=hw_color,
            linewidth=2.0,
            linestyle="-.",
            label=r"HW: $\widehat{\kappa}_{2n}$",
        )
        axis.set_title(DISPLAY_NAMES[variable], fontsize=11)
        axis.set_xlabel(r"Null value $\beta_0$")
        if variable == "age":
            panel_beta_min = AGE_CONDITIONING_BETA_MIN
            panel_beta_max = AGE_CONDITIONING_BETA_MAX
            axis.set_xticks([-0.0312, -0.0156, 0.0, 0.0156, 0.0312])
        else:
            panel_beta_min = EXOGENOUS_CONDITIONING_BETA_MIN
            panel_beta_max = EXOGENOUS_CONDITIONING_BETA_MAX
            axis.set_xticks([-0.125, -0.0625, 0.0, 0.0625, 0.125])
        axis.set_xlim(panel_beta_min, panel_beta_max)
        axis.tick_params(axis="x", labelbottom=True)
        visible_grid = grid.loc[
            grid["beta0"].between(
                panel_beta_min,
                panel_beta_max,
                inclusive="both",
            )
        ]
        panel_maximum = float(
            max(
                visible_grid["kappa_hat_1n"].max(),
                visible_grid["kappa_hat_2n"].max(),
            )
        )
        axis.set_ylim(0.0, 1.08 * panel_maximum)
        axis.ticklabel_format(axis="y", style="plain")
        axis.grid(
            True,
            which="major",
            color="#d1d5db",
            alpha=0.55,
            linewidth=0.7,
        )

    figure.supylabel("Conditioning statistic")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=True,
        facecolor="white",
        edgecolor="#d1d5db",
        framealpha=0.95,
        fontsize=10,
        bbox_to_anchor=(0.5, 0.995),
    )
    figure.subplots_adjust(
        left=0.075,
        right=0.985,
        bottom=0.09,
        top=0.90,
        hspace=0.35,
        wspace=0.18,
    )
    figure.savefig(
        RESULTS_DIR / "exogenous_conditioning_statistics.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)


def fixed_figure6_beta_grid() -> np.ndarray:
    """User-requested Figure 6 window with exact 0.001 increments."""

    number = int(
        round(
            (GENDER_FIGURE6_BETA_MAX - GENDER_FIGURE6_BETA_MIN)
            / GENDER_FIGURE6_STEP
        )
    ) + 1
    return np.linspace(
        GENDER_FIGURE6_BETA_MIN,
        GENDER_FIGURE6_BETA_MAX,
        number,
    )


def make_gender_spec1_figure6(data: pd.DataFrame) -> None:
    """Reproduce the left panel of GKM Figure 6 for Gender Specification 1."""

    problem = build_gender_spec1_problem(data)
    estimate, standard_error = gender_spec1_two_sls(data)
    conditional = invert_test(problem, "C", estimate, standard_error)
    conditional_hw = invert_test(problem, "CHW", estimate, standard_error)
    unconditional = invert_test(problem, "U", estimate, standard_error)

    chw_difference = max(
        abs(conditional.lower - conditional_hw.lower),
        abs(conditional.upper - conditional_hw.upper),
    )
    if chw_difference > 2e-10:
        raise ReplicationError(
            "Gender Specification 1 must have CHW=C because m_W=1"
        )
    benchmark_difference = max(
        abs(conditional.lower - GKM_GENDER_SPEC1_REPORTED["C"][0]),
        abs(conditional.upper - GKM_GENDER_SPEC1_REPORTED["C"][1]),
        abs(unconditional.lower - GKM_GENDER_SPEC1_REPORTED["U"][0]),
        abs(unconditional.upper - GKM_GENDER_SPEC1_REPORTED["U"][1]),
    )
    # The paper reports endpoints from a 0.001 grid, whereas these are Brent-
    # refined intersections of GKM's linearly interpolated CV table.
    if benchmark_difference > 2.01e-3:
        raise ReplicationError(
            "Gender Specification 1 C/U intervals do not reproduce Figure 6"
        )

    beta_grid = fixed_figure6_beta_grid()
    records: list[dict[str, float]] = []
    for beta0 in beta_grid:
        roots = gkm_eigenvalues(problem, float(beta0))
        conditioning = float(roots[0])
        ar_statistic = float(roots[-1])
        record = {
            "beta0": float(beta0),
            "kappa_hat_1n_gkm_and_hw_conditioning": conditioning,
            "kappa_hat_2n_feasible_subvector_ar": ar_statistic,
        }
        for alpha in (0.01, 0.05, 0.10):
            suffix = f"{int(round(alpha * 100)):02d}pct"
            record[f"conditional_cv_{suffix}"] = gkm_conditional_cv(
                conditioning, 2, alpha
            )
            record[f"unconditional_cv_{suffix}"] = float(
                chi2.ppf(1.0 - alpha, 2)
            )
        records.append(record)
    frame = pd.DataFrame(records)
    frame.to_csv(
        GRID_DIR / "gender_spec1_figure6_grid.csv",
        index=False,
        float_format="%.15g",
    )

    if float(frame["kappa_hat_2n_feasible_subvector_ar"].max()) < 14.0:
        raise ReplicationError("Gender Specification 1 grid misses the full AR shape")
    cv_5 = frame["conditional_cv_05pct"]
    if float(cv_5.min()) < 5.4 or float(cv_5.max()) >= float(
        chi2.ppf(0.95, 2)
    ):
        raise ReplicationError("Gender Specification 1 5% CV range is implausible")

    figure, axis = plt.subplots(figsize=(10.0, 6.3))
    ar_color = "#dc2626"
    conditional_color = "#a21caf"
    unconditional_color = "#16a34a"
    axis.plot(
        frame["beta0"],
        frame["kappa_hat_2n_feasible_subvector_ar"],
        color=ar_color,
        linewidth=2.25,
        label=r"Feasible subvector AR ($\widehat{\kappa}_{2n}$)",
        zorder=4,
    )
    for alpha in (0.01, 0.05, 0.10):
        suffix = f"{int(round(alpha * 100)):02d}pct"
        emphasized = math.isclose(alpha, 0.05)
        line_width = 2.1 if emphasized else 1.25
        line_alpha = 1.0 if emphasized else 0.72
        axis.plot(
            frame["beta0"],
            frame[f"conditional_cv_{suffix}"],
            color=conditional_color,
            linewidth=line_width,
            linestyle="--",
            alpha=line_alpha,
            label=(
                r"GKM conditional CV "
                r"$c_{1-\alpha}(\widehat{\kappa}_{1n},2)$"
                if math.isclose(alpha, 0.01)
                else None
            ),
            zorder=3,
        )
        unconditional_cv = float(frame[f"unconditional_cv_{suffix}"].iloc[0])
        axis.axhline(
            unconditional_cv,
            color=unconditional_color,
            linewidth=line_width,
            linestyle=":",
            alpha=line_alpha,
            label=(
                r"Unconditional $\chi^2_2$ CV"
                if math.isclose(alpha, 0.01)
                else None
            ),
            zorder=2,
        )
        axis.text(
            GENDER_FIGURE6_BETA_MIN + 0.015,
            unconditional_cv + 0.16,
            rf"$\alpha={int(round(alpha * 100))}\%$",
            color="#374151",
            fontsize=9,
        )

    for endpoint in (conditional.lower, conditional.upper):
        axis.axvline(
            endpoint,
            color=conditional_color,
            linewidth=1.35,
            linestyle="--",
            label="95% C = CHW endpoints" if endpoint == conditional.lower else None,
        )
    for endpoint in (unconditional.lower, unconditional.upper):
        axis.axvline(
            endpoint,
            color=unconditional_color,
            linewidth=1.35,
            linestyle=":",
            label="95% U endpoints" if endpoint == unconditional.lower else None,
        )

    axis.set_xlim(GENDER_FIGURE6_BETA_MIN, GENDER_FIGURE6_BETA_MAX)
    axis.set_ylim(0.0, 15.0)
    axis.set_xticks([-0.5, -0.25, 0.0, 0.25, 0.5, 0.75])
    axis.set_yticks([0.0, 5.0, 10.0, 15.0])
    axis.set_xlabel(r"Null value $\beta_0$")
    axis.set_ylabel("Statistic / critical value")
    axis.set_title("Gender (Specification 1): replication of GKM Figure 6")
    axis.text(
        0.5,
        0.975,
        r"$m_W=1$: HW CHW and GKM C coincide pointwise",
        transform=axis.transAxes,
        ha="center",
        va="top",
        color="#4b5563",
        fontsize=10,
    )
    axis.grid(True, color="#d1d5db", alpha=0.45, linewidth=0.65)
    axis.legend(
        frameon=True,
        facecolor="white",
        edgecolor="#d1d5db",
        framealpha=0.92,
        fontsize=9,
        loc="lower right",
    )
    figure.tight_layout()
    figure.savefig(
        RESULTS_DIR / "gender_spec1_gkm_figure6.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)

    print("\nGENDER SPECIFICATION 1 / GKM FIGURE 6")
    print("======================================")
    print("  x-axis beta0                  = [-0.5, 0.75]")
    print("  k                             = 3")
    print("  m_W                           = 1")
    print("  k - m_W                       = 2")
    print(f"  2SLS Gender estimate          = {estimate:.12g}")
    print(
        "  5% unconditional CV          = "
        f"chi-square_2(0.95) = {chi2.ppf(0.95, 2):.12g}"
    )
    print(
        "  5% conditional CV range      = "
        f"[{cv_5.min():.9f}, {cv_5.max():.9f}]"
    )
    print(f"  GKM C = HW CHW                = {interval_text(conditional, 6)}")
    print(f"  unconditional U               = {interval_text(unconditional, 6)}")
    print(f"  max published-endpoint diff   = {benchmark_difference:.3e}")


def extended_gender_spec2_grid(problem: TestProblem) -> pd.DataFrame:
    """Specification 2 Gender grid on the same x window for comparison."""

    frame = pd.DataFrame(
        [test_at_beta0(problem, float(beta0)) for beta0 in fixed_figure6_beta_grid()]
    )
    frame.to_csv(
        GRID_DIR / "gender_spec2_extended_beta_grid.csv",
        index=False,
        float_format="%.15g",
    )
    return frame


def extended_south_spec2_grid(problem: TestProblem) -> pd.DataFrame:
    """Specification 2 South grid wide enough to display both curve tails."""

    beta_grid = np.arange(
        SOUTH_PLOT_BETA_MIN,
        SOUTH_PLOT_BETA_MAX + 0.5 * SOUTH_PLOT_BETA_STEP,
        SOUTH_PLOT_BETA_STEP,
    )
    frame = pd.DataFrame(
        [test_at_beta0(problem, float(beta0)) for beta0 in beta_grid]
    )
    frame.to_csv(
        GRID_DIR / "south_spec2_extended_beta_grid.csv",
        index=False,
        float_format="%.15g",
    )
    return frame


def print_problem_summary(problems: dict[str, TestProblem]) -> None:
    print("\nGKM TEST-PROBLEM DIMENSIONS")
    print("===========================")
    for variable in ROW_ORDER:
        problem = problems[variable]
        print(f"\n{problem.display_name} ({problem.case} coefficient)")
        print(f"  controls C       = {list(problem.controls)}")
        print(f"  tested Y         = {variable}")
        nuisance_names = (
            list(ENDOGENOUS)
            if problem.case == "exogenous"
            else [name for name in ENDOGENOUS if name != variable]
        )
        print(f"  nuisance W       = {nuisance_names}")
        print(f"  instruments Z    = {list(problem.instruments)}")
        print(f"  N                = {problem.original_n}")
        print(f"  rank(C)          = {problem.rank_c}")
        print(f"  effective n      = {problem.effective_n}")
        print(f"  k                = {problem.k}")
        print(f"  m_W              = {problem.m_w}")
        print(f"  k - m_W          = {problem.k - problem.m_w}")
        print(f"  n - k            = {problem.denominator_df}")
        print(
            f"  first-stage rank = {problem.first_stage_rank}/{problem.k}"
        )
        if problem.m_w == 2:
            print(
                "  sample roots: kappa_hat_1n=GKM cond., "
                "kappa_hat_2n=HW cond., kappa_hat_3n=AR"
            )
        else:
            print(
                "  sample roots: kappa_hat_1n=GKM/HW cond., "
                "kappa_hat_2n=AR; therefore CHW=C"
            )


def summarize_screening(
    rows: list[RowResult],
    diagnostics: pd.DataFrame,
    boundary_diagnostics: pd.DataFrame,
    max_estimate_diff: float,
    max_wald_diff: float,
    max_c_diff: float,
    max_u_diff: float,
    whitening_max: float,
) -> None:
    print("\nGKM SPECIFICATION 2 REPLICATION")
    print("===============================")
    print(f"N = {EXPECTED_N}")
    print(f"Endogenous regressors: {list(ENDOGENOUS)}")
    print(f"Excluded instruments:  {list(EXCLUDED_INSTRUMENTS)}")
    print(f"Exogenous regressors:   {list(INCLUDED_EXOGENOUS)}")
    print(f"Maximum |2SLS estimate - reported| = {max_estimate_diff:.9g}")
    print(f"Maximum |W endpoint - reported|     = {max_wald_diff:.9g}")
    print(f"Maximum |C endpoint - reported|     = {max_c_diff:.9g}")
    print(f"Maximum |U endpoint - reported|     = {max_u_diff:.9g}")
    print(f"Maximum generalized/whitened diff   = {whitening_max:.3e}")
    print("GKM replication succeeds at the precision of the published 0.001 grid.")

    print("\nHW COMPARISON")
    print("=============")
    for row in rows:
        assert row.conditional_hw is not None
        reduction = row.conditional_gkm.length - row.conditional_hw.length
        percentage = 100.0 * reduction / row.conditional_gkm.length
        print(f"\n{row.display_name}")
        print(f"  GKM C     = {interval_text(row.conditional_gkm, 6)}")
        print(f"  HW CHW    = {interval_text(row.conditional_hw, 6)}")
        print(f"  reduction = {reduction:.9g} ({percentage:.3f}%)")
        if row.variable in ENDOGENOUS:
            print("  CHW = C because m_W = 1.")

    print("\nSAMPLE-EIGENVALUE AND CRITICAL-VALUE GAP DIAGNOSTICS")
    print("================================================")
    print(diagnostics.to_string(index=False, float_format=lambda value: f"{value:.9g}"))
    print("\nSAMPLE ROOTS AND CRITICAL VALUES AT C / CHW BOUNDARIES")
    print("=================================================")
    boundary_columns = [
        "variable",
        "method",
        "side",
        "beta0",
        "kappa_hat_1n",
        "kappa_hat_2n",
        "kappa_hat_3n",
        "cv_gkm",
        "cv_chw",
    ]
    print(
        boundary_diagnostics[boundary_columns].to_string(
            index=False, float_format=lambda value: f"{value:.9g}"
        )
    )


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    GRID_DIR.mkdir(parents=True, exist_ok=True)
    log_path = RESULTS_DIR / "vietnam_gkm_spec2_replication.log"

    with log_path.open("w", encoding="utf-8") as log_file:
        with redirect_stdout(Tee(sys.stdout, log_file)):
            inspect_vietnam_directory()
            data = load_vietnam_data()
            build_tcn_specification()
            two_sls = two_stage_least_squares(data)
            first_stages = first_stage_diagnostics(data)
            first_stages.to_csv(
                RESULTS_DIR / "first_stage_diagnostics.csv",
                index=False,
                float_format="%.15g",
            )

            validate_2sls_and_wald(two_sls)
            validate_critical_value_table()

            problems = {
                variable: build_test_problem(data, variable) for variable in ROW_ORDER
            }
            print_problem_summary(problems)

            whitening_max = 0.0
            decomposition_max = 0.0
            for variable in ROW_ORDER:
                problem = problems[variable]
                estimate = two_sls.coefficients[variable]
                discrepancy, decomposition = validate_eigenvalues(
                    problem, (0.0, estimate, estimate - 0.1, estimate + 0.1)
                )
                whitening_max = max(whitening_max, discrepancy)
                decomposition_max = max(decomposition_max, decomposition)
            print("\nEigenvalue implementation cross-check:")
            print(f"  maximum generalized/whitened discrepancy = {whitening_max:.3e}")
            print(f"  maximum |Q'Q - ESS - RSS|                = {decomposition_max:.3e}")

            # Stage 1: replicate GKM C and U for every row.  No CHW calculation
            # occurs before validate_gkm_intervals returns successfully.
            rows = [
                replicate_gkm_row(problems[variable], two_sls)
                for variable in ROW_ORDER
            ]
            max_c_diff, max_u_diff = validate_gkm_intervals(rows)

            # Stage 2: the source GKM implementation has passed, so compute CHW.
            print("\nGKM replication validated. Computing CHW...")
            for row in rows:
                compute_chw_row(
                    problems[row.variable],
                    row,
                    two_sls.standard_errors[row.variable],
                )

            all_boundary_frames: list[pd.DataFrame] = []
            gap_rows: list[dict[str, float | str]] = []
            grid_frames: dict[str, pd.DataFrame] = {}
            for row in rows:
                grid, boundary, summary = run_beta_grid(problems[row.variable], row)
                grid_frames[row.variable] = grid
                all_boundary_frames.append(boundary)
                gap_rows.append({"variable": row.display_name, **summary})
            boundary_diagnostics = pd.concat(
                all_boundary_frames, ignore_index=True
            )
            boundary_diagnostics.to_csv(
                RESULTS_DIR / "boundary_diagnostics.csv",
                index=False,
                float_format="%.15g",
            )
            gap_diagnostics = pd.DataFrame(gap_rows)
            gap_diagnostics.to_csv(
                RESULTS_DIR / "eigenvalue_cv_gap_diagnostics.csv",
                index=False,
                float_format="%.15g",
            )

            make_exogenous_conditioning_figure(problems)

            final_frame = make_table(rows)

            make_gender_spec1_figure6(data)

            exogenous_rows = [row for row in rows if row.variable in INCLUDED_EXOGENOUS]
            largest_reduction_row = max(
                exogenous_rows,
                key=lambda row: row.conditional_gkm.length
                - (row.conditional_hw.length if row.conditional_hw else math.inf),
            )
            rows_to_plot = {largest_reduction_row.variable: largest_reduction_row}
            for row in exogenous_rows:
                assert row.conditional_hw is not None
                if row.conditional_gkm.contains(0.0) != row.conditional_hw.contains(0.0):
                    rows_to_plot[row.variable] = row
            for variable in REQUESTED_PLOT_VARIABLES:
                rows_to_plot[variable] = next(
                    row for row in exogenous_rows if row.variable == variable
                )
            for variable, row in rows_to_plot.items():
                plot_path = RESULTS_DIR / f"{safe_slug(row.display_name)}_gkm_vs_chw.png"
                if variable == "gender":
                    plot_grid = extended_gender_spec2_grid(problems[variable])
                elif variable == "south":
                    plot_grid = extended_south_spec2_grid(problems[variable])
                else:
                    plot_grid = grid_frames[variable]
                make_diagnostic_plot(
                    problems[variable],
                    row,
                    plot_grid,
                    plot_path,
                    show_two_sls_estimate=variable not in {"gender", "south"},
                    use_gkm_paper_style=variable == "south",
                    monochrome=variable == "south",
                )

            max_estimate_diff = max(
                abs(two_sls.coefficients[variable] - GKM_REPORTED[variable]["estimate"])
                for variable in ROW_ORDER
            )
            max_wald_diff = max(
                abs(value - target)
                for variable in ROW_ORDER
                for value, target in zip(
                    (
                        two_sls.wald_intervals[variable].lower,
                        two_sls.wald_intervals[variable].upper,
                    ),
                    GKM_REPORTED[variable]["W"],
                    strict=True,
                )
            )
            summarize_screening(
                rows,
                gap_diagnostics,
                boundary_diagnostics,
                max_estimate_diff,
                max_wald_diff,
                max_c_diff,
                max_u_diff,
                whitening_max,
            )

            print("\nFINAL TABLE (root-refined calculations)")
            print("=======================================")
            display_columns = [
                "display_name",
                "estimate",
                "wald_lower",
                "wald_upper",
                "gkm_c_lower",
                "gkm_c_upper",
                "hw_chw_lower",
                "hw_chw_upper",
                "unconditional_u_lower",
                "unconditional_u_upper",
            ]
            print(
                final_frame[display_columns].to_string(
                    index=False, float_format=lambda value: f"{value:.6f}"
                )
            )
            print("\nOutputs written to:")
            for output in sorted(RESULTS_DIR.rglob("*")):
                if output.is_file():
                    print(f"  {output.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    try:
        main()
    except (ReplicationError, FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"REPLICATION STOPPED: {exc}") from exc
