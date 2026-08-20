"""Read-only live monitor for a long adaptive EMW power-bound run.

This process deliberately stays separate from :mod:`alfd_eigval`: plotting or
network failures must never interrupt, slow, or change the numerical run.  It
polls the atomically published partial checkpoint and switches to the final
artifact once that artifact is compatible with the same run.

The finite-sample Appendix A.3 curves and the asymptotic EMW curve are not the
same statistical experiment.  Their overlay is therefore a useful diagnostic,
not a theorem-level bound check.  The invariant benchmark stored by
``alfd_eigval.py`` *is* from the bound calculation's experiment and is shown
separately.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import os
import re
import tempfile
import time
import warnings
from dataclasses import dataclass
from typing import Any, Mapping, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import new_power_comparison as comparison


_HEX_SIGNATURE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class DgpCurves:
    """Authenticated finite-sample Appendix A.3 reference curves."""

    betas: np.ndarray
    power_chi2: np.ndarray
    power_c1: np.ndarray
    power_cp1: np.ndarray
    settings: Mapping[str, Any]
    run_signature: str
    path: str


@dataclass(frozen=True)
class BoundProgress:
    """One internally consistent snapshot of the EMW checkpoint/artifact."""

    betas: np.ndarray
    bounds_confidence: np.ndarray
    bounds_point: np.ndarray
    benchmark_lower_confidence: np.ndarray
    run_signature: str
    source_path: str
    is_final: bool
    settings: Optional[Mapping[str, Any]] = None


def _scalar(archive: np.lib.npyio.NpzFile, key: str) -> Any:
    if key not in archive.files:
        raise ValueError(f"missing metadata key {key!r}")
    value = np.asarray(archive[key])
    if value.shape != ():
        raise ValueError(
            f"metadata key {key!r} must be scalar, got shape {value.shape}")
    return value.item()


def _canonical_settings_and_signature(raw: Any) -> tuple[dict[str, Any], str]:
    if not isinstance(raw, str):
        raise ValueError("settings_json must be a string")
    try:
        settings = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid settings_json: {exc}") from exc
    if not isinstance(settings, dict):
        raise ValueError("settings_json must encode an object")
    canonical = json.dumps(settings, sort_keys=True, separators=(",", ":"))
    signature = hashlib.sha256(canonical.encode()).hexdigest()
    return settings, signature


def _validate_signature(signature: Any, calculated: Optional[str] = None) -> str:
    signature = str(signature)
    if not _HEX_SIGNATURE.fullmatch(signature):
        raise ValueError(f"invalid run_signature {signature!r}")
    if calculated is not None and signature != calculated:
        raise ValueError(
            "run_signature does not authenticate the canonical settings_json")
    return signature


def _plain_int(value: Any, *, positive: bool = False) -> int:
    if (not isinstance(value, (int, np.integer))
            or isinstance(value, (bool, np.bool_))):
        raise ValueError(f"expected integer metadata, got {value!r}")
    value = int(value)
    if positive and value <= 0:
        raise ValueError(f"expected a positive integer, got {value}")
    return value


def _probability(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not np.isfinite(result) or not 0.0 < result < 1.0:
        raise ValueError(f"{name} must lie in (0, 1)")
    return result


def load_dgp_curves(version: str, path: Optional[str] = None) -> DgpCurves:
    """Strictly load the canonical finite-sample DGP cache.

    The cache's signed settings provide the simulation count, seed, chunking,
    grid, and experiment configuration.  Those values are then passed through
    ``new_power_comparison.load_compatible_dgp_cache``.  Consequently the
    existing full provenance contract (including current source hashes and
    software environment) remains the single source of truth.
    """

    if version not in comparison.VERSION_LABELS:
        raise ValueError(f"unknown version label {version!r}")
    if path is None:
        path = comparison.dgp_cache_path(version)
    path = os.fspath(path)

    with np.load(path, allow_pickle=False) as archive:
        settings, calculated = _canonical_settings_and_signature(
            _scalar(archive, "settings_json"))
        signature = _validate_signature(
            _scalar(archive, "run_signature"), calculated)

    required = (
        "version_label", "kappas", "k", "n", "alpha", "betas",
        "num_simulations", "base_seed", "chunk_size",
    )
    missing = [key for key in required if key not in settings]
    if missing:
        raise ValueError(
            "DGP settings_json is missing: " + ", ".join(missing))
    if settings["version_label"] != version:
        raise ValueError(
            f"DGP version_label={settings['version_label']!r}, expected {version!r}")

    try:
        kappas = np.asarray(settings["kappas"], dtype=float)
        betas = np.asarray(settings["betas"], dtype=float)
        k = _plain_int(settings["k"], positive=True)
        n = _plain_int(settings["n"], positive=True)
        alpha = _probability(settings["alpha"], "DGP alpha")
        num_simulations = _plain_int(
            settings["num_simulations"], positive=True)
        base_seed = _plain_int(settings["base_seed"])
        chunk_size = _plain_int(settings["chunk_size"], positive=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid DGP settings_json: {exc}") from exc

    loaded = comparison.load_compatible_dgp_cache(
        path, version_label=version, kappas=kappas, k=k, n=n, alpha=alpha,
        betas=betas, num_simulations=num_simulations,
        base_seed=base_seed, chunk_size=chunk_size)
    return DgpCurves(
        *(np.asarray(value, dtype=float).copy() for value in loaded),
        settings=dict(settings), run_signature=signature, path=path)


def _validate_bound_arrays(
        betas: Any, confidence: Any, point: Any, benchmark: Any,
        *, alpha: Optional[float]) -> tuple[np.ndarray, ...]:
    arrays = tuple(np.asarray(value, dtype=float).copy() for value in (
        betas, confidence, point, benchmark))
    betas, confidence, point, benchmark = arrays
    if betas.ndim != 1 or betas.size == 0:
        raise ValueError("bound betas must be a nonempty one-dimensional array")
    if any(value.shape != betas.shape for value in arrays[1:]):
        raise ValueError("bound progress arrays must all match the beta grid")
    if not np.all(np.isfinite(betas)) or np.any(np.diff(betas) <= 0.0):
        raise ValueError("bound beta grid must be finite and strictly increasing")

    complete = np.isfinite(confidence)
    for name, value in (
            ("bounds_point", point),
            ("benchmark_lower_confidence", benchmark)):
        if not np.array_equal(np.isfinite(value), complete):
            raise ValueError(
                f"{name} and bounds_confidence have different completion masks")
    for name, value in (
            ("bounds_confidence", confidence), ("bounds_point", point),
            ("benchmark_lower_confidence", benchmark)):
        finite = value[complete]
        if np.any((finite < 0.0) | (finite > 1.0)):
            raise ValueError(f"completed {name} values must lie in [0, 1]")
    if alpha is not None and np.any(confidence[complete] < alpha - 1e-12):
        raise ValueError("a completed EMW upper bound lies below alpha")
    if np.any(confidence[complete] + 1e-12 < benchmark[complete]):
        raise ValueError(
            "EMW confidence upper lies below its same-experiment benchmark lower")
    return arrays


def _validate_partial_metadata(
        archive: np.lib.npyio.NpzFile, version: str,
        expected_run_signature: Optional[str]) -> tuple[str, Optional[dict[str, Any]]]:
    signature = _validate_signature(_scalar(archive, "run_signature"))
    if expected_run_signature is not None and signature != expected_run_signature:
        raise ValueError(
            f"run signature differs: {signature!r} != {expected_run_signature!r}")

    # Older checkpoints contain only the signature and numerical arrays.  Such
    # a snapshot can be shape/range checked but not provenance-authenticated.
    # New checkpoints publish this complete metadata group before bank fitting.
    metadata = {
        "schema_version", "algorithm", "producer", "bound_kind",
        "version_label", "settings_json", "kappas", "k", "n", "alpha",
        "curve_confidence",
    }
    present = metadata.intersection(archive.files)
    if not present:
        return signature, None
    missing = metadata.difference(archive.files)
    if missing:
        raise ValueError(
            "partial checkpoint has incomplete provenance metadata: "
            + ", ".join(sorted(missing)))

    settings, calculated = _canonical_settings_and_signature(
        _scalar(archive, "settings_json"))
    _validate_signature(signature, calculated)
    scalar_expected = {
        "schema_version": comparison.ALFD_SCHEMA_VERSION,
        "algorithm": comparison.ALFD_ALGORITHM,
        "producer": comparison.ALFD_PRODUCER,
        "bound_kind": comparison.ALFD_BOUND_KIND,
        "version_label": version,
    }
    mismatches = []
    for key, expected in scalar_expected.items():
        saved = _scalar(archive, key)
        if saved != expected:
            mismatches.append(f"{key}={saved!r}, expected {expected!r}")
        # `bound_kind` is an authenticated top-level result label rather than
        # a numerical run setting; the other fields are signed in both places.
        if key != "bound_kind" and settings.get(key) != saved:
            mismatches.append(f"settings_json {key} disagrees with checkpoint")

    for key in ("k", "n"):
        saved = _scalar(archive, key)
        if settings.get(key) != saved:
            mismatches.append(f"settings_json {key} disagrees with checkpoint")
    for key in ("alpha", "curve_confidence"):
        saved = _scalar(archive, key)
        try:
            matches = bool(np.isclose(
                settings.get(key), saved, rtol=0.0, atol=1e-12))
        except TypeError:
            matches = False
        if not matches:
            mismatches.append(f"settings_json {key} disagrees with checkpoint")

    saved_kappas = np.asarray(archive["kappas"], dtype=float)
    try:
        settings_kappas = np.asarray(settings["kappas"], dtype=float)
    except (KeyError, TypeError, ValueError):
        mismatches.append("settings_json has invalid kappas")
    else:
        if (saved_kappas.shape != settings_kappas.shape
                or not np.allclose(saved_kappas, settings_kappas,
                                   rtol=0.0, atol=1e-12)):
            mismatches.append("settings_json kappas disagree with checkpoint")

    expected_methods = {
        "calibration_method": comparison.ALFD_CALIBRATION_METHOD,
        "confidence_allocation_method": (
            comparison.ALFD_CONFIDENCE_ALLOCATION_METHOD),
        "fit_grid_strategy": comparison.ALFD_COMMON_GRID_METHOD,
        "pooled_importance_method": comparison.ALFD_POOLED_IS_METHOD,
    }
    for key, expected in expected_methods.items():
        if settings.get(key) != expected:
            mismatches.append(
                f"settings_json {key}={settings.get(key)!r}, expected {expected!r}")
    if settings.get("profile") not in ("production", "reference"):
        mismatches.append("partial checkpoint is not a production/reference run")

    repository = os.path.dirname(os.path.abspath(__file__))
    library_name = "libmhg.dylib" if os.sys.platform == "darwin" else "libmhg.so"
    current_hashes = {
        "source_sha256": comparison._sha256_file(
            os.path.join(repository, "alfd_eigval.py")),
        "mhg_core_sha256": comparison._sha256_file(
            os.path.join(repository, "koev", "mhg15", "mhg_core.c")),
        "mhg_library_sha256": comparison._sha256_file(
            os.path.join(repository, "koev", "mhg15", library_name)),
    }
    current_hashes["mhg_build_source_sha256"] = current_hashes[
        "mhg_core_sha256"]
    for key, expected in current_hashes.items():
        if settings.get(key) != expected:
            mismatches.append(f"settings_json {key} is not current")
    if mismatches:
        raise ValueError("incompatible partial checkpoint: " + "; ".join(mismatches))
    return signature, settings


def _load_partial(
        version: str, path: str,
        expected_run_signature: Optional[str]) -> BoundProgress:
    with np.load(path, allow_pickle=False) as archive:
        signature, settings = _validate_partial_metadata(
            archive, version, expected_run_signature)
        required = (
            "betas", "bounds_confidence", "bounds_point",
            "benchmark_lower_confidence",
        )
        missing = [key for key in required if key not in archive.files]
        if missing:
            raise ValueError(
                "partial checkpoint is missing arrays: " + ", ".join(missing))
        alpha = None if settings is None else _probability(
            settings.get("alpha"), "bound alpha")
        arrays = _validate_bound_arrays(
            archive["betas"], archive["bounds_confidence"],
            archive["bounds_point"], archive["benchmark_lower_confidence"],
            alpha=alpha)
        if settings is not None:
            beta_count = settings.get("beta_count")
            if (not isinstance(beta_count, (int, np.integer))
                    or isinstance(beta_count, (bool, np.bool_))
                    or int(beta_count) != arrays[0].size):
                raise ValueError(
                    "settings_json beta_count disagrees with checkpoint beta grid")
    return BoundProgress(
        *arrays, run_signature=signature, source_path=path,
        is_final=False, settings=None if settings is None else dict(settings))


def _load_final(
        version: str, path: str,
        expected_run_signature: Optional[str]) -> BoundProgress:
    with np.load(path, allow_pickle=False) as archive:
        settings, calculated = _canonical_settings_and_signature(
            _scalar(archive, "settings_json"))
        signature = _validate_signature(
            _scalar(archive, "run_signature"), calculated)
    if expected_run_signature is not None and signature != expected_run_signature:
        raise ValueError(
            f"run signature differs: {signature!r} != {expected_run_signature!r}")

    required = ("kappas", "k", "n", "alpha")
    missing = [key for key in required if key not in settings]
    if missing:
        raise ValueError("final settings_json is missing: " + ", ".join(missing))
    kappas = np.asarray(settings["kappas"], dtype=float)
    k = _plain_int(settings["k"], positive=True)
    n = _plain_int(settings["n"], positive=True)
    alpha = _probability(settings["alpha"], "bound alpha")

    # This validates all v3 strategy/provenance fields and current code hashes.
    strict_betas, strict_bounds, _ = comparison.load_compatible_alfd_bound(
        path, version_label=version, kappas=kappas, k=k, n=n, alpha=alpha)
    with np.load(path, allow_pickle=False) as archive:
        point = np.asarray(archive["bounds_point"], dtype=float)
        benchmark = np.asarray(
            archive["invariant_benchmark_lower_confidence"], dtype=float)
    arrays = _validate_bound_arrays(
        strict_betas, strict_bounds, point, benchmark, alpha=alpha)
    return BoundProgress(
        *arrays, run_signature=signature, source_path=path,
        is_final=True, settings=dict(settings))


def load_bound_progress(
        version: str, partial_path: Optional[str] = None,
        final_path: Optional[str] = None,
        expected_run_signature: Optional[str] = None) -> Optional[BoundProgress]:
    """Load one atomic progress snapshot, preferring the active run.

    A compatible partial checkpoint wins over a final artifact with a different
    signature (the usual ``--force`` situation with an old final file).  Once a
    final artifact has the same signature, it wins.  If the partial is stale or
    malformed but a compatible requested final exists, the final is used.
    """

    if version not in comparison.VERSION_LABELS:
        raise ValueError(f"unknown version label {version!r}")
    adaptive = os.path.join(version, "adaptive")
    if partial_path is None:
        partial_path = os.path.join(
            adaptive, f"alfd_eigval_{version}.partial.npz")
    if final_path is None:
        final_path = os.path.join(adaptive, f"alfd_eigval_{version}.npz")
    partial_path, final_path = os.fspath(partial_path), os.fspath(final_path)

    partial = None
    partial_error = None
    if os.path.isfile(partial_path):
        try:
            partial = _load_partial(
                version, partial_path, expected_run_signature)
        except (OSError, KeyError, TypeError, ValueError) as exc:
            partial_error = exc

    final = None
    final_error = None
    if os.path.isfile(final_path):
        # With no explicit signature, don't let an old final replace an active
        # new partial.  It is still useful to load it so a matching completed
        # artifact can take over during the final atomic handoff.
        final_expected = (expected_run_signature if expected_run_signature
                          is not None else (
                              None if partial is None else partial.run_signature))
        try:
            final = _load_final(version, final_path, final_expected)
        except (OSError, KeyError, TypeError, ValueError) as exc:
            final_error = exc

    if final is not None and partial is not None:
        # A deliberate --force rerun can have the same statistical signature
        # as an older completed artifact.  Prefer whichever atomic state was
        # published most recently: the new partial while the rerun is active,
        # then the final after the driver's final-write/partial-delete handoff.
        try:
            partial_mtime = os.stat(partial_path).st_mtime_ns
        except FileNotFoundError:
            # Normal final handoff: the driver writes the final artifact and
            # then removes the partial between this function's load and stat.
            return final
        try:
            final_mtime = os.stat(final_path).st_mtime_ns
        except FileNotFoundError:
            # A concurrent force rerun may move an old final aside after it was
            # loaded; the already validated active partial remains usable.
            return partial
        return final if final_mtime >= partial_mtime else partial
    if final is not None:
        return final
    if partial is not None:
        return partial
    if partial_error is not None:
        if final_error is not None:
            raise ValueError(
                f"invalid partial checkpoint ({partial_error}); invalid final "
                f"artifact ({final_error})") from partial_error
        raise ValueError(f"invalid partial checkpoint: {partial_error}") from partial_error
    if final_error is not None:
        raise ValueError(f"invalid final artifact: {final_error}") from final_error
    return None


def _validate_same_experiment(dgp: DgpCurves, progress: BoundProgress) -> None:
    if progress.settings is None:
        warnings.warn(
            "partial checkpoint predates authenticated progress metadata; "
            "the overlay cannot cross-check its experiment against the DGP cache",
            RuntimeWarning)
        return
    mismatches = []
    for key in ("version_label", "k", "n"):
        if progress.settings.get(key) != dgp.settings.get(key):
            mismatches.append(key)
    try:
        alpha_matches = bool(np.isclose(
            progress.settings.get("alpha"), dgp.settings.get("alpha"),
            rtol=0.0, atol=1e-12))
    except TypeError:
        alpha_matches = False
    if not alpha_matches:
        mismatches.append("alpha")
    try:
        bound_kappas = np.asarray(progress.settings["kappas"], dtype=float)
        dgp_kappas = np.asarray(dgp.settings["kappas"], dtype=float)
        kappas_match = (bound_kappas.shape == dgp_kappas.shape
                        and np.allclose(bound_kappas, dgp_kappas,
                                        rtol=0.0, atol=1e-12))
    except (KeyError, TypeError, ValueError):
        kappas_match = False
    if not kappas_match:
        mismatches.append("kappas")
    if (progress.betas[0] < dgp.betas[0] - 1e-12
            or progress.betas[-1] > dgp.betas[-1] + 1e-12):
        mismatches.append("beta-grid coverage")
    if mismatches:
        raise ValueError(
            "DGP cache and bound checkpoint describe different configurations: "
            + ", ".join(mismatches))


def _progress_metrics(
        dgp: DgpCurves,
        progress: Optional[BoundProgress]) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "completed_beta_count": 0,
        "total_beta_count": 0 if progress is None else int(progress.betas.size),
        "is_final": bool(progress is not None and progress.is_final),
        "cross_experiment_crossing_count": 0,
        "cross_experiment_min_gap": float("nan"),
        "same_experiment_min_certified_slack": float("nan"),
    }
    if progress is None:
        return metrics
    complete = np.isfinite(progress.bounds_confidence)
    metrics["completed_beta_count"] = int(np.count_nonzero(complete))
    if not np.any(complete):
        return metrics
    beta = progress.betas[complete]
    confidence = progress.bounds_confidence[complete]
    cp1 = np.interp(beta, dgp.betas, dgp.power_cp1)
    gap = confidence - cp1
    slack = confidence - progress.benchmark_lower_confidence[complete]
    metrics.update({
        "latest_beta": float(beta[-1]),
        "latest_confidence_upper": float(confidence[-1]),
        "latest_power_cp1": float(cp1[-1]),
        "latest_cross_experiment_gap": float(gap[-1]),
        "cross_experiment_crossing_count": int(np.count_nonzero(gap < -1e-12)),
        "cross_experiment_min_gap": float(np.min(gap)),
        "same_experiment_min_certified_slack": float(np.min(slack)),
    })
    return metrics


def build_progress_figure(
        dgp: DgpCurves,
        progress: Optional[BoundProgress] = None):
    """Build the exact local/W&B monitoring figure and summary metrics."""

    metrics = _progress_metrics(dgp, progress)
    figure, axis = plt.subplots(figsize=(10, 6))
    axis.plot(dgp.betas, dgp.power_chi2, color="tab:blue", linewidth=1.5,
              label=r"Feasible $\chi^2$ curve (finite-sample DGP)")
    axis.plot(dgp.betas, dgp.power_c1, color="tab:orange", linewidth=1.5,
              label=r"Feasible $\phi_{c_1}$ / power_c1 (finite-sample DGP)")
    axis.plot(dgp.betas, dgp.power_cp1, color="tab:red", linewidth=2.2,
              label=(r"Feasible $\phi_{c_{p-1}}$ / power_cp1 "
                     r"(finite-sample DGP)"))

    if progress is not None:
        complete = np.isfinite(progress.bounds_confidence)
        beta = progress.betas[complete]
        confidence = progress.bounds_confidence[complete]
        point = progress.bounds_point[complete]
        benchmark = progress.benchmark_lower_confidence[complete]
        confidence_level = None
        if progress.settings is not None:
            confidence_level = progress.settings.get("curve_confidence")
        level = ("confidence-valid" if confidence_level is None
                 else f"{float(confidence_level):.0%} simultaneous-MC")
        axis.plot(beta, confidence, color="green", marker="o", linewidth=2.4,
                  markersize=6, label=f"EMW upper ({level}; primary)")
        axis.plot(beta, point, color="green", marker=".", linestyle="--",
                  linewidth=1.4, label="EMW paper-style point estimate")
        axis.plot(beta, benchmark, color="purple", marker="^", linestyle=":",
                  linewidth=1.5,
                  label="Invariant benchmark lower (same experiment)")

        if beta.size:
            cp1 = np.interp(beta, dgp.betas, dgp.power_cp1)
            crossed = confidence < cp1 - 1e-12
            if np.any(crossed):
                axis.vlines(beta[crossed], confidence[crossed], cp1[crossed],
                            colors="crimson", linestyles="--", alpha=0.65)
                axis.scatter(
                    beta[crossed], confidence[crossed], marker="x", s=90,
                    linewidths=2.2, color="crimson", zorder=6,
                    label="Cross-experiment crossing (diagnostic only)")

    alpha = float(dgp.settings["alpha"])
    axis.axhline(alpha, color="gray", linestyle=":", linewidth=1.0,
                 label=rf"Size $\alpha={alpha:g}$")
    axis.set_xlabel(r"True $\beta$")
    axis.set_ylabel("Rejection probability / power")
    completed = metrics["completed_beta_count"]
    total = metrics["total_beta_count"]
    state = "waiting for checkpoint" if progress is None else (
        "final" if progress.is_final else f"live: {completed}/{total} betas")
    demo_prefix = ("SYNTHETIC W&B DEMO — NOT A SCIENTIFIC RESULT — "
                   if dgp.settings.get("demo") else "")
    axis.set_title(
        f"{demo_prefix}EMW power-bound progress — "
        f"{dgp.settings['version_label']} ({state})")
    plotted_max = max(
        float(np.max(dgp.power_chi2)), float(np.max(dgp.power_c1)),
        float(np.max(dgp.power_cp1)))
    if progress is not None:
        complete = np.isfinite(progress.bounds_confidence)
        if np.any(complete):
            plotted_max = max(
                plotted_max,
                float(np.max(progress.bounds_confidence[complete])),
                float(np.max(progress.bounds_point[complete])),
                float(np.max(
                    progress.benchmark_lower_confidence[complete])))
    axis.set_ylim(0.0, min(1.02, max(0.15, plotted_max + 0.05)))
    x_min = float(dgp.betas[0])
    x_max = float(dgp.betas[-1])
    if progress is not None:
        x_min = min(x_min, float(progress.betas[0]))
        x_max = max(x_max, float(progress.betas[-1]))
    axis.set_xlim(x_min, x_max)
    axis.grid(True, alpha=0.25)
    axis.text(
        0.01, 0.99,
        "DGP curves and EMW bounds are from different experiments.\n"
        "A crossing is an early diagnostic warning, not a formal bound violation.",
        transform=axis.transAxes, va="top", ha="left", fontsize=8.5,
        color="darkred", bbox=dict(facecolor="white", edgecolor="lightcoral",
                                    alpha=0.88, boxstyle="round,pad=0.35"))
    axis.legend(loc="best", fontsize=8)
    figure.tight_layout()
    return figure, metrics


def write_progress_plot(
        dgp: DgpCurves, progress: Optional[BoundProgress],
        output_path: str) -> dict[str, Any]:
    """Render and atomically publish the local PNG."""

    output_path = os.fspath(output_path)
    directory = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(directory, exist_ok=True)
    figure, metrics = build_progress_figure(dgp, progress)
    temporary = None
    try:
        handle = tempfile.NamedTemporaryFile(
            mode="wb", suffix=".png", prefix=".power-progress-",
            dir=directory, delete=False)
        temporary = handle.name
        handle.close()
        figure.savefig(temporary, dpi=140)
        os.replace(temporary, output_path)
        temporary = None
    finally:
        plt.close(figure)
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
    return metrics


def write_progress_values(
        dgp: DgpCurves, progress: Optional[BoundProgress],
        output_path: str) -> None:
    """Atomically write every plotted value in tidy, long-format CSV form."""

    output_path = os.fspath(output_path)
    directory = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(directory, exist_ok=True)
    temporary = None
    handle = None
    try:
        handle = tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="", suffix=".csv",
            prefix=".power-progress-", dir=directory, delete=False)
        temporary = handle.name
        writer = csv.DictWriter(handle, fieldnames=(
            "scope", "series", "beta", "value", "completed", "is_final",
            "bound_run_signature", "dgp_run_signature", "synthetic_demo"))
        writer.writeheader()

        dgp_scope = ("synthetic_demo" if dgp.settings.get("demo")
                     else "finite_sample_dgp")
        for series, values in (
                ("power_chi2", dgp.power_chi2),
                ("power_c1", dgp.power_c1),
                ("power_cp1", dgp.power_cp1)):
            for beta, value in zip(dgp.betas, values):
                writer.writerow(dict(
                    scope=dgp_scope, series=series, beta=f"{beta:.17g}",
                    value=f"{value:.17g}", completed="true",
                    is_final=str(bool(progress is not None
                                      and progress.is_final)).lower(),
                    bound_run_signature=(
                        "" if progress is None else progress.run_signature),
                    dgp_run_signature=dgp.run_signature,
                    synthetic_demo=str(bool(dgp.settings.get("demo"))).lower()))

        if progress is not None:
            complete = np.isfinite(progress.bounds_confidence)
            limit_scope = ("synthetic_demo" if (
                progress.settings is not None
                and progress.settings.get("demo")) else "limit_experiment")
            series_values = (
                ("bounds_confidence", progress.bounds_confidence),
                ("bounds_point", progress.bounds_point),
                ("benchmark_lower_confidence",
                 progress.benchmark_lower_confidence),
            )
            for series, values in series_values:
                for index, (beta, value) in enumerate(
                        zip(progress.betas, values)):
                    writer.writerow(dict(
                        scope=limit_scope, series=series,
                        beta=f"{beta:.17g}",
                        value=(f"{value:.17g}" if np.isfinite(value) else ""),
                        completed=str(bool(complete[index])).lower(),
                        is_final=str(bool(progress.is_final)).lower(),
                        bound_run_signature=progress.run_signature,
                        dgp_run_signature=dgp.run_signature,
                        synthetic_demo=str(bool(
                            progress.settings is not None
                            and progress.settings.get("demo"))).lower()))

            cp1_at_beta = np.interp(
                progress.betas, dgp.betas, dgp.power_cp1)
            gaps = progress.bounds_confidence - cp1_at_beta
            for series, values in (
                    ("power_cp1_interpolated_at_bound_beta", cp1_at_beta),
                    ("bounds_confidence_minus_power_cp1", gaps)):
                for index, (beta, value) in enumerate(
                        zip(progress.betas, values)):
                    writer.writerow(dict(
                        scope="cross_experiment_diagnostic", series=series,
                        beta=f"{beta:.17g}",
                        value=(f"{value:.17g}" if np.isfinite(value) else ""),
                        completed=str(bool(complete[index])).lower(),
                        is_final=str(bool(progress.is_final)).lower(),
                        bound_run_signature=progress.run_signature,
                        dgp_run_signature=dgp.run_signature,
                        synthetic_demo=str(bool(dgp.settings.get("demo"))).lower()))
        handle.flush()
        os.fsync(handle.fileno())
        handle.close()
        os.replace(temporary, output_path)
        temporary = None
    finally:
        if handle is not None and not handle.closed:
            handle.close()
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


class WandbLogger:
    """Small failure-isolating adapter around the optional W&B dependency."""

    def __init__(
            self, *, project: Optional[str], entity: Optional[str] = None,
            name: Optional[str] = None, run_id: Optional[str] = None,
            mode: str = "online"):
        self.project = project
        self.entity = entity
        self.name = name
        self.run_id = run_id
        self.mode = mode
        self.module = None
        self.run = None
        self.upload_failures = 0

    @property
    def requested(self) -> bool:
        return bool(self.project) and self.mode != "disabled"

    def start(self, dgp: DgpCurves, progress: BoundProgress) -> None:
        if not self.requested or self.run is not None:
            return
        try:
            module = importlib.import_module("wandb")
        except ImportError as exc:
            raise RuntimeError(
                "W&B monitoring was requested but 'wandb' is not installed; "
                "install it with `python3 -m pip install wandb`") from exc
        is_demo = bool(dgp.settings.get("demo"))
        run_id = self.run_id or (
            f"emw-demo-{progress.run_signature[:16]}" if is_demo
            else f"emw-{progress.run_signature[:16]}")
        default_prefix = ("[SYNTHETIC DEMO]" if is_demo else "EMW")
        name = self.name or (
            f"{default_prefix} {dgp.settings['version_label']} "
            f"{progress.run_signature[:8]}")
        try:
            run = module.init(
                project=self.project, entity=self.entity, name=name, id=run_id,
                resume="allow", mode=self.mode,
                config={
                    "version_label": dgp.settings["version_label"],
                    "bound_run_signature": progress.run_signature,
                    "dgp_run_signature": dgp.run_signature,
                    "dgp_cache": dgp.path,
                    "synthetic_demo": bool(dgp.settings.get("demo")),
                    "cross_experiment_overlay": True,
                    "cross_experiment_note": (
                        "DGP/EMW crossing is diagnostic, not a formal violation"),
                })
        except Exception as exc:
            raise RuntimeError(f"could not initialize W&B: {exc}") from exc
        self.module, self.run = module, run

    def log(self, output_path: str, metrics: Mapping[str, Any]) -> bool:
        if self.run is None:
            return False
        try:
            payload = dict(metrics)
            payload["power_progress"] = self.module.Image(output_path)
            self.run.log(payload)
            self.upload_failures = 0
            return True
        except Exception as exc:
            self.upload_failures += 1
            print(
                "WARNING: W&B upload failed; the local PNG is current and "
                f"the next checkpoint will retry (failure "
                f"{self.upload_failures}): {exc}")
            return False

    def finish(self) -> None:
        if self.run is None:
            return
        try:
            self.run.finish()
        except Exception as exc:
            print(f"WARNING: W&B finish failed: {exc}")
        finally:
            self.run = None


def _synthetic_demo_dgp(version: str) -> DgpCurves:
    """Create clearly labelled deterministic curves for telemetry testing."""

    betas = np.linspace(-2.0, 2.0, 81)
    scale = np.abs(betas / 2.0)
    positive = np.maximum(betas, 0.0) / 2.0
    power_chi2 = 0.05 + 0.055 * scale ** 1.5
    power_c1 = 0.05 + 0.085 * scale ** 1.4 + 0.008 * positive
    power_cp1 = 0.05 + 0.125 * scale ** 1.35 + 0.015 * positive
    settings = {
        "demo": True,
        "demo_notice": "synthetic telemetry test; not a scientific result",
        "version_label": version,
        "kappas": list(comparison.VERSION_LABELS[version]),
        "k": 7, "n": 250, "alpha": 0.05,
        "betas": betas.tolist(),
        "created_ns": time.time_ns(),
    }
    signature = hashlib.sha256(json.dumps(
        settings, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return DgpCurves(
        betas=betas, power_chi2=power_chi2, power_c1=power_c1,
        power_cp1=power_cp1, settings=settings,
        run_signature=signature, path="<synthetic-demo>")


def _synthetic_demo_progress(
        dgp: DgpCurves, completed: int, run_signature: str) -> BoundProgress:
    betas = np.linspace(-2.0, 2.0, 5)
    completed = min(max(int(completed), 0), betas.size)
    cp1 = np.interp(betas, dgp.betas, dgp.power_cp1)
    upper_target = np.minimum(cp1 + 0.05, 0.995)
    point_target = np.minimum(cp1 + 0.025, upper_target)
    benchmark_target = np.maximum(0.05, cp1 - 0.01)
    exact_null = np.isclose(betas, 0.0, rtol=0.0, atol=1e-15)
    upper_target[exact_null] = 0.05
    point_target[exact_null] = 0.05
    benchmark_target[exact_null] = 0.05
    arrays = [np.full(betas.size, np.nan) for _ in range(3)]
    for destination, source in zip(
            arrays, (upper_target, point_target, benchmark_target)):
        destination[:completed] = source[:completed]
    settings = dict(dgp.settings)
    settings.update(curve_confidence=0.95, beta_count=int(betas.size))
    return BoundProgress(
        betas=betas, bounds_confidence=arrays[0], bounds_point=arrays[1],
        benchmark_lower_confidence=arrays[2],
        run_signature=run_signature, source_path="<synthetic-demo>",
        is_final=completed == betas.size, settings=settings)


def _run_synthetic_demo(
        args, logger: WandbLogger) -> None:
    dgp = _synthetic_demo_dgp(args.version)
    bound_signature = hashlib.sha256(
        ("synthetic-bound:" + dgp.run_signature).encode()).hexdigest()
    demo_directory = os.path.join(args.version, "adaptive", "demo")
    demo_tag = bound_signature[:12]
    output = args.output or os.path.join(
        demo_directory, f"live_power_progress_demo_{demo_tag}.png")
    csv_output = args.csv_output or os.path.splitext(output)[0] + ".csv"
    print(
        "SYNTHETIC DEMO: these values test plotting/W&B only and must not be "
        "used as a power bound.")
    try:
        for completed in range(0, 6):
            progress = _synthetic_demo_progress(
                dgp, completed, bound_signature)
            metrics = write_progress_plot(dgp, progress, output)
            write_progress_values(dgp, progress, csv_output)
            logger.start(dgp, progress)
            logger.log(output, metrics)
            print(
                f"Demo update {completed}/5: wrote {output} and {csv_output}")
            if completed < 5 and args.demo_delay:
                time.sleep(args.demo_delay)
    finally:
        logger.finish()


def _fingerprint(progress: Optional[BoundProgress]) -> Any:
    if progress is None:
        return None
    complete = np.isfinite(progress.bounds_confidence)
    digest = hashlib.sha256()
    for value in (
            progress.betas, complete, progress.bounds_confidence[complete],
            progress.bounds_point[complete],
            progress.benchmark_lower_confidence[complete]):
        digest.update(np.ascontiguousarray(value).view(np.uint8))
    return progress.run_signature, progress.is_final, digest.hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Watch an adaptive EMW checkpoint, render a live feasible-curve "
            "overlay, and optionally publish it to Weights & Biases."))
    parser.add_argument(
        "--version", required=True, choices=list(comparison.VERSION_LABELS))
    parser.add_argument("--dgp-cache", default=None)
    parser.add_argument("--partial-path", default=None)
    parser.add_argument("--final-path", default=None)
    parser.add_argument("--output", default=None,
                        help="atomic local PNG destination")
    parser.add_argument("--csv-output", default=None,
                        help="atomic long-format CSV destination")
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--once", action="store_true",
                        help="render the current snapshot once and exit")
    parser.add_argument("--expected-run-signature", default=None,
                        help="optional 64-character ALFD run signature pin")
    parser.add_argument("--demo", action="store_true",
                        help="log a short, clearly synthetic W&B demo")
    parser.add_argument("--demo-delay", type=float, default=0.5,
                        help="seconds between synthetic demo updates")
    parser.add_argument("--wandb-project", default=None,
                        help="enable W&B uploads to this project")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--wandb-run-id", default=None)
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"),
                        default="online")
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    parser = _parser()
    args = parser.parse_args(argv)
    if not np.isfinite(args.poll_seconds) or args.poll_seconds <= 0.0:
        parser.error("--poll-seconds must be finite and positive")
    if args.expected_run_signature is not None:
        try:
            _validate_signature(args.expected_run_signature)
        except ValueError as exc:
            parser.error(str(exc))
    if args.wandb_mode != "disabled" and any((
            args.wandb_entity, args.wandb_name, args.wandb_run_id)) \
            and not args.wandb_project:
        parser.error("--wandb-project is required when W&B options are supplied")
    if args.wandb_mode == "offline" and not args.wandb_project:
        parser.error("--wandb-project is required for --wandb-mode offline")
    if not np.isfinite(args.demo_delay) or args.demo_delay < 0.0:
        parser.error("--demo-delay must be finite and nonnegative")
    if args.demo and any((
            args.dgp_cache, args.partial_path, args.final_path,
            args.expected_run_signature, args.once)):
        parser.error(
            "--demo cannot be combined with DGP/checkpoint paths, a run "
            "signature pin, or --once")
    if (args.demo and args.wandb_run_id is not None
            and not args.wandb_run_id.startswith("emw-demo-")):
        parser.error("a demo --wandb-run-id must start with 'emw-demo-'")

    logger = WandbLogger(
        project=args.wandb_project, entity=args.wandb_entity,
        name=args.wandb_name, run_id=args.wandb_run_id, mode=args.wandb_mode)
    if args.demo:
        _run_synthetic_demo(args, logger)
        return

    dgp = load_dgp_curves(args.version, args.dgp_cache)
    adaptive = os.path.join(args.version, "adaptive")
    output = args.output or os.path.join(
        adaptive, f"live_power_progress_{args.version}.png")
    csv_output = args.csv_output or os.path.splitext(output)[0] + ".csv"

    seen = object()
    pinned_signature = args.expected_run_signature
    try:
        while True:
            progress = load_bound_progress(
                args.version, partial_path=args.partial_path,
                final_path=args.final_path,
                expected_run_signature=pinned_signature)
            if progress is not None:
                _validate_same_experiment(dgp, progress)
                if pinned_signature is None:
                    pinned_signature = progress.run_signature
            fingerprint = _fingerprint(progress)
            if fingerprint != seen:
                metrics = write_progress_plot(dgp, progress, output)
                write_progress_values(dgp, progress, csv_output)
                seen = fingerprint
                if progress is None:
                    print(
                        f"Published {output} and {csv_output}; waiting for an "
                        "ALFD checkpoint.")
                else:
                    print(
                        f"Published {output} and {csv_output}: "
                        f"{metrics['completed_beta_count']}/"
                        f"{metrics['total_beta_count']} betas complete; "
                        f"cross-experiment diagnostic crossings="
                        f"{metrics['cross_experiment_crossing_count']}.")
                    logger.start(dgp, progress)
                    logger.log(output, metrics)

            if args.once or (progress is not None and progress.is_final):
                break
            time.sleep(args.poll_seconds)
    except KeyboardInterrupt:
        print("Watcher stopped by user; the ALFD calculation was not touched.")
    finally:
        logger.finish()


if __name__ == "__main__":
    main()
