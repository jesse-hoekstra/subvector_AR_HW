"""Add beta points to a completed GKM curve without rebuilding its null bank.

Keep alfd_eigval.py and the original scientific environment unchanged: their
provenance identifies the cached bank. This driver writes into a separate
directory and resumes only its own, compatible partial result.
"""

import argparse
import csv
import fcntl
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
import scipy

import alfd_eigval as alfd
import watch_power_progress as watcher


POINT_FIELDS = (
    "ncp", "bounds", "bounds_se", "mixture_power", "mixture_power_se",
    "epsilon_grid", "fitted_weights", "fitted_log_weights",
    "fit_rejection_probabilities", "grid_rejection_probabilities",
    "fit_iterations", "max_m_used", "diagnostics_json",
)
PROVENANCE_FIELDS = (
    "schema_version", "algorithm", "producer", "calibration_method",
    "source_sha256", "mhg_core_sha256", "mhg_library_sha256",
    "mhg_build_source_sha256", "python_version", "numpy_version",
    "scipy_version", "platform",
)


def _read_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def _signature(settings):
    canonical = json.dumps(settings, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def _validate_points(payload, settings):
    """Reject damaged rows before treating them as completed calculations."""
    watcher._validate_bound_arrays(
        payload["betas"], payload["bounds"], payload["bounds_se"],
        require_complete=False)
    count = len(payload["betas"])
    support = len(settings["common_grid"])
    complete = np.isfinite(payload["bounds"])
    for key in POINT_FIELDS:
        value = payload[key]
        shape = (count, 4) if key == "ncp" else (
            (count, support) if key in (
                "fitted_weights", "fitted_log_weights",
                "fit_rejection_probabilities", "grid_rejection_probabilities")
            else (count,))
        if value.shape != shape:
            raise ValueError(f"invalid {key} shape: {value.shape}, expected {shape}")
        if key == "diagnostics_json":
            if value.dtype.kind != "U":
                raise ValueError("diagnostics_json must contain Unicode strings")
            for row in value[complete]:
                if not isinstance(json.loads(str(row)), dict):
                    raise ValueError("invalid completed-point diagnostics")
        elif key in ("fit_iterations", "max_m_used"):
            if value.dtype.kind not in "iu" or np.any(value[complete] < 0):
                raise ValueError(f"invalid {key}")
        elif key == "fitted_log_weights":
            if (value.dtype.kind != "f" or np.any(np.isnan(value[complete]))
                    or np.any(np.isposinf(value[complete]))):
                raise ValueError("invalid completed fitted_log_weights")
        else:
            if value.dtype.kind != "f" or not np.all(np.isfinite(value[complete])):
                raise ValueError(f"nonfinite or non-floating completed {key}")


def _load_existing_bank(source, settings):
    """Load and authenticate the bank; there is deliberately no build path."""
    bank_id = str(source["bank_id"].item())
    if len(bank_id) != 64 or any(c not in "0123456789abcdef" for c in bank_id):
        raise ValueError("invalid source bank_id")
    path = Path(settings["_source_directory"]) / f"pooled_gkm_{bank_id[:16]}.npz"
    if not path.is_file():
        raise FileNotFoundError(f"Required existing null bank is missing: {path}")
    with np.load(path, allow_pickle=False) as archive:
        fields = {
            key: archive[key].copy() for key in
            ("grid", "eigs", "log_f", "log_q", "base_weights", "strata")
        }
        fields.update({key: archive[key].item() for key in (
            "n_per_stratum", "role", "bank_id", "sampling_seed", "k_eff",
            "experiment_signature", "settings_json", "content_signature")})
        fields["mhg_diagnostics"] = json.loads(
            str(archive["mhg_diagnostics_json"].item()))
    bank = alfd.PooledISBank(**fields)
    alfd._validate_pooled_is_bank(bank)
    bank_settings = alfd._authenticated_pooled_bank_settings(bank)
    provenance = {key: settings[key] for key in PROVENANCE_FIELDS}
    expected = alfd._pooled_bank_settings(
        settings["common_grid"], "gkm", settings["k"], settings["n_fit"],
        settings["bank_seed"], settings["M_start"], settings["M_step"],
        settings["M_max"], settings["mhg_rtol"], provenance)
    if (bank_settings != expected or bank.bank_id != bank_id
            or bank.content_signature != str(source["bank_content_signature"].item())):
        raise ValueError("null bank does not match the completed source result")
    print(f"Reusing verified null bank: {path}", flush=True)
    return bank


def _new_seeds(source, settings, additions):
    # The original driver seeds by beta index. Reusing refined-grid indices
    # would collide with the original nine streams, so use a separate namespace
    # tied to the beta value and explicitly avoid every saved seed.
    reserved = {int(settings["bank_seed"])}
    for raw in source["diagnostics_json"]:
        seed = json.loads(str(raw)).get("power_seed")
        if seed is not None:
            reserved.add(int(seed))
    seeds = {}
    for beta in additions:
        nonce = 0
        while True:
            identity = f"gkm-refinement-v1:{settings['seed']}:{float(beta).hex()}:{nonce}"
            seed = int.from_bytes(hashlib.sha256(identity.encode()).digest()[:4], "big")
            if seed not in reserved:
                break
            nonce += 1
        reserved.add(seed)
        seeds[float(beta).hex()] = seed
    return seeds


def _prepare_payload(source, settings, betas, source_path):
    old_indices = np.searchsorted(betas, source["betas"])
    if (np.any(old_indices >= len(betas))
            or not np.array_equal(betas[old_indices], source["betas"])):
        raise ValueError("the refined beta grid must contain every original beta exactly")
    additions = np.setdiff1d(betas, source["betas"])
    if not len(additions):
        raise ValueError("the requested grid adds no beta values")
    settings = dict(settings, beta_count=len(betas))
    settings["refinement"] = dict(
        driver=Path(__file__).name,
        driver_sha256=alfd._sha256_file(__file__),
        parent_run_signature=str(source["run_signature"].item()),
        parent_sha256=alfd._sha256_file(str(source_path)),
        added_betas=additions.tolist(),
        new_beta_seeds=_new_seeds(source, settings, additions),
    )
    payload = {key: value.copy() for key, value in source.items()}
    payload["settings_json"] = np.array(json.dumps(settings, sort_keys=True))
    payload["run_signature"] = np.array(_signature(settings))
    payload["betas"] = betas.copy()
    for key in POINT_FIELDS:
        original = source[key]
        fill = "" if original.dtype.kind == "U" else (
            0 if original.dtype.kind in "iu" else np.nan)
        expanded = np.full((len(betas),) + original.shape[1:], fill, dtype=original.dtype)
        expanded[old_indices] = original
        payload[key] = expanded
    for beta in additions:
        index = int(np.searchsorted(betas, beta))
        payload["ncp"][index] = np.maximum(alfd.asymptotic_ncp_eigenvalues(
            beta, source["kappas"], settings["k"], settings["n"]), 0.0)
    return payload, settings, old_indices


def _load_resume(path, initial, source, old_indices, version, *, is_final):
    watcher._load_snapshot(
        version, str(path), str(initial["run_signature"].item()), is_final=is_final)
    saved = _read_npz(path)
    if saved.keys() != initial.keys():
        raise ValueError("refinement checkpoint fields differ")
    for key in initial:
        if key in POINT_FIELDS:
            if saved[key].dtype != initial[key].dtype:
                raise ValueError(f"refinement checkpoint dtype differs: {key}")
            if not np.array_equal(saved[key][old_indices], source[key]):
                raise ValueError(f"refinement checkpoint changed an original {key} row")
        elif not np.array_equal(saved[key], initial[key]):
            raise ValueError(f"refinement checkpoint metadata differs: {key}")
    if not np.array_equal(saved["ncp"], initial["ncp"]):
        raise ValueError("refinement checkpoint changed alternative eigenvalues")
    settings = json.loads(str(saved["settings_json"].item()))
    _validate_points(saved, settings)
    for beta, raw, complete in zip(
            saved["betas"], saved["diagnostics_json"], np.isfinite(saved["bounds"])):
        seed = settings["refinement"]["new_beta_seeds"].get(float(beta).hex())
        if complete and seed is not None and json.loads(str(raw)).get("power_seed") != seed:
            raise ValueError("refinement checkpoint changed a midpoint RNG seed")
    return saved


def _record_result(payload, index, result, seed):
    mapping = dict(
        bounds="bound", bounds_se="bound_se", mixture_power="mixture_power",
        mixture_power_se="mixture_power_se", epsilon_grid="epsilon_grid",
        fitted_weights="weights", fitted_log_weights="log_weights",
        fit_rejection_probabilities="fit_rejection_probabilities",
        grid_rejection_probabilities="grid_rejection_probabilities",
        fit_iterations="fit_iterations")
    for key, attribute in mapping.items():
        payload[key][index] = getattr(result, attribute)
    payload["max_m_used"][index] = int(result.mhg_diagnostics["max_order"])
    record = json.dumps(alfd._json_safe(dict(
        power_seed=seed, mixture_rule=result.mixture_rule,
        grid_rule=result.grid_rule, importance=result.importance_diagnostics,
        mhg=result.mhg_diagnostics)), sort_keys=True)
    if len(record) > payload["diagnostics_json"].dtype.itemsize // 4:
        raise ValueError("per-beta diagnostics exceed checkpoint field capacity")
    payload["diagnostics_json"][index] = record


def _write_csv(path, payload):
    temporary = str(path) + ".tmp"
    with open(temporary, "w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["beta", "bound", "bound_se", "completed"])
        for beta, bound, se in zip(payload["betas"], payload["bounds"], payload["bounds_se"]):
            writer.writerow([float(beta), float(bound), float(se), bool(np.isfinite(bound))])
    os.replace(temporary, path)


def run(args):
    source_path = Path(args.source or (
        f"{args.version}/gkm_direct/gkm_eigval_{args.version}.npz")).resolve()
    progress = watcher._load_final(args.version, str(source_path), None)
    source = _read_npz(source_path)
    settings = progress.settings
    _validate_points(source, settings)
    if not np.array_equal(source["common_null_grid"], np.asarray(settings["common_grid"])):
        raise ValueError("source common_null_grid differs from its settings")
    current_runtime = dict(python_version=sys.version, numpy_version=np.__version__,
                           scipy_version=scipy.__version__, platform=platform.platform())
    for key, value in current_runtime.items():
        if settings.get(key) != value:
            raise ValueError(f"{key} differs from the original run; use its environment")
    library = "libmhg.dylib" if sys.platform == "darwin" else "libmhg.so"
    alfd._verify_mhg_build_provenance(alfd.MHG_DIR, library)
    betas = np.linspace(-2.0, 2.0, args.beta_count)
    initial, refined_settings, old_indices = _prepare_payload(source, settings, betas, source_path)
    output_dir = Path(args.output_dir or source_path.parent / "refined").resolve()
    final_path = output_dir / f"gkm_eigval_{args.version}.npz"
    partial_path = output_dir / f"gkm_eigval_{args.version}.partial.npz"
    csv_path = output_dir / f"gkm_bounds_{args.version}.csv"
    if output_dir == source_path.parent:
        raise ValueError("--output-dir must differ from the original result directory")
    if final_path.exists():
        _load_resume(final_path, initial, source, old_indices, args.version, is_final=True)
        print(f"Compatible complete refinement already exists: {final_path}; nothing to do.")
        return
    payload = (_load_resume(partial_path, initial, source, old_indices, args.version,
                            is_final=False) if partial_path.exists() else initial)
    pending = np.flatnonzero(~np.isfinite(payload["bounds"]))
    print(f"Preserving all {len(source['betas'])} original beta points.")
    print(f"Pending beta values ({len(pending)}): {payload['betas'][pending].tolist()}")
    print(f"Merged {len(betas)}-point result: {final_path}")
    print(f"Using original N0={settings['n_fit']}, N1={settings['n_power']}, O={settings['n_iter']}.")
    bank = _load_existing_bank(source, dict(settings, _source_directory=str(source_path.parent)))
    if args.preflight_only:
        print("Preflight passed; no simulation or output files written.")
        return
    if not args.acknowledge_expensive:
        raise ValueError("pass --acknowledge-expensive to calculate the pending beta values")
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / ".refinement.lock", "a") as lock, \
            open(output_dir / "bound_run.log", "a", buffering=1) as log:
        stdout, stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = alfd._Tee(stdout, log), alfd._Tee(stderr, log)
        try:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError(f"another refinement is writing to {output_dir}") from exc
            # Re-read after acquiring the lock: another writer might have
            # completed a point during bank validation above.
            if final_path.exists():
                _load_resume(final_path, initial, source, old_indices, args.version, is_final=True)
                print(f"Compatible complete refinement already exists: {final_path}; nothing to do.")
                return
            if partial_path.exists():
                payload = _load_resume(partial_path, initial, source, old_indices,
                                       args.version, is_final=False)
            pending = np.flatnonzero(~np.isfinite(payload["bounds"]))
            alfd._atomic_savez(str(partial_path), **payload)
            _write_csv(csv_path, payload)
            for index in pending:
                beta = float(betas[index])
                seed = refined_settings["refinement"]["new_beta_seeds"][beta.hex()]
                print(f"========== new beta = {beta:+.2f}; seed={seed} ==========", flush=True)
                started = time.time()
                result = alfd.gkm_eigval_bound_from_pooled_bank(
                    kappas_alt=payload["ncp"][index], bank=bank, k_eff=settings["k"],
                    alpha=settings["alpha"], n_sim_power=settings["n_power"],
                    n_iter=settings["n_iter"], seed=seed, verbose=True,
                    n_workers=args.workers, M_trunc=settings["M_start"],
                    M_step=settings["M_step"], M_max=settings["M_max"],
                    mhg_tol=settings["mhg_rtol"])
                _record_result(payload, index, result, seed)
                _validate_points(payload, refined_settings)
                alfd._atomic_savez(str(partial_path), **payload)
                _write_csv(csv_path, payload)
                print(f"Saved beta={beta:+.2f}: bound={result.bound:.5f}, "
                      f"MC SE={result.bound_se:.5f}; "
                      f"runtime={alfd._format_duration(time.time() - started)}", flush=True)
            alfd._atomic_savez(str(final_path), **payload)
            partial_path.unlink()
            print(f"Saved complete merged curve: {final_path}\nNumeric CSV: {csv_path}", flush=True)
        finally:
            sys.stdout, sys.stderr = stdout, stderr


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", default="352515", choices=list(alfd.VERSION_LABELS))
    parser.add_argument("--source", help="completed original NPZ; its null bank must be alongside it")
    parser.add_argument("--output-dir", help="separate destination (default: original directory/refined)")
    parser.add_argument("--beta-count", type=int, default=17)
    parser.add_argument("--workers", type=int, default=min(os.cpu_count() or 1, 16))
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--acknowledge-expensive", action="store_true")
    args = parser.parse_args(argv)
    if args.beta_count < 3 or args.beta_count % 2 != 1 or args.workers < 1:
        parser.error("--beta-count must be odd and >= 3; --workers must be positive")
    run(args)


if __name__ == "__main__":
    mp.freeze_support()
    main()
