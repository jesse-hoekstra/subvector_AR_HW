"""Compute a power-bound curve across SSH nodes sharing one existing null bank.

Prepare once, run a different shard on each node, and merge completed beta
points. This driver only reads the null bank; it never constructs one.
"""

import argparse
from contextlib import contextmanager
import csv
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import time
import zipfile

import numpy as np

import alfd_eigval as alfd


SCHEMA_VERSION = 1
PRODUCER = "power_bound_cluster.py"
_SCALARS = ("bound", "bound_se", "mixture_power", "mixture_power_se", "epsilon_grid")
_VECTORS = ("weights", "log_weights", "fit_rejection_probabilities",
            "grid_rejection_probabilities")
_READ_ERRORS = (OSError, ValueError, TypeError, KeyError, EOFError, zipfile.BadZipFile)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _signature(value):
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _driver_sha256():
    return alfd._sha256_file(os.path.abspath(__file__))


@contextmanager
def _exclusive_lock(path):
    """Keep lock files persistent: deleting them can split an advisory lock."""
    with open(path, "a+") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another process holds {path}; use a different shard.") from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _atomic_write(path, writer):
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", prefix=path.name + ".tmp-",
                                         dir=path.parent, delete=False) as handle:
            temporary = handle.name
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)


def _load_bank(path, version):
    """Read and authenticate an existing bank, with no build-or-load fallback."""
    path = Path(path)
    try:
        with np.load(path, allow_pickle=False) as archive:
            diagnostics_json = str(archive["mhg_diagnostics_json"].item())
            bank = alfd.PooledISBank(
                **{name: np.asarray(archive[name]) for name in
                   ("grid", "eigs", "log_f", "log_q", "base_weights", "strata")},
                **{name: int(archive[name].item()) for name in
                   ("n_per_stratum", "sampling_seed", "k_eff")},
                **{name: str(archive[name].item()) for name in
                   ("role", "bank_id", "experiment_signature", "settings_json",
                    "content_signature")},
                mhg_diagnostics=json.loads(diagnostics_json))
        settings = alfd._authenticated_pooled_bank_settings(bank)
        dimension = len(alfd.VERSION_LABELS[version])
        if (bank.role != "gkm" or bank.k_eff != 7
                or bank.grid.shape[1] != dimension):
            raise ValueError("bank role, k_eff, or nuisance dimension differs")
        if settings.get("provenance") != alfd._gkm_provenance(dimension):
            raise ValueError("bank code, library, or environment provenance differs")
        if diagnostics_json != alfd._canonical_pooled_mhg_diagnostics(
                bank.mhg_diagnostics, bank.log_f.size):
            raise ValueError("bank MHG diagnostics are not canonical")
        rebuilt = alfd._pooled_bank_settings(
            bank.grid, bank.role, bank.k_eff, bank.n_per_stratum,
            bank.sampling_seed, settings["M_start"], settings["M_step"],
            settings["M_max"], settings["mhg_tol"], settings["provenance"])
        if settings != rebuilt:
            raise ValueError("bank settings differ from the scientific implementation")
        for name in ("grid", "eigs", "log_f", "log_q", "base_weights", "strata"):
            getattr(bank, name).setflags(write=False)
        return bank, settings
    except _READ_ERRORS as exc:
        raise RuntimeError(f"Cannot trust existing null bank {path}: {exc}") from exc


def _select_bank(version, bank, bank_dir):
    if bank is not None and bank_dir is not None:
        raise ValueError("provide either bank or bank_dir, not both")
    if bank is not None:
        path = Path(bank).resolve()
        loaded, settings = _load_bank(path, version)
        return path, loaded, settings
    folder = Path(bank_dir) if bank_dir is not None else Path(version) / "gkm_direct"
    candidates = sorted(folder.glob("pooled_gkm_*.npz"))
    compatible, failures = [], []
    # Do not hold several large bank arrays in memory during discovery.
    for path in candidates:
        try:
            loaded, settings = _load_bank(path, version)
            compatible.append(path.resolve())
            del loaded
        except RuntimeError as exc:
            failures.append(str(exc))
    if len(compatible) != 1:
        detail = "; ".join(failures)
        raise RuntimeError(
            f"Expected exactly one compatible existing null bank in {folder}; "
            f"found {len(compatible)}. Use --bank PATH to select it explicitly."
            + (f" {detail}" if detail else ""))
    loaded, settings = _load_bank(compatible[0], version)
    return compatible[0], loaded, settings


def _run_settings(version, profile, seed, beta_count, n_power, n_iter, alpha):
    if version not in alfd.VERSION_LABELS:
        raise ValueError(f"unknown version: {version}")
    if profile not in ("production", "reference"):
        raise ValueError("profile must be production or reference")
    seed = alfd._validated_integer("seed", seed, minimum=0)
    beta_count = alfd._validated_integer("beta_count", beta_count, minimum=3)
    if beta_count % 2 != 1:
        raise ValueError("beta_count must be odd so beta=0 is included")
    n_power = (100000 if profile == "reference" else 50000) if n_power is None else n_power
    n_power = alfd._validated_integer("n_power", n_power, minimum=2)
    n_iter = alfd._validated_integer("n_iter", 600 if n_iter is None else n_iter)
    if not np.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0,1)")
    return dict(version=version, profile=profile, seed=seed, beta_count=beta_count,
                n_power=n_power, n_iter=n_iter, alpha=float(alpha), k=7, n=250,
                kappas=np.asarray(alfd.VERSION_LABELS[version], dtype=float).tolist())


def _points(settings, shards, bank_seed):
    points, nonnull_index = [], 0
    for index, beta in enumerate(np.linspace(-2.0, 2.0, settings["beta_count"])):
        seed, shard = None, None
        if beta != 0.0:
            seed = int(np.random.SeedSequence(
                [settings["seed"], 0x42455441, index]).generate_state(1, dtype=np.uint32)[0])
            if seed == bank_seed:
                seed = (seed + 1) % (2 ** 32)
            shard = nonnull_index % shards
            nonnull_index += 1
        ncp = np.maximum(alfd.asymptotic_ncp_eigenvalues(
            beta, settings["kappas"], settings["k"], settings["n"]), 0.0)
        points.append(dict(index=index, beta=float(beta), ncp=ncp.tolist(),
                           seed=seed, shard=shard))
    return points


def _load_manifest(directory, check_environment=True):
    try:
        manifest = json.loads((Path(directory) / "manifest.json").read_text())
        if manifest["manifest_signature"] != _signature({
                key: value for key, value in manifest.items() if key != "manifest_signature"}):
            raise ValueError("manifest signature differs")
        if manifest["schema_version"] != SCHEMA_VERSION or manifest["producer"] != PRODUCER:
            raise ValueError("unsupported manifest schema or producer")
        settings = manifest["settings"]
        rebuilt = _run_settings(**{key: settings[key] for key in
                                  ("version", "profile", "seed", "beta_count",
                                   "n_power", "n_iter", "alpha")})
        if settings != rebuilt:
            raise ValueError("run settings are inconsistent")
        shards = alfd._validated_integer("shard_count", manifest["shard_count"])
        if shards >= settings["beta_count"]:
            raise ValueError("more shards than nonzero beta points")
        bank_settings = manifest["bank"]["settings"]
        if (bank_settings["k_eff"] != 7 or bank_settings["role"] != "gkm"
                or bank_settings["provenance"] != manifest["provenance"]
                or len(bank_settings["grid"][0]) != len(settings["kappas"])):
            raise ValueError("inconsistent bank experiment")
        if _signature(bank_settings) != manifest["bank"]["bank_id"]:
            raise ValueError("bank identity differs from its settings")
        if manifest["points"] != _points(settings, shards, bank_settings["seed"]):
            raise ValueError("beta points, alternative NCPs, seeds, or shard assignments differ")
        if check_environment:
            if manifest["driver_sha256"] != _driver_sha256():
                raise ValueError("power_bound_cluster.py differs from the prepare code")
            if manifest["provenance"] != alfd._gkm_provenance(len(settings["kappas"])):
                raise ValueError("code, library, or environment provenance differs from prepare")
        return manifest
    except _READ_ERRORS as exc:
        raise RuntimeError(f"Cannot trust power preparation in {directory}: {exc}") from exc


def _beta_path(directory, index):
    return Path(directory) / f"beta_{index:05d}.npz"


def _result_arrays(manifest, index, result):
    point = manifest["points"][index]
    diagnostics = alfd._json_safe(dict(
        power_seed=point["seed"], mixture_rule=result.mixture_rule,
        grid_rule=result.grid_rule, importance=result.importance_diagnostics,
        mhg=result.mhg_diagnostics))
    # The scientific routine permits infinite tail thresholds; retain its JSON
    # convention and authenticate the complete diagnostic string verbatim.
    diagnostics_json = json.dumps(diagnostics, sort_keys=True, separators=(",", ":"))
    arrays = {name: np.asarray(getattr(result, name), dtype=np.float64)
              for name in _SCALARS + _VECTORS}
    arrays.update(
        index=np.array(index, dtype=np.int64), beta=np.array(point["beta"]),
        ncp=np.asarray(point["ncp"], dtype=np.float64),
        seed=np.array(-1 if point["seed"] is None else point["seed"], dtype=np.int64),
        shard=np.array(-1 if point["shard"] is None else point["shard"], dtype=np.int64),
        manifest_signature=np.array(manifest["manifest_signature"]),
        bank_id=np.array(manifest["bank"]["bank_id"]),
        bank_content_signature=np.array(manifest["bank"]["content_signature"]),
        fit_iterations=np.array(result.fit_iterations, dtype=np.int64),
        max_m_used=np.array(result.mhg_diagnostics["max_order"], dtype=np.int64),
        diagnostics_json=np.array(diagnostics_json))
    arrays["content_signature"] = np.array(alfd._pooled_bank_content_signature(**arrays))
    return arrays


def _validate_beta(arrays, manifest, index):
    expected = set(_SCALARS + _VECTORS) | {
        "index", "beta", "ncp", "seed", "shard", "manifest_signature", "bank_id",
        "bank_content_signature", "fit_iterations", "max_m_used", "diagnostics_json",
        "content_signature"}
    if set(arrays) != expected:
        raise ValueError("missing or unexpected beta result fields")
    content = alfd._pooled_bank_content_signature(**{
        key: value for key, value in arrays.items() if key != "content_signature"})
    if arrays["content_signature"].item() != content:
        raise ValueError("beta result content hash differs")
    point = manifest["points"][index]
    identity = dict(index=index, beta=point["beta"],
                    seed=-1 if point["seed"] is None else point["seed"],
                    shard=-1 if point["shard"] is None else point["shard"],
                    manifest_signature=manifest["manifest_signature"],
                    bank_id=manifest["bank"]["bank_id"],
                    bank_content_signature=manifest["bank"]["content_signature"])
    for name, value in identity.items():
        if arrays[name].shape != () or arrays[name].item() != value:
            raise ValueError(f"beta result {name} differs from the manifest")
    if not np.array_equal(arrays["ncp"], np.asarray(point["ncp"])):
        raise ValueError("alternative NCP differs from the manifest")
    H = len(manifest["bank"]["settings"]["grid"])
    for name in _SCALARS + _VECTORS:
        shape = () if name in _SCALARS else (H,)
        if (arrays[name].dtype != np.dtype("float64") or arrays[name].shape != shape
                or not np.all(np.isfinite(arrays[name]))):
            raise ValueError(f"invalid {name} values or shape")
    if (any(not 0 <= arrays[name].item() <= 1 for name in _SCALARS)
            or arrays["bound"] > arrays["mixture_power"] + 1e-12
            or not np.isclose(arrays["epsilon_grid"],
                              arrays["mixture_power"] - arrays["bound"], atol=1e-12)):
        raise ValueError("inconsistent power estimates")
    if (np.any(arrays["weights"] < 0) or not np.isclose(arrays["weights"].sum(), 1.0)
            or not np.allclose(np.exp(arrays["log_weights"]), arrays["weights"],
                               rtol=1e-12, atol=0.0)
            or np.any(arrays["fit_rejection_probabilities"] < 0)
            or np.any(arrays["grid_rejection_probabilities"] < 0)
            or np.max(arrays["grid_rejection_probabilities"])
            > manifest["settings"]["alpha"] + 2e-12):
        raise ValueError("invalid fitted weights or grid rejection probabilities")
    diagnostics_json = str(arrays["diagnostics_json"].item())
    diagnostics = json.loads(diagnostics_json)
    if diagnostics_json != json.dumps(diagnostics, sort_keys=True, separators=(",", ":")):
        raise ValueError("beta diagnostics are not canonical")
    if (diagnostics["power_seed"] != point["seed"]
            or not isinstance(diagnostics["importance"], dict)
            or arrays["max_m_used"].item() != diagnostics["mhg"]["max_order"]):
        raise ValueError("beta diagnostics differ from result identity")
    expected_iterations = 0 if point["beta"] == 0.0 else manifest["settings"]["n_iter"]
    if arrays["fit_iterations"].shape != () or arrays["fit_iterations"].item() != expected_iterations:
        raise ValueError("fit iteration count differs from settings")
    for name in ("mixture_rule", "grid_rule"):
        rule = diagnostics[name]
        if (set(rule) != {"threshold", "tie_probability", "empirical_size", "method"}
                or not isinstance(rule["method"], str)
                or np.isnan(rule["threshold"])
                or not 0 <= rule["tie_probability"] <= 1
                or not np.isfinite(rule["empirical_size"])):
            raise ValueError("invalid saved tail rule")
    if point["beta"] == 0.0:
        exact = _result_arrays(manifest, index,
                               alfd._exact_gkm_result(manifest["settings"]["alpha"], H))
        if any(not np.array_equal(arrays[key], value) for key, value in exact.items()):
            raise ValueError("beta zero differs from exact null randomization")
    return arrays


def _load_beta(directory, manifest, index):
    path = _beta_path(directory, index)
    try:
        with np.load(path, allow_pickle=False) as archive:
            arrays = {name: np.asarray(archive[name]) for name in archive.files}
        return _validate_beta(arrays, manifest, index)
    except _READ_ERRORS as exc:
        raise RuntimeError(f"Cannot trust beta result {path}: {exc}") from exc


def _save_beta(directory, manifest, index, result):
    arrays = _validate_beta(_result_arrays(manifest, index, result), manifest, index)
    path = _beta_path(directory, index)
    _atomic_write(path, lambda handle: np.savez(handle, **arrays))
    return path


def prepare_power(directory, *, version="10015", profile="reference", shards=4,
                  seed=42, beta_count=17, bank=None, bank_dir=None,
                  n_power=None, n_iter=None, alpha=0.05):
    """Authenticate an existing bank and freeze the curve, seeds, and assignments."""
    settings = _run_settings(version, profile, seed, beta_count, n_power, n_iter, alpha)
    shards = alfd._validated_integer("shards", shards)
    if shards >= settings["beta_count"]:
        raise ValueError("shards cannot exceed the number of nonzero beta points")
    path, loaded, bank_settings = _select_bank(version, bank, bank_dir)
    manifest = dict(
        schema_version=SCHEMA_VERSION, producer=PRODUCER, settings=settings,
        shard_count=shards, driver_sha256=_driver_sha256(),
        provenance=bank_settings["provenance"],
        bank=dict(path=str(path), file_sha256=alfd._sha256_file(str(path)),
                  bank_id=loaded.bank_id, content_signature=loaded.content_signature,
                  settings=bank_settings,
                  diagnostics_json=alfd._canonical_pooled_mhg_diagnostics(
                      loaded.mhg_diagnostics, loaded.log_f.size)),
        points=_points(settings, shards, loaded.sampling_seed))
    manifest["manifest_signature"] = _signature(manifest)
    directory = Path(directory)
    # Invalid inputs/banks and incompatible existing preparations fail before
    # creating an output directory or touching a lock file.
    if (directory / "manifest.json").exists():
        if _load_manifest(directory) != manifest:
            raise RuntimeError("Existing preparation has different settings; use a new directory.")
    elif directory.exists() and any(directory.glob("beta_*.npz")):
        raise RuntimeError("Beta results exist without a manifest; use a new directory.")
    directory.mkdir(parents=True, exist_ok=True)
    with _exclusive_lock(directory / "prepare.lock"):
        if (directory / "manifest.json").exists():
            if _load_manifest(directory) != manifest:
                raise RuntimeError("Existing preparation has different settings; use a new directory.")
        else:
            _atomic_write(directory / "manifest.json", lambda handle: handle.write(
                (_canonical(manifest) + "\n").encode("utf-8")))
        zero = settings["beta_count"] // 2
        with _exclusive_lock(_beta_path(directory, zero).with_suffix(".lock")):
            if _beta_path(directory, zero).exists():
                _load_beta(directory, manifest, zero)
            else:
                _save_beta(directory, manifest, zero,
                           alfd._exact_gkm_result(alpha, len(loaded.grid)))
    H = len(loaded.grid)
    pairs = H * loaded.n_per_stratum + (H + 1) * settings["n_power"]
    print(f"Prepared {settings['beta_count']} beta points; beta=0 saved exactly as alpha={alpha}.\n"
          f"Using existing bank {path}\n"
          f"H={H}, N0={loaded.n_per_stratum:,} per null, N1={settings['n_power']:,}, "
          f"iterations={settings['n_iter']}; {pairs:,} density pairs per nonzero beta.", flush=True)
    for shard in range(shards):
        points = [point for point in manifest["points"] if point["shard"] == shard]
        pending = sum(not _beta_path(directory, point["index"]).exists() for point in points)
        print(f"  shard {shard}: beta=" + ", ".join(f"{point['beta']:+.2f}" for point in points)
              + f"; {pending * pairs:,} remaining density pairs", flush=True)
    return manifest


def _verified_manifest_bank(manifest):
    saved = manifest["bank"]
    if alfd._sha256_file(saved["path"]) != saved["file_sha256"]:
        raise RuntimeError("Existing null bank file differs from the prepared file hash")
    bank, settings = _load_bank(saved["path"], manifest["settings"]["version"])
    if (bank.bank_id != saved["bank_id"] or bank.content_signature != saved["content_signature"]
            or settings != saved["settings"]
            or alfd._canonical_pooled_mhg_diagnostics(bank.mhg_diagnostics, bank.log_f.size)
            != saved["diagnostics_json"]):
        raise RuntimeError("Existing null bank differs from the prepared experiment")
    return bank


def run_worker(directory, shard, workers=48):
    """Compute assigned betas sequentially, with local multiprocessing per beta."""
    shard = alfd._validated_integer("shard", shard, minimum=0)
    workers = alfd._validated_integer("workers", workers)
    manifest = _load_manifest(directory)
    if shard >= manifest["shard_count"]:
        raise ValueError(f"shard must be in 0..{manifest['shard_count'] - 1}")
    completed = []
    with _exclusive_lock(Path(directory) / f"shard_{shard:05d}.lock"):
        bank = _verified_manifest_bank(manifest)
        settings, bs = manifest["settings"], manifest["bank"]["settings"]
        points = [point for point in manifest["points"] if point["shard"] == shard]
        print(f"Shard {shard}: {len(points)} beta points, {workers} local workers; "
              f"loaded existing bank {bank.bank_id[:16]}", flush=True)
        for position, point in enumerate(points, 1):
            index = point["index"]
            path = _beta_path(directory, index)
            with _exclusive_lock(path.with_suffix(".lock")):
                if path.exists():
                    _load_beta(directory, manifest, index)
                    print(f"[{position}/{len(points)}] beta={point['beta']:+.2f}: "
                          "already complete and verified", flush=True)
                    completed.append(path)
                    continue
                start = time.monotonic()
                print(f"[{position}/{len(points)}] starting beta={point['beta']:+.2f}; "
                      f"global index={index}, power seed={point['seed']}", flush=True)
                result = alfd.gkm_eigval_bound_from_pooled_bank(
                    kappas_alt=point["ncp"], bank=bank, k_eff=settings["k"],
                    alpha=settings["alpha"], n_sim_power=settings["n_power"],
                    n_iter=settings["n_iter"], seed=point["seed"], verbose=True,
                    n_workers=workers, M_trunc=bs["M_start"], M_step=bs["M_step"],
                    M_max=bs["M_max"], mhg_tol=bs["mhg_tol"])
                completed.append(_save_beta(directory, manifest, index, result))
                print(f"[{position}/{len(points)}] saved beta={point['beta']:+.2f}: "
                      f"bound={result.bound:.6f}, MC SE={result.bound_se:.6f}; "
                      f"elapsed={alfd._format_duration(time.monotonic() - start)}", flush=True)
    print(f"Shard {shard} complete: {len(completed)} beta results verified.", flush=True)
    return completed


def power_status(directory):
    """Read-only verification of complete, missing, and invalid beta artifacts."""
    manifest = _load_manifest(directory, check_environment=False)
    result = dict(beta_count=manifest["settings"]["beta_count"],
                  shard_count=manifest["shard_count"], complete=[], missing=[], invalid={})
    for point in manifest["points"]:
        index = point["index"]
        if not _beta_path(directory, index).exists():
            result["missing"].append(index)
            continue
        try:
            _load_beta(directory, manifest, index)
            result["complete"].append(index)
        except RuntimeError as exc:
            result["invalid"][index] = str(exc)
    return result


def merge_power(directory):
    """Verify every beta and write the curve NPZ, CSV, and PNG atomically."""
    directory = Path(directory)
    manifest = _load_manifest(directory)
    with _exclusive_lock(directory / "merge.lock"):
        bank = _verified_manifest_bank(manifest)
        status = power_status(directory)
        if status["missing"] or status["invalid"]:
            raise RuntimeError(f"Cannot merge: missing beta indices {status['missing']}; "
                               f"invalid beta results {status['invalid']}")
        results = [_load_beta(directory, manifest, point["index"])
                   for point in manifest["points"]]
        settings, bs = manifest["settings"], manifest["bank"]["settings"]
        mapping = dict(bound="bounds", bound_se="bounds_se", weights="fitted_weights",
                       log_weights="fitted_log_weights")
        payload = {mapping.get(name, name): np.stack([result[name] for result in results])
                   for name in _SCALARS + _VECTORS +
                   ("fit_iterations", "max_m_used", "diagnostics_json")}
        payload.update(
            schema_version=np.array(SCHEMA_VERSION), producer=np.array(PRODUCER),
            algorithm=np.array("distributed_" + manifest["provenance"]["algorithm"]),
            calibration_method=np.array(alfd.CALIBRATION_METHOD),
            bound_kind=np.array(alfd.BOUND_KIND), version_label=np.array(settings["version"]),
            profile=np.array(settings["profile"]), run_signature=np.array(manifest["manifest_signature"]),
            settings_json=np.array(_canonical(settings)), manifest_json=np.array(_canonical(manifest)),
            provenance_json=np.array(_canonical(manifest["provenance"])),
            density_accuracy_scope=np.array("adaptive_empirical_tail_criterion"),
            betas=np.array([point["beta"] for point in manifest["points"]]),
            ncp=np.array([point["ncp"] for point in manifest["points"]]),
            power_seeds=np.array([-1 if point["seed"] is None else point["seed"]
                                  for point in manifest["points"]], dtype=np.int64),
            kappas=np.array(settings["kappas"]), k=np.array(settings["k"]), n=np.array(settings["n"]),
            alpha=np.array(settings["alpha"]), seed=np.array(settings["seed"]),
            n_fit=np.array(bank.n_per_stratum), n_power=np.array(settings["n_power"]),
            n_iter=np.array(settings["n_iter"]), common_null_grid=bank.grid,
            common_grid_size=np.array(len(bank.grid)), bank_id=np.array(bank.bank_id),
            bank_content_signature=np.array(bank.content_signature),
            bank_file_sha256=np.array(manifest["bank"]["file_sha256"]),
            bank_settings_json=np.array(bank.settings_json),
            bank_mhg_diagnostics_json=np.array(manifest["bank"]["diagnostics_json"]),
            M_start=np.array(bs["M_start"]), M_step=np.array(bs["M_step"]),
            M_max=np.array(bs["M_max"]), mhg_rtol=np.array(bs["mhg_tol"]),
            beta_content_signatures=np.array([result["content_signature"].item() for result in results]))
        stem = settings["version"]
        path = directory / f"gkm_eigval_{stem}.npz"
        _atomic_write(path, lambda handle: np.savez(handle, **payload))
        buffer = io.StringIO(newline="")
        writer = csv.writer(buffer)
        columns = ("betas", "bounds", "bounds_se", "mixture_power", "mixture_power_se", "epsilon_grid")
        writer.writerow(("beta", "bound", "bound_se", "mixture_power", "mixture_power_se", "epsilon_grid"))
        writer.writerows(zip(*(payload[name] for name in columns)))
        csv_path = directory / f"gkm_bounds_{stem}.csv"
        _atomic_write(csv_path, lambda handle: handle.write(buffer.getvalue().encode("utf-8")))
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        figure = Figure(figsize=(7, 4.5), tight_layout=True)
        FigureCanvasAgg(figure)
        axes = figure.subplots()
        axes.errorbar(payload["betas"], payload["bounds"], yerr=payload["bounds_se"],
                      fmt="o-", capsize=3, label="GKM bound (±1 Monte Carlo SE)")
        axes.axhline(settings["alpha"], color="gray", linestyle="--", label=f"alpha={settings['alpha']:g}")
        axes.set(xlabel="Beta", ylabel="Power bound", ylim=(0, 1),
                 title=f"mW={len(settings['kappas'])}, kappa={settings['kappas']}")
        axes.grid(alpha=0.2)
        axes.legend(fontsize=9)
        png_path = directory / f"power_bound_{stem}.png"
        _atomic_write(png_path, lambda handle: figure.savefig(handle, format="png", dpi=160))
    print(f"Merged {len(results)} verified beta points:\n  {path}\n  {csv_path}\n  {png_path}", flush=True)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="freeze the curve using one existing null bank")
    prepare.add_argument("--directory", required=True)
    prepare.add_argument("--version", choices=alfd.VERSION_LABELS, default="10015")
    prepare.add_argument("--profile", choices=("production", "reference"), default="reference")
    prepare.add_argument("--shards", type=int, default=4)
    prepare.add_argument("--seed", type=int, default=42)
    prepare.add_argument("--beta-count", type=int, default=17)
    prepare.add_argument("--alpha", type=float, default=0.05)
    prepare.add_argument("--n-power", type=int)
    prepare.add_argument("--n-iter", type=int)
    bank_options = prepare.add_mutually_exclusive_group()
    bank_options.add_argument("--bank", help="exact existing pooled_gkm_*.npz bank path")
    bank_options.add_argument("--bank-dir", help="find exactly one compatible bank (default VERSION/gkm_direct)")
    worker = commands.add_parser("worker", help="compute one shard with local CPU workers")
    worker.add_argument("--directory", required=True)
    worker.add_argument("--shard", type=int, required=True)
    worker.add_argument("--workers", type=int, default=48)
    for command, help_text in (("status", "verify beta completion without computing"),
                               ("merge", "merge every completed beta into NPZ, CSV, and PNG")):
        subparser = commands.add_parser(command, help=help_text)
        subparser.add_argument("--directory", required=True)
    arguments = vars(parser.parse_args(argv))
    command = arguments.pop("command")
    try:
        if command == "prepare":
            prepare_power(**arguments)
        elif command == "worker":
            run_worker(**arguments)
        elif command == "merge":
            merge_power(**arguments)
        else:
            result = power_status(**arguments)
            print(json.dumps(result, indent=2))
            if result["invalid"]:
                return 1
    except (RuntimeError, ValueError, OSError) as exc:
        parser.exit(2, f"error: {exc}\n")
    return 0


if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    raise SystemExit(main())
