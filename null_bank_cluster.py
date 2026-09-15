"""Build one reusable null bank across SSH-accessible nodes sharing a folder.

Prepare samples once, run a distinct shard on each node, then merge. Each shard
uses the existing local multiprocessing implementation. No scheduler or MPI is
needed; all machines must use the same code, libraries, and Python environment.
"""

import argparse
from collections import Counter
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile
import zipfile

import numpy as np

import alfd_eigval as alfd


SCHEMA_VERSION = 1


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _signature(value):
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _driver_sha256():
    return alfd._sha256_file(os.path.abspath(__file__))


@contextmanager
def _exclusive_lock(path):
    """Advisory shared-filesystem lock, automatically released after a crash."""
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
    """Publish a complete file with a same-directory atomic rename."""
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


def _prepare_settings(*, version, profile, seed, n_fit, grid_shapes,
                      grid_strengths, grid_max_strength, m_start, m_step,
                      m_max, mhg_rtol):
    if version not in alfd.VERSION_LABELS:
        raise ValueError(f"unknown version: {version}")
    if profile not in ("production", "reference"):
        raise ValueError("profile must be production or reference")
    kappas = np.asarray(alfd.VERSION_LABELS[version], dtype=float)
    config = alfd.ALLOWED_CONFIGS[tuple(kappas)]
    n_fit = (10000 if profile == "reference" else 2000) if n_fit is None else n_fit
    n_fit = alfd._validated_integer("n_fit", n_fit, minimum=2)
    seed = alfd._validated_integer("seed", seed, minimum=0)
    grid_shapes = (5 if len(kappas) == 2 else 9) if grid_shapes is None else grid_shapes
    grid_shapes = alfd._validated_integer("grid_shapes", grid_shapes)
    grid_strengths = alfd._validated_integer("grid_strengths", grid_strengths)
    if not np.isfinite(grid_max_strength) or grid_max_strength <= 0.1:
        raise ValueError("grid_max_strength must be finite and greater than 0.1")
    m_start = config["M_start"] if m_start is None else m_start
    m_start = alfd._validated_integer("m_start", m_start)
    m_step = alfd._validated_integer("m_step", m_step)
    m_max = alfd._validated_integer("m_max", m_max)
    if m_start > m_max:
        raise ValueError("m_start must not exceed m_max")
    if not np.isfinite(mhg_rtol) or not 1e-13 <= mhg_rtol < 1.0:
        raise ValueError("mhg_rtol must be finite and lie in [1e-13, 1)")

    nuisance_path = np.asarray([
        np.maximum(alfd.asymptotic_ncp_eigenvalues(beta, kappas, 7, 250), 0.0)[:-1]
        for beta in np.linspace(-2.0, 2.0, alfd.GRID_DESIGN_BETA_COUNT)
    ])
    grid_builder = alfd.common_null_grid_2d if len(kappas) == 2 else alfd.common_null_grid_3d
    grid = grid_builder(nuisance_path, kappas, standard_points=config["standard"],
                        n_shapes=grid_shapes, n_strengths=grid_strengths,
                        max_strength=grid_max_strength)
    bank_seed = int(np.random.SeedSequence([seed, 0x474B4D34]).generate_state(
        1, dtype=np.uint32)[0])
    settings = alfd._pooled_bank_settings(
        grid, "gkm", 7, n_fit, bank_seed, m_start, m_step, m_max, mhg_rtol,
        metadata=alfd._gkm_provenance(len(kappas)))
    request = dict(version=version, profile=profile, seed=seed, n_fit=n_fit,
                   grid_shapes=grid_shapes, grid_strengths=grid_strengths,
                   grid_max_strength=float(grid_max_strength), m_start=m_start,
                   m_step=m_step, m_max=m_max, mhg_rtol=float(mhg_rtol))
    return settings, request


def _load_manifest(directory, check_environment=True):
    directory = Path(directory)
    try:
        manifest = json.loads((directory / "manifest.json").read_text())
        if not isinstance(manifest, dict):
            raise ValueError("manifest must be a JSON object")
        unsigned = {key: value for key, value in manifest.items()
                    if key != "manifest_signature"}
        if manifest["manifest_signature"] != _signature(unsigned):
            raise ValueError("manifest signature differs")
        if manifest["schema_version"] != SCHEMA_VERSION:
            raise ValueError("unsupported manifest schema")
        settings = manifest["settings"]
        version = manifest["version"]
        dimension = len(alfd.VERSION_LABELS[version])
        grid = alfd._validated_null_grid(settings["grid"], dimension, "shared null grid")
        sample_count = len(grid) * alfd._validated_integer(
            "n_per_stratum", settings["n_per_stratum"], minimum=2)
        shards = alfd._validated_integer("shard_count", manifest["shard_count"])
        if shards > sample_count:
            raise ValueError("there are more shards than samples")
        if settings["role"] != "gkm" or settings["k_eff"] != 7:
            raise ValueError("unexpected pooled-bank role or dimension")
        rebuilt = alfd._pooled_bank_settings(
            grid, settings["role"], settings["k_eff"], settings["n_per_stratum"],
            settings["seed"], settings["M_start"], settings["M_step"],
            settings["M_max"], settings["mhg_tol"], settings["provenance"])
        if settings != rebuilt:
            raise ValueError("pooled-bank settings are inconsistent")
        eigs = np.load(directory / "eigs.npy", allow_pickle=False, mmap_mode="r")
        if (eigs.dtype != np.dtype("float64")
                or eigs.shape != (sample_count, dimension + 1)
                or not np.all(np.isfinite(eigs)) or np.any(eigs < 0.0)
                or np.any(np.diff(eigs, axis=1) > 0.0)):
            raise ValueError("invalid canonical eigenvalue sample")
        if alfd._pooled_bank_content_signature(eigs=eigs) != manifest["eigs_signature"]:
            raise ValueError("canonical eigenvalue content hash differs")
        if check_environment:
            if manifest["driver_sha256"] != _driver_sha256():
                raise ValueError("null_bank_cluster.py differs from the prepare code")
            if settings["provenance"] != alfd._gkm_provenance(dimension):
                raise ValueError("code, library, or environment provenance differs from prepare")
        return manifest, eigs
    except (OSError, ValueError, TypeError, KeyError, EOFError, zipfile.BadZipFile) as exc:
        raise RuntimeError(f"Cannot trust shared null-bank preparation in {directory}: {exc}") from exc


def prepare_bank(directory, *, version="10015", profile="reference", shards=1,
                 seed=42, n_fit=None, grid_shapes=None, grid_strengths=7,
                 grid_max_strength=100.0, m_start=None,
                 m_step=alfd.MHG_DEFAULT_STEP, m_max=alfd.MHG_DEFAULT_MAX,
                 mhg_rtol=alfd.MHG_CONV_TOL):
    """Create immutable samples/settings once, without any density evaluation."""
    shards = alfd._validated_integer("shards", shards)
    settings, request = _prepare_settings(
        version=version, profile=profile, seed=seed, n_fit=n_fit,
        grid_shapes=grid_shapes, grid_strengths=grid_strengths,
        grid_max_strength=grid_max_strength, m_start=m_start, m_step=m_step,
        m_max=m_max, mhg_rtol=mhg_rtol)
    sample_count = len(settings["grid"]) * settings["n_per_stratum"]
    if shards > sample_count:
        raise ValueError("shards cannot exceed the number of null samples")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with _exclusive_lock(directory / "prepare.lock"):
        if (directory / "manifest.json").exists():
            manifest, _ = _load_manifest(directory)
            if (manifest["settings"] != settings or manifest["request"] != request
                    or manifest["shard_count"] != shards):
                raise RuntimeError("Existing preparation has different settings; use a new directory.")
            print(f"Reusing prepared bank in {directory}", flush=True)
            return manifest
        if (directory / "eigs.npy").exists() or any(directory.glob("shard_*.npz")):
            raise RuntimeError("Incomplete preparation without manifest; use a new directory.")
        eigs = alfd._sample_pooled_null_eigenvalues(
            settings["grid"], settings["k_eff"], settings["n_per_stratum"], settings["seed"])
        manifest = dict(schema_version=SCHEMA_VERSION, version=version,
                        settings=settings, request=request, shard_count=shards,
                        driver_sha256=_driver_sha256(),
                        eigs_signature=alfd._pooled_bank_content_signature(eigs=eigs))
        manifest["manifest_signature"] = _signature(manifest)
        _atomic_write(directory / "eigs.npy", lambda handle: np.save(handle, eigs, allow_pickle=False))
        _atomic_write(directory / "manifest.json",
                      lambda handle: handle.write((_canonical(manifest) + "\n").encode("utf-8")))
    print(f"Prepared {len(settings['grid'])} nulls, {sample_count:,} samples, "
          f"{sample_count * len(settings['grid']):,} density pairs across {shards} shards "
          f"in {directory}", flush=True)
    return manifest


def _shard_path(directory, shard):
    return Path(directory) / f"shard_{shard:05d}.npz"


def _shard_indices(manifest, shard):
    count = len(manifest["settings"]["grid"]) * manifest["settings"]["n_per_stratum"]
    return np.arange(shard, count, manifest["shard_count"], dtype=np.int64)


def _load_shard(directory, manifest, shard):
    path = _shard_path(directory, shard)
    try:
        with path.open("rb") as handle, np.load(handle, allow_pickle=False) as archive:
            index = np.asarray(archive["shard_index"])
            signature = np.asarray(archive["manifest_signature"])
            indices = np.asarray(archive["indices"])
            log_f = np.asarray(archive["log_f"])
            diagnostics_json = str(np.asarray(archive["mhg_diagnostics_json"]).item())
            diagnostics = json.loads(diagnostics_json)
            content_signature = str(np.asarray(archive["content_signature"]).item())
        if (index.shape != () or not np.issubdtype(index.dtype, np.integer)
                or int(index) != shard or signature.shape != ()
                or str(signature.item()) != manifest["manifest_signature"]):
            raise ValueError("shard identity differs from the manifest")
        expected_indices = _shard_indices(manifest, shard)
        if (indices.dtype != np.dtype("int64")
                or not np.array_equal(indices, expected_indices)):
            raise ValueError("shard sample indices differ")
        if (log_f.dtype != np.dtype("float64")
                or log_f.shape != (len(manifest["settings"]["grid"]), len(indices))
                or not np.all(np.isfinite(log_f))):
            raise ValueError("invalid shard density matrix")
        canonical_diagnostics = alfd._canonical_pooled_mhg_diagnostics(diagnostics, log_f.size)
        if canonical_diagnostics != diagnostics_json:
            raise ValueError("shard diagnostics are not canonical")
        calculated = alfd._pooled_bank_content_signature(
            shard_index=index, manifest_signature=signature, indices=indices,
            log_f=log_f, diagnostics_json=diagnostics_json)
        if content_signature != calculated:
            raise ValueError("shard numeric content hash differs")
        return indices, log_f, diagnostics
    except (OSError, ValueError, TypeError, KeyError, EOFError, zipfile.BadZipFile) as exc:
        raise RuntimeError(f"Cannot trust null-bank shard {path}: {exc}") from exc


def run_worker(directory, shard, workers=1):
    """Evaluate one strided sample shard, or validate and reuse its completed file."""
    shard = alfd._validated_integer("shard", shard, minimum=0)
    workers = alfd._validated_integer("workers", workers)
    manifest, eigs = _load_manifest(directory)
    if shard >= manifest["shard_count"]:
        raise ValueError(f"shard must be in 0..{manifest['shard_count'] - 1}")
    path = _shard_path(directory, shard)
    with _exclusive_lock(path.with_suffix(".lock")):
        if path.exists():
            _load_shard(directory, manifest, shard)
            print(f"Shard {shard} already complete and verified: {path}", flush=True)
            return path
        settings = manifest["settings"]
        indices = _shard_indices(manifest, shard)
        omegas = np.column_stack((settings["grid"], np.zeros(len(settings["grid"]))))
        print(f"Shard {shard}/{manifest['shard_count'] - 1}: {len(indices):,} samples, "
              f"{len(indices) * len(omegas):,} density pairs, {workers} local workers", flush=True)
        log_f, diagnostics = alfd.log_eigval_density_partial(
            eigs[indices], omegas, settings["k_eff"] / 2.0,
            M_trunc=settings["M_start"], chunk_size=100,
            progress_label=f"null-shard-{shard}", n_workers=workers,
            M_step=settings["M_step"], M_max=settings["M_max"],
            mhg_tol=settings["mhg_tol"], return_diagnostics=True)
        diagnostics_json = alfd._canonical_pooled_mhg_diagnostics(diagnostics, log_f.size)
        arrays = dict(shard_index=np.array(shard, dtype=np.int64),
                      manifest_signature=np.array(manifest["manifest_signature"]),
                      indices=indices, log_f=log_f)
        content_signature = alfd._pooled_bank_content_signature(
            **arrays, diagnostics_json=diagnostics_json)
        _atomic_write(path, lambda handle: np.savez(
            handle, **arrays, mhg_diagnostics_json=np.array(diagnostics_json),
            content_signature=np.array(content_signature)))
        _load_shard(directory, manifest, shard)
    print(f"Shard {shard} complete: {path}", flush=True)
    return path


def _reload_bank(settings, cache_dir):
    return alfd.build_or_load_pooled_is_bank(
        settings["grid"], settings["k_eff"], settings["n_per_stratum"], settings["seed"],
        M_start=settings["M_start"], M_step=settings["M_step"], M_max=settings["M_max"],
        mhg_tol=settings["mhg_tol"], role=settings["role"], cache_dir=str(cache_dir),
        cache_metadata=settings["provenance"])


def merge_bank(directory, cache_dir=None):
    """Authenticate every shard and publish the normal reusable pooled-bank cache."""
    manifest, eigs = _load_manifest(directory)
    settings = manifest["settings"]
    cache_dir = Path(cache_dir) if cache_dir is not None else Path(manifest["version"]) / "gkm_direct"
    with _exclusive_lock(Path(directory) / "merge.lock"):
        missing = [shard for shard in range(manifest["shard_count"])
                   if not _shard_path(directory, shard).exists()]
        if missing:
            raise RuntimeError(f"Cannot merge: missing shards {missing}")
        log_f = np.empty((len(settings["grid"]), len(eigs)), dtype=float)
        diagnostics = dict(pairs=0, raw_evaluations=0, order_counts=Counter(),
                           max_order=0, max_remainder_ratio=0.0)
        for shard in range(manifest["shard_count"]):
            indices, shard_log_f, shard_diagnostics = _load_shard(directory, manifest, shard)
            log_f[:, indices] = shard_log_f
            alfd._merge_mhg_diagnostics(diagnostics, shard_diagnostics)
        diagnostics["order_counts"] = dict(sorted(diagnostics["order_counts"].items()))
        bank = alfd._finalize_pooled_is_bank(settings, eigs, log_f, diagnostics)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"pooled_gkm_{bank.bank_id[:16]}.npz"
        with _exclusive_lock(cache_path.with_suffix(".merge.lock")):
            if not cache_path.exists():
                alfd._finalize_pooled_is_bank(settings, eigs, log_f, diagnostics,
                                             cache_dir=str(cache_dir))
            reloaded = _reload_bank(settings, cache_dir)
            if reloaded.content_signature != bank.content_signature:
                raise RuntimeError("Existing bank content differs from the authenticated shards.")
    print(f"Merged and verified reusable null bank: {cache_path}", flush=True)
    return reloaded


def bank_status(directory):
    """Return verified completed/missing/invalid shard indices without computing."""
    manifest, _ = _load_manifest(directory, check_environment=False)
    result = dict(shard_count=manifest["shard_count"], complete=[], missing=[], invalid={})
    for shard in range(manifest["shard_count"]):
        if not _shard_path(directory, shard).exists():
            result["missing"].append(shard)
            continue
        try:
            _load_shard(directory, manifest, shard)
            result["complete"].append(shard)
        except RuntimeError as exc:
            result["invalid"][shard] = str(exc)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="sample once; no density evaluation")
    prepare.add_argument("--directory", required=True)
    prepare.add_argument("--version", choices=alfd.VERSION_LABELS, default="10015")
    prepare.add_argument("--profile", choices=("production", "reference"), default="reference")
    prepare.add_argument("--shards", type=int, required=True)
    prepare.add_argument("--seed", type=int, default=42)
    prepare.add_argument("--n-fit", type=int)
    prepare.add_argument("--grid-shapes", type=int)
    prepare.add_argument("--grid-strengths", type=int, default=7)
    prepare.add_argument("--grid-max-strength", type=float, default=100.0)
    prepare.add_argument("--m-start", type=int)
    prepare.add_argument("--m-step", type=int, default=alfd.MHG_DEFAULT_STEP)
    prepare.add_argument("--m-max", type=int, default=alfd.MHG_DEFAULT_MAX)
    prepare.add_argument("--mhg-rtol", type=float, default=alfd.MHG_CONV_TOL)
    worker = commands.add_parser("worker", help="evaluate one shard using local CPU workers")
    worker.add_argument("--directory", required=True)
    worker.add_argument("--shard", type=int, required=True, help="zero-based shard index")
    worker.add_argument("--workers", type=int, default=1)
    merge = commands.add_parser("merge", help="verify and merge every shard into a normal bank")
    merge.add_argument("--directory", required=True)
    merge.add_argument("--cache-dir")
    status = commands.add_parser("status", help="verify shard completion without computing")
    status.add_argument("--directory", required=True)
    arguments = vars(parser.parse_args(argv))
    command = arguments.pop("command")
    try:
        if command == "prepare":
            prepare_bank(**arguments)
        elif command == "worker":
            run_worker(**arguments)
        elif command == "merge":
            merge_bank(**arguments)
        else:
            result = bank_status(**arguments)
            print(json.dumps(result, indent=2))
            if result["invalid"]:
                return 1
    except (RuntimeError, ValueError, OSError) as exc:
        parser.exit(2, f"error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
