"""Plot local DGP simulations alongside a completed distributed power bound.

Reads saved results only. No null bank, native MHG library, or simulation is
needed. Bound provenance is checked internally, not against the viewing machine.
"""

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

import new_power_comparison as comparison


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _signature(value):
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _payload_signature(diagnostics_json=None, **arrays):
    """Portable implementation of the frozen power driver's array hash format."""
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


def _read_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name].copy() for name in archive.files}


def _scalar(arrays, name):
    if arrays[name].shape != ():
        raise ValueError(f"{name} must be a scalar")
    return arrays[name].item()


def _validate_series(betas, values, name, probability=True):
    if values.shape != betas.shape or not np.all(np.isfinite(values)):
        raise ValueError(f"invalid {name} shape or nonfinite values")
    if np.any(values < 0) or (probability and np.any(values > 1)):
        raise ValueError(f"invalid {name} range")


def load_dgp(path):
    arrays = _read_npz(path)
    settings = json.loads(_scalar(arrays, "settings_json"))
    args = {name: settings[name] for name in (
        "version_label", "kappas", "k", "n", "alpha", "betas",
        "num_simulations", "base_seed", "chunk_size")}
    loaded = comparison.load_compatible_dgp_cache(path, **args)
    version = settings["version_label"]
    if version not in comparison.VERSION_LABELS or not np.array_equal(
            settings["kappas"], comparison.VERSION_LABELS[version]):
        raise ValueError("DGP configuration differs from its version label")
    return dict(settings=settings, run_signature=_scalar(arrays, "run_signature"),
                **dict(zip(("betas", "power_chi2", "power_c1", "power_cp1"), loaded)))


def load_bound(path):
    arrays = _read_npz(path)
    if (_scalar(arrays, "schema_version") != 1
            or _scalar(arrays, "producer") != "power_bound_cluster.py"):
        raise ValueError("Expected a completed power_bound_cluster.py merged result")
    manifest = json.loads(_scalar(arrays, "manifest_json"))
    signature = manifest["manifest_signature"]
    if signature != _signature({key: value for key, value in manifest.items()
                                if key != "manifest_signature"}):
        raise ValueError("bound manifest signature differs")
    settings = manifest["settings"]
    if (_scalar(arrays, "run_signature") != signature
            or _scalar(arrays, "settings_json") != _canonical(settings)
            or manifest["producer"] != "power_bound_cluster.py"
            or manifest["schema_version"] != 1):
        raise ValueError("bound settings or producer differ from its manifest")
    mw = len(settings["kappas"])
    expected_algorithm = f"gkm_eigval_mw{mw}_adaptive_v4"
    if (_scalar(arrays, "algorithm") != "distributed_" + expected_algorithm
            or manifest["provenance"]["algorithm"] != expected_algorithm
            or _scalar(arrays, "provenance_json") != _canonical(manifest["provenance"])
            or _scalar(arrays, "calibration_method") != "gkm_step6_reused_pooled_bank"
            or _scalar(arrays, "bound_kind") != "gkm_d3_2_grid_adjusted_mc_power_bound"):
        raise ValueError("unrecognized bound algorithm or inconsistent provenance")
    for key in ("k", "n", "alpha", "seed", "profile", "n_power", "n_iter"):
        if _scalar(arrays, key) != settings[key]:
            raise ValueError(f"bound {key} differs from manifest")
    if (_scalar(arrays, "version_label") != settings["version"]
            or not np.array_equal(arrays["kappas"], settings["kappas"])):
        raise ValueError("bound configuration differs from manifest")
    for key in ("bank_id", "bank_content_signature", "bank_file_sha256"):
        manifest_key = {"bank_content_signature": "content_signature",
                        "bank_file_sha256": "file_sha256"}.get(key, key)
        if _scalar(arrays, key) != manifest["bank"][manifest_key]:
            raise ValueError(f"bound {key} differs from manifest")
    betas = arrays["betas"]
    points = manifest["points"]
    if (betas.ndim != 1 or len(betas) != settings["beta_count"]
            or not np.all(np.isfinite(betas)) or np.any(np.diff(betas) <= 0)
            or not np.array_equal(betas, [point["beta"] for point in points])
            or not np.array_equal(arrays["ncp"], [point["ncp"] for point in points])):
        raise ValueError("bound beta grid or alternative NCPs differ from manifest")
    for key in ("bounds", "mixture_power", "epsilon_grid"):
        _validate_series(betas, arrays[key], key)
    for key in ("bounds_se", "mixture_power_se"):
        _validate_series(betas, arrays[key], key, probability=False)
    if (np.any(arrays["bounds"] > arrays["mixture_power"] + 1e-12)
            or not np.allclose(arrays["epsilon_grid"],
                               arrays["mixture_power"] - arrays["bounds"],
                               rtol=0, atol=1e-12)):
        raise ValueError("bound power estimates are inconsistent")

    # Reconstruct the exact per-beta payloads, so changing any result array
    # after the remote merge is detected without consulting the remote bank.
    mapping = dict(bounds="bound", bounds_se="bound_se",
                   fitted_weights="weights", fitted_log_weights="log_weights")
    result_keys = ("bounds", "bounds_se", "mixture_power", "mixture_power_se",
                   "epsilon_grid", "fitted_weights", "fitted_log_weights",
                   "fit_rejection_probabilities", "grid_rejection_probabilities",
                   "fit_iterations", "max_m_used", "diagnostics_json")
    for key in result_keys + ("beta_content_signatures", "power_seeds"):
        if arrays[key].ndim < 1 or arrays[key].shape[0] != len(betas):
            raise ValueError(f"incomplete bound array {key}")
    for index, point in enumerate(points):
        seed = -1 if point["seed"] is None else point["seed"]
        if point["index"] != index or arrays["power_seeds"][index] != seed:
            raise ValueError("bound index or seed differs from manifest")
        row = {mapping.get(key, key): np.asarray(arrays[key][index])
               for key in result_keys}
        row.update(index=np.array(index, dtype=np.int64),
                   beta=np.array(point["beta"]), ncp=np.asarray(point["ncp"], dtype=float),
                   seed=np.array(seed, dtype=np.int64),
                   shard=np.array(-1 if point["shard"] is None else point["shard"], dtype=np.int64),
                   manifest_signature=np.array(signature),
                   bank_id=np.array(manifest["bank"]["bank_id"]),
                   bank_content_signature=np.array(manifest["bank"]["content_signature"]))
        if _payload_signature(**row) != arrays["beta_content_signatures"][index]:
            raise ValueError(f"bound beta index {index} content signature differs")
    return dict(settings=settings, run_signature=signature, betas=betas,
                bounds=arrays["bounds"], bounds_se=arrays["bounds_se"])


def _same_experiment(dgp, bound):
    left, right = dgp["settings"], bound["settings"]
    if (left["version_label"] != right["version"]
            or any(left[key] != right[key] for key in ("k", "n", "alpha"))
            or not np.array_equal(left["kappas"], right["kappas"])):
        raise ValueError("DGP and bound must have matching version, kappa, k, n, and alpha")
    if bound["betas"][0] < dgp["betas"][0] or bound["betas"][-1] > dgp["betas"][-1]:
        raise ValueError("DGP beta range must cover the bound beta range")


def _atomic_write(path, writer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent, delete=False) as handle:
            temporary = handle.name
            writer(handle)
        os.replace(temporary, path)
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)


def write_comparison(dgp_path, bound_path=None, output=None):
    dgp = load_dgp(dgp_path)
    bound = None if bound_path is None else load_bound(bound_path)
    if bound is not None:
        _same_experiment(dgp, bound)
    settings = dgp["settings"]
    version, mw = settings["version_label"], len(settings["kappas"])
    output = Path(output or Path(version) / f"power_comparison_{version}.png")
    if output.suffix.lower() != ".png":
        raise ValueError("--output must be a PNG path")
    csv_path = output.with_suffix(".csv")
    figure = Figure(figsize=(9, 5.5), tight_layout=True)
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer)
    writer.writerow(("series", "beta", "power", "mc_se", "run_signature"))
    for key, label, style, color in (
            ("power_chi2", r"$\chi^2$", "--", "tab:blue"),
            ("power_c1", r"$c_1$", "-", "tab:orange"),
            ("power_cp1", rf"$c_{mw}$", "-.", "tab:red")):
        curve = dgp[key]
        axis.plot(dgp["betas"], curve, label=label, linestyle=style, color=color)
        se = np.sqrt(curve * (1 - curve) / settings["num_simulations"])
        writer.writerows((key, beta, value, error, dgp["run_signature"])
                         for beta, value, error in zip(dgp["betas"], curve, se))
    if bound is not None:
        axis.plot(bound["betas"], bound["bounds"], "o-", color="tab:green",
                  markersize=4, label=rf"GKM power bound ($m_W={mw}$)")
        writer.writerows(("gkm_bound", beta, value, error, bound["run_signature"])
                         for beta, value, error in zip(
                             bound["betas"], bound["bounds"], bound["bounds_se"]))
    axis.axhline(settings["alpha"], color="gray", linestyle=":",
                 label=rf"$\alpha={settings['alpha']:g}$")
    axis.set(xlabel=r"True $\beta$", ylabel="Rejection probability",
             title=rf"$m_W={mw}$, $\kappa={settings['kappas']}$, "
                   rf"$k={settings['k']}$, $n={settings['n']}$", ylim=(0, 1))
    axis.legend(loc="best")
    axis.grid(alpha=0.25)
    _atomic_write(output, lambda handle: figure.savefig(handle, format="png", dpi=160))
    _atomic_write(csv_path, lambda handle: handle.write(buffer.getvalue().encode("utf-8")))
    return output, csv_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", default="10015", choices=list(comparison.VERSION_LABELS))
    parser.add_argument("--dgp-cache", help="default: VERSION/dgp/dgp_curves_VERSION.npz")
    parser.add_argument("--bound", help="completed power_bound_cluster.py merged NPZ")
    parser.add_argument("--output", help="PNG path; CSV is saved beside it")
    args = parser.parse_args(argv)
    path = args.dgp_cache or comparison.dgp_cache_path(args.version)
    try:
        if load_dgp(path)["settings"]["version_label"] != args.version:
            raise ValueError("DGP cache does not match --version")
        for result in write_comparison(path, args.bound, args.output):
            print(f"Saved {result}")
    except (OSError, ValueError, TypeError, KeyError, IndexError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()
