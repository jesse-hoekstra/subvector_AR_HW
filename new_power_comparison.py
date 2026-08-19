import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.stats import chi2
from scipy.linalg import inv, sqrtm, eigh
import os
import sys
import time
import hashlib
import json
import platform
from alfd_eigval import VERSION_LABELS, _Tee
from gkm import critical_value


# Only bounds produced by the calibrated adaptive EMW implementation are safe
# to overlay.  In particular, do not silently fall back to the legacy
# M_trunc-specific files: those files have no algorithm/schema provenance and
# may have been produced by the pre-calibration implementation.
ALFD_SCHEMA_VERSION = 2
ALFD_ALGORITHM = 'emw_eigval_adaptive_v2'
ALFD_PRODUCER = 'alfd_eigval.py'
ALFD_CALIBRATION_METHOD = 'independent_mixture_quantile'
ALFD_BOUND_KIND = (
    'simultaneous_mc_confidence_upper_conditional_on_density_accuracy')

DGP_CACHE_SCHEMA_VERSION = 1
DGP_CACHE_ALGORITHM = 'appendix_a3_feasible_power_v1'
DGP_CACHE_PRODUCER = 'new_power_comparison.py'


def adaptive_alfd_path(version_label):
    """Canonical path for a calibrated adaptive-EMW bound artifact."""
    return os.path.join(
        version_label, 'adaptive', f'alfd_eigval_{version_label}.npz')


def dgp_cache_path(version_label):
    """Canonical cache path for finite-sample Appendix A.3 power curves."""
    return os.path.join(
        version_label, 'dgp', f'dgp_curves_{version_label}.npz')


def _npz_scalar(archive, key):
    """Read one required scalar from an ``np.load`` archive."""
    if key not in archive.files:
        raise ValueError(f"missing metadata key {key!r}")
    value = np.asarray(archive[key])
    if value.shape != ():
        raise ValueError(
            f"metadata key {key!r} must be scalar, got shape {value.shape}")
    return value.item()


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _atomic_savez(path, **arrays):
    temporary = path + '.tmp.npz'
    np.savez(temporary, **arrays)
    os.replace(temporary, path)


def _dgp_cache_settings(*, version_label, kappas, k, n, alpha, betas,
                        num_simulations, base_seed, chunk_size):
    repository = os.path.dirname(os.path.abspath(__file__))
    kappas = np.asarray(kappas, dtype=float)
    betas = np.asarray(betas, dtype=float)
    if (kappas.ndim != 1 or kappas.size == 0
            or not np.all(np.isfinite(kappas)) or np.any(kappas < 0.0)):
        raise ValueError('kappas must be a nonempty finite nonnegative vector')
    if (betas.ndim != 1 or betas.size == 0
            or not np.all(np.isfinite(betas)) or np.any(np.diff(betas) <= 0.0)):
        raise ValueError('betas must be a nonempty finite increasing vector')
    if int(k) <= 0 or int(n) <= 0:
        raise ValueError('k and n must be positive')
    if not (0.0 < float(alpha) < 1.0):
        raise ValueError('alpha must lie in (0, 1)')
    if int(num_simulations) <= 0 or int(chunk_size) <= 0:
        raise ValueError('num_simulations and chunk_size must be positive')
    return {
        'schema_version': DGP_CACHE_SCHEMA_VERSION,
        'algorithm': DGP_CACHE_ALGORITHM,
        'producer': DGP_CACHE_PRODUCER,
        'version_label': str(version_label),
        'kappas': kappas.tolist(),
        'k': int(k),
        'n': int(n),
        'alpha': float(alpha),
        'betas': betas.tolist(),
        'num_simulations': int(num_simulations),
        'base_seed': int(base_seed),
        # Chunking changes the SeedSequence partition, hence the realized draws.
        'chunk_size': int(chunk_size),
        'source_sha256': _sha256_file(os.path.abspath(__file__)),
        'gkm_sha256': _sha256_file(os.path.join(repository, 'gkm.py')),
        'python_version': sys.version,
        'numpy_version': np.__version__,
        'scipy_version': scipy.__version__,
        'platform': platform.platform(),
    }


def _settings_json_and_signature(settings):
    encoded = json.dumps(settings, sort_keys=True, separators=(',', ':'))
    signature = hashlib.sha256(encoded.encode()).hexdigest()
    return encoded, signature


def _validate_dgp_curves(betas, power_chi2, power_c1, power_cp1,
                         expected_betas):
    betas = np.asarray(betas, dtype=float)
    curves = tuple(np.asarray(curve, dtype=float)
                   for curve in (power_chi2, power_c1, power_cp1))
    expected_betas = np.asarray(expected_betas, dtype=float)
    if (betas.ndim != 1 or betas.size == 0
            or betas.shape != expected_betas.shape):
        raise ValueError(
            f'DGP beta grid has shape {betas.shape}, expected {expected_betas.shape}')
    if not np.all(np.isfinite(betas)) or np.any(np.diff(betas) <= 0.0):
        raise ValueError('DGP beta grid must be finite and strictly increasing')
    if not np.allclose(betas, expected_betas, rtol=0.0, atol=1e-12):
        raise ValueError('DGP beta grid does not match the requested grid')
    for name, curve in zip(
            ('power_chi2', 'power_c1', 'power_cp1'), curves):
        if curve.shape != betas.shape:
            raise ValueError(
                f'{name} has shape {curve.shape}, expected {betas.shape}')
        if not np.all(np.isfinite(curve)) or np.any((curve < 0.0) | (curve > 1.0)):
            raise ValueError(f'{name} must contain finite probabilities in [0, 1]')
    return (betas.copy(),) + tuple(curve.copy() for curve in curves)


def save_dgp_cache(path, *, version_label, kappas, k, n, alpha, betas,
                   power_chi2, power_c1, power_cp1, num_simulations,
                   base_seed, chunk_size, workers_used):
    """Atomically save a provenance-complete finite-sample curve cache."""
    betas, power_chi2, power_c1, power_cp1 = _validate_dgp_curves(
        betas, power_chi2, power_c1, power_cp1, betas)
    settings = _dgp_cache_settings(
        version_label=version_label, kappas=kappas, k=k, n=n, alpha=alpha,
        betas=betas, num_simulations=num_simulations, base_seed=base_seed,
        chunk_size=chunk_size)
    settings_json, run_signature = _settings_json_and_signature(settings)
    _atomic_savez(
        path,
        schema_version=np.array(DGP_CACHE_SCHEMA_VERSION),
        algorithm=np.array(DGP_CACHE_ALGORITHM),
        producer=np.array(DGP_CACHE_PRODUCER),
        version_label=np.array(str(version_label)),
        source_sha256=np.array(settings['source_sha256']),
        gkm_sha256=np.array(settings['gkm_sha256']),
        settings_json=np.array(settings_json),
        run_signature=np.array(run_signature),
        betas=betas,
        power_chi2=power_chi2,
        power_c1=power_c1,
        power_cp1=power_cp1,
        kappas=np.asarray(kappas, dtype=float),
        k=np.array(int(k)), n=np.array(int(n)), alpha=np.array(float(alpha)),
        num_simulations=np.array(int(num_simulations)),
        base_seed=np.array(int(base_seed)), chunk_size=np.array(int(chunk_size)),
        workers_used=np.array(int(workers_used)),
    )


def load_compatible_dgp_cache(path, *, version_label, kappas, k, n, alpha,
                              betas, num_simulations, base_seed, chunk_size):
    """Load a DGP cache only when its full settings/provenance contract matches."""
    expected_settings = _dgp_cache_settings(
        version_label=version_label, kappas=kappas, k=k, n=n, alpha=alpha,
        betas=betas, num_simulations=num_simulations, base_seed=base_seed,
        chunk_size=chunk_size)
    expected_json, expected_signature = _settings_json_and_signature(
        expected_settings)
    with np.load(path, allow_pickle=False) as archive:
        scalar_expectations = {
            'schema_version': DGP_CACHE_SCHEMA_VERSION,
            'algorithm': DGP_CACHE_ALGORITHM,
            'producer': DGP_CACHE_PRODUCER,
            'version_label': str(version_label),
            'source_sha256': expected_settings['source_sha256'],
            'gkm_sha256': expected_settings['gkm_sha256'],
            'settings_json': expected_json,
            'run_signature': expected_signature,
            'k': int(k), 'n': int(n), 'alpha': float(alpha),
            'num_simulations': int(num_simulations),
            'base_seed': int(base_seed), 'chunk_size': int(chunk_size),
        }
        mismatches = []
        for name, expected in scalar_expectations.items():
            saved = _npz_scalar(archive, name)
            if isinstance(expected, float):
                try:
                    matches = bool(np.isclose(
                        saved, expected, rtol=0.0, atol=1e-12))
                except TypeError:
                    matches = False
            else:
                matches = saved == expected
            if not matches:
                mismatches.append(f'{name}={saved!r} does not match {expected!r}')

        if 'kappas' not in archive.files:
            raise ValueError("missing metadata key 'kappas'")
        saved_kappas = np.asarray(archive['kappas'], dtype=float)
        expected_kappas = np.asarray(kappas, dtype=float)
        if (saved_kappas.shape != expected_kappas.shape or
                not np.allclose(saved_kappas, expected_kappas,
                                rtol=0.0, atol=1e-12)):
            mismatches.append('kappas do not match the requested configuration')
        if mismatches:
            raise ValueError('incompatible DGP cache: ' + '; '.join(mismatches))

        required = ('betas', 'power_chi2', 'power_c1', 'power_cp1')
        missing = [name for name in required if name not in archive.files]
        if missing:
            raise ValueError(
                'missing required DGP array(s): ' + ', '.join(missing))
        return _validate_dgp_curves(
            archive['betas'], archive['power_chi2'], archive['power_c1'],
            archive['power_cp1'], betas)


def load_compatible_alfd_bound(path, *, version_label, kappas, k, n, alpha,
                               return_metadata=False):
    """Load and validate one calibrated adaptive-EMW bound artifact.

    Legacy or otherwise incompatible files raise ``ValueError`` rather than
    being plotted as an upper bound.  Returned arrays are copies, so callers
    cannot mutate the on-disk archive accidentally.
    """
    with np.load(path, allow_pickle=False) as archive:
        schema_version = _npz_scalar(archive, 'schema_version')
        algorithm = _npz_scalar(archive, 'algorithm')
        producer = _npz_scalar(archive, 'producer')
        saved_version = _npz_scalar(archive, 'version_label')
        calibration_method = _npz_scalar(archive, 'calibration_method')
        bound_kind = _npz_scalar(archive, 'bound_kind')
        saved_k = _npz_scalar(archive, 'k')
        saved_n = _npz_scalar(archive, 'n')
        saved_alpha = _npz_scalar(archive, 'alpha')
        saved_source_hash = _npz_scalar(archive, 'source_sha256')
        saved_core_hash = _npz_scalar(archive, 'mhg_core_sha256')
        saved_library_hash = _npz_scalar(archive, 'mhg_library_sha256')
        saved_build_source_hash = _npz_scalar(
            archive, 'mhg_build_source_sha256')
        confidence_scope = _npz_scalar(archive, 'confidence_scope')
        density_accuracy_scope = _npz_scalar(
            archive, 'density_accuracy_scope')
        curve_confidence = _npz_scalar(archive, 'curve_confidence')
        settings_json = _npz_scalar(archive, 'settings_json')
        saved_run_signature = _npz_scalar(archive, 'run_signature')
        try:
            saved_settings = json.loads(settings_json)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f'invalid settings_json: {exc}') from exc
        if not isinstance(saved_settings, dict):
            raise ValueError('settings_json must encode an object')
        canonical_settings = json.dumps(
            saved_settings, sort_keys=True, separators=(',', ':'))
        calculated_run_signature = hashlib.sha256(
            canonical_settings.encode()).hexdigest()
        profile = saved_settings.get('profile')

        repository = os.path.dirname(os.path.abspath(__file__))
        library_name = 'libmhg.dylib' if sys.platform == 'darwin' else 'libmhg.so'
        current_hashes = {
            'source_sha256': _sha256_file(os.path.join(repository, 'alfd_eigval.py')),
            'mhg_core_sha256': _sha256_file(
                os.path.join(repository, 'koev', 'mhg15', 'mhg_core.c')),
            'mhg_library_sha256': _sha256_file(
                os.path.join(repository, 'koev', 'mhg15', library_name)),
        }
        current_hashes['mhg_build_source_sha256'] = current_hashes[
            'mhg_core_sha256']

        mismatches = []
        if saved_run_signature != calculated_run_signature:
            mismatches.append(
                'run_signature does not authenticate the canonical settings_json')
        if schema_version != ALFD_SCHEMA_VERSION:
            mismatches.append(
                f"schema_version={schema_version!r} (expected {ALFD_SCHEMA_VERSION})")
        if algorithm != ALFD_ALGORITHM:
            mismatches.append(
                f"algorithm={algorithm!r} (expected {ALFD_ALGORITHM!r})")
        if producer != ALFD_PRODUCER:
            mismatches.append(
                f"producer={producer!r} (expected {ALFD_PRODUCER!r})")
        if saved_version != version_label:
            mismatches.append(
                f"version_label={saved_version!r} (expected {version_label!r})")
        if calibration_method != ALFD_CALIBRATION_METHOD:
            mismatches.append(
                f"calibration_method={calibration_method!r} "
                f"(expected {ALFD_CALIBRATION_METHOD!r})")
        if bound_kind != ALFD_BOUND_KIND:
            mismatches.append(
                f"bound_kind={bound_kind!r} (expected {ALFD_BOUND_KIND!r})")
        if confidence_scope != 'saved_beta_grid_only':
            mismatches.append(
                f"confidence_scope={confidence_scope!r} "
                "(expected 'saved_beta_grid_only')")
        if density_accuracy_scope != 'adaptive_empirical_tail_criterion':
            mismatches.append(
                f"density_accuracy_scope={density_accuracy_scope!r} "
                "(expected 'adaptive_empirical_tail_criterion')")
        if saved_k != k:
            mismatches.append(f"k={saved_k!r} (expected {k})")
        if saved_n != n:
            mismatches.append(f"n={saved_n!r} (expected {n})")
        if not np.isclose(saved_alpha, alpha, rtol=0.0, atol=1e-12):
            mismatches.append(f"alpha={saved_alpha!r} (expected {alpha})")
        try:
            confidence_ok = bool(
                np.isfinite(curve_confidence)
                and 0.0 < float(curve_confidence) < 1.0)
        except (TypeError, ValueError):
            confidence_ok = False
        if not confidence_ok:
            mismatches.append(
                f'curve_confidence={curve_confidence!r} is not in (0, 1)')
        settings_confidence = saved_settings.get('curve_confidence')
        try:
            settings_confidence_matches = bool(np.isclose(
                settings_confidence, curve_confidence, rtol=0.0, atol=1e-12))
        except TypeError:
            settings_confidence_matches = False
        if not settings_confidence_matches:
            mismatches.append(
                'settings_json curve_confidence disagrees with artifact metadata')
        if profile not in ('production', 'reference'):
            mismatches.append(
                f'profile={profile!r} cannot produce a plottable p=4 artifact')

        settings_scalar_expectations = {
            'schema_version': schema_version,
            'algorithm': algorithm,
            'producer': producer,
            'calibration_method': calibration_method,
            'version_label': saved_version,
            'k': saved_k,
            'n': saved_n,
            'alpha': saved_alpha,
            'curve_confidence': curve_confidence,
            'source_sha256': saved_source_hash,
            'mhg_core_sha256': saved_core_hash,
            'mhg_library_sha256': saved_library_hash,
            'mhg_build_source_sha256': saved_build_source_hash,
        }
        for name, expected in settings_scalar_expectations.items():
            if name not in saved_settings:
                mismatches.append(f'settings_json is missing {name!r}')
                continue
            saved_setting = saved_settings[name]
            if isinstance(expected, float):
                try:
                    matches = bool(np.isclose(
                        saved_setting, expected, rtol=0.0, atol=1e-12))
                except TypeError:
                    matches = False
            else:
                matches = saved_setting == expected
            if not matches:
                mismatches.append(
                    f'settings_json {name} disagrees with artifact metadata')
        for name, saved in (
                ('source_sha256', saved_source_hash),
                ('mhg_core_sha256', saved_core_hash),
                ('mhg_library_sha256', saved_library_hash),
                ('mhg_build_source_sha256', saved_build_source_hash)):
            if saved != current_hashes[name]:
                mismatches.append(f"{name} does not match the current implementation")

        if 'kappas' not in archive.files:
            raise ValueError("missing metadata key 'kappas'")
        saved_kappas = np.asarray(archive['kappas'], dtype=float)
        expected_kappas = np.asarray(kappas, dtype=float)
        if (saved_kappas.shape != expected_kappas.shape or
                not np.allclose(saved_kappas, expected_kappas,
                                rtol=0.0, atol=1e-12)):
            mismatches.append(
                f"kappas={saved_kappas.tolist()} "
                f"(expected {expected_kappas.tolist()})")
        try:
            settings_kappas = np.asarray(saved_settings['kappas'], dtype=float)
        except (KeyError, TypeError, ValueError):
            mismatches.append("settings_json has invalid or missing 'kappas'")
        else:
            if (settings_kappas.shape != saved_kappas.shape or
                    not np.allclose(settings_kappas, saved_kappas,
                                    rtol=0.0, atol=1e-12)):
                mismatches.append(
                    'settings_json kappas disagree with artifact metadata')

        if mismatches:
            raise ValueError("incompatible metadata: " + "; ".join(mismatches))

        required_arrays = ('betas', 'bounds', 'bounds_se')
        missing = [name for name in required_arrays if name not in archive.files]
        if missing:
            raise ValueError(
                "missing required result array(s): " + ", ".join(missing))

        betas = np.asarray(archive['betas'], dtype=float).copy()
        bounds = np.asarray(archive['bounds'], dtype=float).copy()
        bounds_se = np.asarray(archive['bounds_se'], dtype=float).copy()

        beta_count = saved_settings.get('beta_count')
        if (not isinstance(beta_count, (int, np.integer))
                or isinstance(beta_count, (bool, np.bool_))
                or int(beta_count) != betas.size):
            raise ValueError(
                'settings_json beta_count disagrees with saved beta grid')

    if betas.ndim != 1 or betas.size == 0:
        raise ValueError(f"betas must be a nonempty 1-D array, got {betas.shape}")
    if bounds.shape != betas.shape or bounds_se.shape != betas.shape:
        raise ValueError(
            "betas, bounds, and bounds_se must have identical 1-D shapes; "
            f"got {betas.shape}, {bounds.shape}, {bounds_se.shape}")
    if not (np.all(np.isfinite(betas)) and np.all(np.isfinite(bounds)) and
            np.all(np.isfinite(bounds_se))):
        raise ValueError("betas, bounds, and bounds_se must all be finite")
    if np.any(np.diff(betas) <= 0):
        raise ValueError("betas must be strictly increasing")
    if np.any((bounds < 0.0) | (bounds > 1.0)):
        raise ValueError("bounds must lie in [0, 1]")
    if np.any(bounds < alpha - 1e-12):
        raise ValueError("an EMW upper bound cannot lie below alpha")
    if np.any(bounds_se < 0.0):
        raise ValueError("bounds_se must be nonnegative")

    if return_metadata:
        return betas, bounds, bounds_se, {
            'curve_confidence': float(curve_confidence),
            'profile': profile,
        }
    return betas, bounds, bounds_se


# ---------------------------------------------------------
# DGP SIMULATION (APPENDIX A.3)
# ---------------------------------------------------------

# Fixed error covariance of (eps, V_X, V_W1, V_W2, V_W3) -- module level so
# both the serial and the parallel worker use the exact same matrix.
_SIGMA = np.array([
    [1.0, 0.1, 0.3, 0.2, 0.8],
    [0.1, 1.0, 0.3, 0.2, 0.1],
    [0.3, 0.3, 1.0, 0.3, 0.2],
    [0.2, 0.2, 0.3, 1.0, 0.3],
    [0.8, 0.1, 0.2, 0.3, 1.0],
])


def _dgp_build_constants(kappas, n, k):
    """Per-(kappas, n, k) DGP constants: Pi_W, pi_x, gamma. Cheap to recompute."""
    Sigma = _SIGMA
    Sigma_eps_eps = Sigma[0, 0]
    Sigma_eps_Vw = Sigma[0, 2:]
    Sigma_Vw_Vw = Sigma[2:, 2:]
    Sigma_Vw_Vw_eps = Sigma_Vw_Vw - np.outer(Sigma_eps_Vw, Sigma_eps_Vw) / Sigma_eps_eps
    sqrt_Sigma = sqrtm(Sigma_Vw_Vw_eps)
    A = np.array([
        [1/np.sqrt(n*3), 0, 0],
        [1/np.sqrt(n*3), 0, 0],
        [1/np.sqrt(n*3), 0, 0],
        [0, 1/np.sqrt(n*2), 0],
        [0, 1/np.sqrt(n*2), 0],
        [0, 0, 1/np.sqrt(n*2)],
        [0, 0, 1/np.sqrt(n*2)],
    ])
    Pi_W = A @ sqrtm(np.diag(kappas)) @ sqrt_Sigma
    pi_x = (4.0 / np.sqrt(k * n)) * np.array([1, 1, 1, -1, 1, 1, 1])
    gamma_params = np.array([-1.0, 1.0, 1.0])
    return Pi_W, pi_x, gamma_params


def _dgp_chunk_worker(args):
    """
    Run `n_sims` independent DGP draws for one beta and return rejection counts.

    Each (beta, chunk) task is independent; counts are aggregated by the caller.
    A dedicated Generator (seeded from a SeedSequence) keeps the streams
    independent and reproducible across workers.

    args : (beta_idx, beta, kappas, n, k, n_sims, hat_k1_grid, cv_grid,
            cv_chi2, seed_seq)
    returns : (beta_idx, rej_chi2, rej_c1, rej_cp1)
    """
    (beta_idx, beta, kappas, n, k, n_sims,
     hat_k1_grid, cv_grid, cv_chi2, seed_seq) = args

    Pi_W, pi_x, gamma_params = _dgp_build_constants(kappas, n, k)
    Sigma = _SIGMA
    rng = np.random.default_rng(seed_seq)
    I_n = np.eye(n)
    mean5 = np.zeros(5)

    rej_chi2 = rej_c1 = rej_cp1 = 0
    for _ in range(n_sims):
        Z = rng.standard_normal((n, k))
        errors = rng.multivariate_normal(mean5, Sigma, n)
        eps = errors[:, 0]
        V_X = errors[:, 1]
        V_W = errors[:, 2:]

        X = Z @ pi_x + V_X
        W = Z @ Pi_W + V_W
        y = X * beta + W @ gamma_params + eps

        y_0 = y                       # H0: beta=0 => y_0 = y
        Z_dm = Z - Z.mean(axis=0)
        y_0_dm = y_0 - y_0.mean()
        W_dm = W - W.mean(axis=0)

        P_Z = Z_dm @ inv(Z_dm.T @ Z_dm) @ Z_dm.T
        M_Z = I_n - P_Z
        y_0_W = np.column_stack([y_0_dm, W_dm])

        Omega_hat = (y_0_W.T @ M_Z @ y_0_W) / (n - k - 1)
        temp = y_0_W.T @ P_Z @ y_0_W

        evals = eigh(temp, Omega_hat, eigvals_only=True)
        evals = np.sort(np.real(evals))[::-1]
        test_stat = evals[-1]

        if test_stat > cv_chi2:
            rej_chi2 += 1
        if test_stat > np.interp(evals[0], hat_k1_grid, cv_grid):
            rej_c1 += 1
        if test_stat > np.interp(evals[-2], hat_k1_grid, cv_grid):
            rej_cp1 += 1

    return beta_idx, rej_chi2, rej_c1, rej_cp1


def simulate_power_dgp(betas, kappas, n=250, k=7, num_simulations=100000,
                      cv_grid=None, hat_k1_grid=None,
                      n_workers=None, chunk_size=5000, base_seed=20240101):
    """
    Simulate the Appendix A.3 DGP power curve, parallelized over (beta, chunk)
    tasks via multiprocessing.

    n_workers   : number of worker processes (default: os.cpu_count()).
                  Set to 1 to force the serial path.
    chunk_size  : simulations per task. Each beta is split into
                  ceil(num_simulations/chunk_size) chunks; all chunks across
                  all betas are dispatched to the pool for good load balance.
    base_seed   : root of the SeedSequence; distinct per chunk for independence.
    """
    m_W = len(kappas)
    alpha = 0.05
    betas = np.asarray(betas, dtype=float)
    n_betas = len(betas)
    if betas.ndim != 1 or n_betas == 0 or not np.all(np.isfinite(betas)):
        raise ValueError('betas must be a nonempty finite vector')
    if int(num_simulations) <= 0 or int(chunk_size) <= 0:
        raise ValueError('num_simulations and chunk_size must be positive')

    k_cv = k - m_W + 1
    if hat_k1_grid is None:
        hat_k1_grid = np.linspace(0.01, 500, 200)
    if cv_grid is None:
        print("Pre-computing critical value grid...")
        cv_grid = np.array([critical_value(h, alpha, k_cv) for h in hat_k1_grid])
        print("Done.")
    cv_chi2 = chi2.ppf(1 - alpha, df=k - m_W)

    if n_workers is None:
        n_workers = os.cpu_count() or 1
    if int(n_workers) <= 0:
        raise ValueError('n_workers must be positive')

    # Build (beta, chunk) tasks. Each beta's num_simulations is split into
    # chunks of <= chunk_size; a unique child SeedSequence per task.
    tasks = []
    chunk_meta = []   # (beta_idx, n_sims) parallel to tasks
    for bi, beta in enumerate(betas):
        remaining = num_simulations
        while remaining > 0:
            nb = min(chunk_size, remaining)
            chunk_meta.append((bi, nb))
            remaining -= nb
    n_tasks = len(chunk_meta)
    seed_seqs = np.random.SeedSequence(base_seed).spawn(n_tasks)
    for t_idx, (bi, nb) in enumerate(chunk_meta):
        tasks.append((bi, float(betas[bi]), kappas, n, k, nb,
                      hat_k1_grid, cv_grid, cv_chi2, seed_seqs[t_idx]))

    print(f"DGP simulation: {n_betas} betas x {num_simulations:,} sims "
          f"= {n_tasks} tasks (chunk_size={chunk_size}), {n_workers} workers")

    counts = np.zeros((n_betas, 3), dtype=np.int64)   # [chi2, c1, cp1]

    if n_workers <= 1:
        for t in tasks:
            bi, rc, r1, rp1 = _dgp_chunk_worker(t)
            counts[bi] += (rc, r1, rp1)
    else:
        from multiprocessing import Pool
        done = 0
        t_start = time.time()
        with Pool(processes=n_workers) as pool:
            for bi, rc, r1, rp1 in pool.imap_unordered(_dgp_chunk_worker, tasks):
                counts[bi] += (rc, r1, rp1)
                done += 1
                if done % max(1, n_tasks // 20) == 0 or done == n_tasks:
                    el = time.time() - t_start
                    eta = el / done * (n_tasks - done)
                    print(f"  {done}/{n_tasks} chunks done "
                          f"({el:.0f}s elapsed, ~{eta:.0f}s left)", flush=True)

    power_chi2 = (counts[:, 0] / num_simulations).tolist()
    power_c1 = (counts[:, 1] / num_simulations).tolist()
    power_cp1 = (counts[:, 2] / num_simulations).tolist()

    for bi, beta in enumerate(betas):
        print(f"Beta: {beta:5.2f} | chi2: {power_chi2[bi]:.4f} | "
              f"c1: {power_c1[bi]:.4f} | cp1: {power_cp1[bi]:.4f}")

    return power_chi2, power_c1, power_cp1

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="DGP power curves + ALFD bound overlay for one config.")
    parser.add_argument('--version', required=True, choices=list(VERSION_LABELS),
                        help="config version label (" + ", ".join(VERSION_LABELS) + ")")
    parser.add_argument('--num-simulations', type=int, default=100000,
                        help='finite-sample draws per beta')
    parser.add_argument('--workers', type=int, default=None,
                        help='worker processes (default: at most 8)')
    parser.add_argument('--seed', type=int, default=20240101,
                        help='root SeedSequence entropy for the DGP sweep')
    parser.add_argument('--chunk-size', type=int, default=5000,
                        help='simulations per deterministic RNG chunk')
    parser.add_argument('--force', action='store_true',
                        help='replace an existing incompatible or compatible DGP cache')
    parser.add_argument('--preflight-only', action='store_true',
                        help='report cache/simulation scale and exit without plotting')
    parser.add_argument('--acknowledge-expensive', action='store_true',
                        help='required before generating a missing DGP cache')
    args = parser.parse_args()

    if args.num_simulations <= 0:
        parser.error('--num-simulations must be positive')
    if args.chunk_size <= 0:
        parser.error('--chunk-size must be positive')
    n_workers = (min(8, os.cpu_count() or 1)
                 if args.workers is None else args.workers)
    if n_workers <= 0:
        parser.error('--workers must be positive')

    # ---- resolve config from version label ----
    k = 7
    n = 250
    alpha = 0.05
    key = VERSION_LABELS[args.version]
    kappas = np.array(key, dtype=float)
    m_W = len(kappas)

    # DGP caches have no matrix-hypergeometric truncation dependency.  Keep them
    # out of the legacy M_trunc directories so numerical-bound settings cannot
    # accidentally select or invalidate a finite-sample curve.
    out_dir = os.path.join(args.version, 'dgp')
    os.makedirs(out_dir, exist_ok=True)

    # Tee all console output into <out_dir>/dgp_curves_run.log
    _log_fh = open(os.path.join(out_dir, "dgp_curves_run.log"), "a", buffering=1)
    sys.stdout = _Tee(sys.stdout, _log_fh)
    sys.stderr = _Tee(sys.stderr, _log_fh)
    print(f"Version {args.version}: kappas={kappas.tolist()}, n={n}, k={k}")
    print(f"  output dir: {out_dir}/")

    betas = np.linspace(-2, 2, 81)

    # Schema-versioned cache for the three feasible-test curves.
    kappa_tag = "_".join(str(int(round(x))) for x in kappas)
    cache_file = dgp_cache_path(args.version)
    use_cache = os.path.isfile(cache_file) and not args.force
    if use_cache:
        print(f"\nValidating cached DGP curves at {cache_file}...")
        try:
            betas, power_chi2, power_c1, power_cp1 = \
                load_compatible_dgp_cache(
                    cache_file, version_label=args.version, kappas=kappas,
                    k=k, n=n, alpha=alpha, betas=betas,
                    num_simulations=args.num_simulations,
                    base_seed=args.seed, chunk_size=args.chunk_size)
        except (OSError, TypeError, ValueError) as exc:
            raise RuntimeError(
                f"Refusing incompatible DGP cache {cache_file}: {exc}. "
                "Re-run with --force to replace it explicitly.") from exc
        print("Loaded compatible DGP cache.")
        if args.preflight_only:
            print("Preflight only; the compatible cache was validated and no plot was written.")
            return
    else:
        if os.path.isfile(cache_file):
            action = ("would replace after preflight"
                      if args.preflight_only else "replacing")
            print(f"\n--force specified; {action} DGP cache {cache_file}...")
        else:
            print(f"\nNo compatible cache at {cache_file}; legacy M_trunc "
                  "DGP caches are intentionally ignored.")
        total_draws = len(betas) * args.num_simulations
        print(f"Requested DGP simulation scale: {len(betas)} betas × "
              f"{args.num_simulations:,} = {total_draws:,} finite-sample draws.")
        if args.preflight_only:
            print("Preflight only; no DGP simulation or result artifact was written.")
            return
        if not args.acknowledge_expensive:
            parser.error(
                "generating a DGP cache requires --acknowledge-expensive; "
                "inspect the simulation count above first")
        print("Running DGP beta sweep...")
        print("Pre-computing critical value grid...")
        hat_k1_grid = np.linspace(0.01, 500, 200)
        cv_grid = np.array([critical_value(h, alpha, k - m_W + 1) for h in hat_k1_grid])
        print("Done.")

        power_chi2, power_c1, power_cp1 = simulate_power_dgp(
            betas=betas,
            kappas=kappas,
            n=n,
            k=k,
            hat_k1_grid=hat_k1_grid,
            cv_grid=cv_grid,
            num_simulations=args.num_simulations,
            n_workers=n_workers,
            chunk_size=args.chunk_size,
            base_seed=args.seed,
        )
        save_dgp_cache(
            cache_file, version_label=args.version, kappas=kappas,
            k=k, n=n, alpha=alpha, betas=betas,
            power_chi2=power_chi2, power_c1=power_c1,
            power_cp1=power_cp1,
            num_simulations=args.num_simulations, base_seed=args.seed,
            chunk_size=args.chunk_size, workers_used=n_workers)
        print(f"Saved {cache_file}")

    # Optional ALFD eigenvalue-density overlay.  Load only the calibrated,
    # schema-versioned adaptive artifact; legacy M_trunc outputs are refused.
    alfd_path = adaptive_alfd_path(args.version)
    has_alfd = False
    if not os.path.isfile(alfd_path):
        print(f"WARNING: no compatible adaptive ALFD artifact at {alfd_path}; "
              "legacy M_trunc bound files are intentionally ignored. "
              "Plotting feasible curves only.")
    else:
        try:
            betas_alfd, bounds_alfd, bounds_se, alfd_metadata = \
                load_compatible_alfd_bound(
                    alfd_path,
                    version_label=args.version,
                    kappas=kappas,
                    k=k,
                    n=n,
                    alpha=alpha,
                    return_metadata=True,
                )
        except (OSError, TypeError, ValueError) as exc:
            print(f"WARNING: refusing ALFD artifact {alfd_path}: {exc}. "
                  "Plotting feasible curves only.")
        else:
            has_alfd = True
            print(f"Loaded {alfd_metadata['curve_confidence']:.1%} simultaneous-MC "
                  "ALFD upper bound conditional on adaptive density accuracy "
                  f"({alfd_metadata['profile']} profile) "
                  f"from {alfd_path}")

    print("\nPlotting results...")
    plt.figure(figsize=(9, 5.5))
    plt.plot(betas, power_chi2, linestyle='--', color='k',  label=r'$\phi_{\chi^2}$')
    plt.plot(betas, power_c1,   linestyle='-',  color='b',  label=r'$\phi_{c_1}$')
    plt.plot(betas, power_cp1,  linestyle='-.', color='r',  label=r'$\phi_{c_{p-1}}$')
    if has_alfd:
        if np.any(bounds_se > 0.0):
            plt.fill_between(betas_alfd, bounds_alfd - bounds_se,
                             bounds_alfd + bounds_se,
                             color='g', alpha=0.2,
                             label=r'EMW numerical uncertainty')
        plt.plot(betas_alfd, bounds_alfd, 'go-', linewidth=2, markersize=5,
                 label=(f"EMW {alfd_metadata['curve_confidence']:.0%} "
                        "simultaneous-MC upper, conditional on density accuracy "
                        f"({alfd_metadata['profile']})"))
    plt.axhline(y=alpha, color='gray', linestyle=':', label=rf'Size ($\alpha$={alpha})')
    plt.title(rf'Power Curve for $\kappa$ = {kappas.tolist()}')
    plt.xlabel(r'True $\beta$')
    plt.ylabel('Rejection Probability')
    y_top = max(
        0.15,
        float(np.max(power_chi2)) + 0.05,
        float(np.max(power_c1)) + 0.05,
        float(np.max(power_cp1)) + 0.05,
    )
    if has_alfd:
        y_top = max(y_top, (bounds_alfd + bounds_se).max() + 0.05)
    plt.ylim(0, y_top)
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    out_png = os.path.join(out_dir, f"power_curve_kappas_{kappa_tag}.png")
    plt.savefig(out_png, dpi=140)
    print(f"Saved {out_png}")
    plt.show()

if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    main()
