import json
import os
import tempfile
import unittest
from unittest import mock
import subprocess
import sys
import csv
from pathlib import Path

import numpy as np
from scipy.linalg import eigh, inv, sqrtm

import new_power_comparison as comparison


class DgpOnlyComparisonTests(unittest.TestCase):
    def setUp(self):
        self.version = '352515'
        self.kappas = np.array([35.0, 25.0, 15.0])
        self.betas = np.array([-0.2, 0.0, 0.2])
        self.curves = (
            np.array([0.04, 0.05, 0.06]),
            np.array([0.05, 0.05, 0.07]),
            np.array([0.06, 0.05, 0.08]),
        )

    def _save(self, path):
        comparison.save_dgp_cache(
            path, version_label=self.version, kappas=self.kappas,
            k=7, n=250, alpha=0.05, betas=self.betas,
            power_chi2=self.curves[0], power_c1=self.curves[1],
            power_cp1=self.curves[2], num_simulations=1000,
            base_seed=123, chunk_size=100, workers_used=2)

    def _load(self, path):
        return comparison.load_compatible_dgp_cache(
            path, version_label=self.version, kappas=self.kappas,
            k=7, n=250, alpha=0.05, betas=self.betas,
            num_simulations=1000, base_seed=123, chunk_size=100)

    @staticmethod
    def _resign_with_source_hash(path, source_hash):
        with np.load(path, allow_pickle=False) as archive:
            payload = {name: np.asarray(archive[name]).copy()
                       for name in archive.files}
        settings = json.loads(str(payload['settings_json']))
        settings['source_sha256'] = source_hash
        settings_json, signature = comparison._settings_json_and_signature(
            settings)
        payload['source_sha256'] = np.array(source_hash)
        payload['settings_json'] = np.array(settings_json)
        payload['run_signature'] = np.array(signature)
        np.savez(path, **payload)

    def test_audited_pre_cleanup_schema_one_cache_remains_compatible(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'dgp.npz')
            self._save(path)
            legacy_hash = next(iter(
                comparison._TRUSTED_PRE_DGP_ONLY_SOURCE_SHA256))
            self._resign_with_source_hash(path, legacy_hash)

            loaded = self._load(path)
            for actual, expected in zip(
                    loaded, (self.betas,) + self.curves):
                np.testing.assert_array_equal(actual, expected)

    def test_arbitrary_even_self_consistently_signed_source_hash_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'dgp.npz')
            self._save(path)
            self._resign_with_source_hash(path, '0' * 64)

            with self.assertRaisesRegex(
                    ValueError, 'audited DGP-equivalent predecessor'):
                self._load(path)

    def test_legacy_alfd_plotter_api_is_absent(self):
        for name in (
                'adaptive_alfd_path', 'load_compatible_alfd_bound',
                'ALFD_SCHEMA_VERSION', 'ALFD_ALGORITHM', 'ALFD_BOUND_KIND'):
            self.assertFalse(hasattr(comparison, name), name)

    def test_predecessor_hash_cannot_authenticate_new_mw2_design(self):
        self.version = '10015'
        self.kappas = np.array([100.0, 15.0])
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'dgp.npz')
            self._save(path)
            self._resign_with_source_hash(
                path, next(iter(comparison._TRUSTED_PRE_DGP_ONLY_SOURCE_SHA256)))
            with self.assertRaisesRegex(ValueError, 'audited DGP-equivalent predecessor'):
                self._load(path)


class TwoNuisanceDgpSimulationTests(unittest.TestCase):
    @staticmethod
    def _explicit_reference(args):
        """Independent n-by-n projection of exactly the configured DGP draws."""
        (bi, beta, kappas, n, k, hat_grid, cv_grid, cv_chi2, seed, count) = args
        m = len(kappas)
        sigma = np.array([
            [1, .1, .3, .2, .8], [.1, 1, .3, .2, .1],
            [.3, .3, 1, .3, .2], [.2, .2, .3, 1, .3],
            [.8, .1, .2, .3, 1],
        ])[:m + 2, :m + 2]
        A = np.zeros((7, 3))
        A[:3, 0] = 1 / np.sqrt(3 * n)
        A[3:5, 1] = 1 / np.sqrt(2 * n)
        A[5:, 2] = 1 / np.sqrt(2 * n)
        cond_sigma = sigma[2:, 2:] - np.outer(sigma[0, 2:], sigma[0, 2:])
        Pi_W = A[:, :m] @ sqrtm(np.diag(kappas)) @ sqrtm(cond_sigma)
        pi_x = 4 / np.sqrt(k * n) * np.array([1, 1, 1, -1, 1, 1, 1])
        gamma = np.array([-1., 1., 1.])[:m]
        rng = np.random.default_rng(seed)
        counts = np.zeros(3, dtype=int)
        matrices = []
        for _ in range(count):
            Z = rng.standard_normal((n, k))
            errors = rng.multivariate_normal(np.zeros(m + 2), sigma, n)
            W = Z @ Pi_W + errors[:, 2:]
            y = beta * (Z @ pi_x + errors[:, 1]) + W @ gamma + errors[:, 0]
            Z = Z - Z.mean(axis=0)
            Y = np.column_stack([y - y.mean(), W - W.mean(axis=0)])
            P = Z @ inv(Z.T @ Z) @ Z.T
            omega = Y.T @ (np.eye(n) - P) @ Y / (n - k - 1)
            fitted = Y.T @ P @ Y
            ev = eigh(fitted, omega, eigvals_only=True)[::-1]
            matrices.append((fitted, omega, ev))
            counts += (ev[-1] > cv_chi2,
                       ev[-1] > np.interp(ev[0], hat_grid, cv_grid),
                       ev[-1] > np.interp(ev[-2], hat_grid, cv_grid))
        return (bi, *counts), matrices

    def test_mw2_design_matches_frozen_bound_design(self):
        import alfd_eigval as alfd
        sigma, A, pi_x, gamma = alfd._dgp_constants(7, 250, m_W=2)
        cond_sigma = sigma[2:, 2:] - np.outer(sigma[0, 2:], sigma[0, 2:])
        actual_Pi_W, actual_pi_x, actual_gamma = comparison._dgp_build_constants(
            [100, 15], 250, 7)
        np.testing.assert_array_equal(comparison._SIGMA[:4, :4], sigma)
        np.testing.assert_allclose(
            actual_Pi_W, A @ sqrtm(np.diag([100, 15])) @ sqrtm(cond_sigma),
            rtol=0, atol=0)
        np.testing.assert_array_equal(actual_pi_x, pi_x)
        np.testing.assert_array_equal(actual_gamma, gamma)
        self.assertEqual(comparison.VERSION_LABELS['10015'], (100, 15))

    def test_mw2_rejections_and_eigenvalues_match_explicit_projection(self):
        grid, cv = np.array([0., 10., 500.]), np.array([0., 4., 10.])
        for beta in (-2., 0., 2.):
            with self.subTest(beta=beta):
                expected, reference = self._explicit_reference(
                    (2, beta, [100, 15], 48, 7, grid, cv, 8., 111, 24))
                actual_matrices = []

                def capture(a, b, **kwargs):
                    ev = eigh(a, b, **kwargs)
                    actual_matrices.append((a.copy(), b.copy(), ev[::-1]))
                    return ev

                with mock.patch.object(comparison, 'eigh', side_effect=capture):
                    actual = comparison._dgp_chunk_worker(
                        (2, beta, [100, 15], 48, 7, 24, grid, cv, 8., 111))
                self.assertEqual(actual, expected)
                self.assertEqual(len(actual_matrices), 24)
                for actual_draw, expected_draw in zip(actual_matrices, reference):
                    for observed, explicit in zip(actual_draw, expected_draw):
                        np.testing.assert_allclose(observed, explicit, rtol=2e-12, atol=2e-12)

    def test_mw3_original_draw_stream_and_numerical_path_are_unchanged(self):
        grid, cv = np.array([0., 10., 500.]), np.array([0., 4., 10.])
        expected, reference = self._explicit_reference(
            (0, .75, [35, 25, 15], 48, 7, grid, cv, 8., 765, 8))
        actual_matrices = []

        def capture(a, b, **kwargs):
            ev = eigh(a, b, **kwargs)
            actual_matrices.append((a.copy(), b.copy(), ev[::-1]))
            return ev

        with mock.patch.object(comparison, 'eigh', side_effect=capture):
            actual = comparison._dgp_chunk_worker(
                (0, .75, [35, 25, 15], 48, 7, 8, grid, cv, 8., 765))
        self.assertEqual(actual, expected)
        for actual_draw, expected_draw in zip(actual_matrices, reference):
            for observed, explicit in zip(actual_draw, expected_draw):
                np.testing.assert_array_equal(observed, explicit)

    def test_mw2_parallel_workers_preserve_all_draws(self):
        options = dict(betas=[-2., 0., 2.], kappas=[100, 15], n=32, k=7,
                       num_simulations=11, chunk_size=4, base_seed=101,
                       hat_k1_grid=np.array([0., 10., 500.]),
                       cv_grid=np.array([0., 4., 10.]))
        serial = comparison.simulate_power_dgp(n_workers=1, **options)
        parallel = comparison.simulate_power_dgp(n_workers=2, **options)
        np.testing.assert_array_equal(serial, parallel)

    def test_cli_reports_exact_17_point_scale_without_running_simulations(self):
        script = Path(comparison.__file__).resolve()
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [sys.executable, str(script), '--version', '10015',
                 '--beta-count', '17', '--preflight-only', '--no-show'],
                cwd=directory, text=True, capture_output=True, check=True)
            self.assertIn('17 betas × 100,000 = 1,700,000', result.stdout)
            self.assertFalse((Path(directory) / '10015/dgp/dgp_curves_10015.npz').exists())

    def test_cached_mw2_cli_writes_csv_errors_and_noninteractive_plot(self):
        script = Path(comparison.__file__).resolve()
        betas = np.linspace(-2., 2., 17)
        powers = np.linspace(.05, .85, 17)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / '10015/dgp'
            output.mkdir(parents=True)
            cache = output / 'dgp_curves_10015.npz'
            comparison.save_dgp_cache(
                str(cache), version_label='10015', kappas=[100, 15], k=7, n=250,
                alpha=.05, betas=betas, power_chi2=powers, power_c1=powers,
                power_cp1=powers, num_simulations=20, base_seed=111,
                chunk_size=4, workers_used=1)
            original = cache.read_bytes()
            result = subprocess.run(
                [sys.executable, str(script), '--version', '10015',
                 '--beta-count', '17', '--num-simulations', '20',
                 '--seed', '111', '--chunk-size', '4', '--no-show'],
                cwd=directory, text=True, capture_output=True, check=True,
                env={**os.environ, 'MPLBACKEND': 'Agg'})
            self.assertIn('Loaded compatible DGP cache.', result.stdout)
            self.assertEqual(cache.read_bytes(), original)
            self.assertTrue((output / 'power_curve_kappas_100_15.png').is_file())
            with (output / 'dgp_curves_10015.csv').open(newline='') as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 17)
            self.assertIn('power_c2', rows[0])
            self.assertNotIn('power_c3', rows[0])
            np.testing.assert_array_equal([float(row['beta']) for row in rows], betas)
            np.testing.assert_allclose(
                [float(row['power_c2_se']) for row in rows],
                np.sqrt(powers * (1 - powers) / 20))


if __name__ == '__main__':
    unittest.main()
