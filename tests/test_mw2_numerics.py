import tempfile
import unittest
from unittest import mock

import numpy as np

import alfd_eigval as alfd


class TwoDimensionalDGPTests(unittest.TestCase):
    def test_reduced_design_retains_first_two_nuisance_coordinates(self):
        sigma, loading, pi_x, gamma = alfd._dgp_constants(7, 250, m_W=2)
        full_sigma, full_loading, full_pi_x, full_gamma = alfd._dgp_constants(7, 250)
        expected_sigma = np.array([
            [1.0, 0.1, 0.3, 0.2],
            [0.1, 1.0, 0.3, 0.2],
            [0.3, 0.3, 1.0, 0.3],
            [0.2, 0.2, 0.3, 1.0],
        ])
        np.testing.assert_array_equal(sigma, expected_sigma)
        np.testing.assert_array_equal(sigma, full_sigma[:4, :4])
        self.assertEqual(loading.shape, (7, 2))
        np.testing.assert_array_equal(loading, full_loading[:, :2])
        np.testing.assert_allclose(250 * loading.T @ loading, np.eye(2), atol=1e-15)
        np.testing.assert_array_equal(gamma, [-1, 1])
        np.testing.assert_array_equal(gamma, full_gamma[:2])
        self.assertEqual(pi_x.shape, (7,))
        np.testing.assert_array_equal(pi_x, full_pi_x)
        np.testing.assert_allclose(
            pi_x, 4 / np.sqrt(7 * 250) * np.array([1, 1, 1, -1, 1, 1, 1]))

    def test_mw2_null_strengths_and_nonnull_rank(self):
        np.testing.assert_allclose(
            alfd.asymptotic_ncp_eigenvalues(0, [100, 15], 7, 250),
            [100, 15, 0], rtol=1e-12, atol=1e-12)
        for beta in (-2, -0.5, 0.5, 2):
            eigs = alfd.asymptotic_ncp_eigenvalues(beta, [100, 15], 7, 250)
            self.assertEqual(eigs.shape, (3,))
            self.assertTrue(np.all(np.diff(eigs) <= 0.0))
            self.assertGreater(eigs[-1], 0.0)

    def test_original_mw3_ncp_values_are_unchanged(self):
        # Baselines from the original three-nuisance Appendix A.3 design.
        expected = [
            [36.07808911683758, 26.16584892423004,
             16.748970353518555, 1.2739472993943581],
            [35, 25, 15, 0],
            [34.39486091571005, 25.806652247159402,
             8.596068621200184, 2.705452425031837],
        ]
        actual = [alfd.asymptotic_ncp_eigenvalues(beta, [35, 25, 15], 7, 250)
                  for beta in (-2, 0, 2)]
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


class TwoDimensionalNullGridTests(unittest.TestCase):
    def test_grid_covers_configuration_boundaries_and_path_with_36_rows(self):
        alternatives = np.array([[110.0, 11.0], [90.0, 36.0]])
        grid = np.asarray(alfd.common_null_grid_2d(alternatives, [100, 15]))
        self.assertEqual(grid.shape, (36, 2))
        np.testing.assert_array_equal(grid[0], [0.0, 0.0])
        self.assertTrue(np.all(grid[:, 0] >= grid[:, 1]))
        self.assertTrue(np.all(grid >= 0.0))
        self.assertEqual(len(np.unique(grid, axis=0)), len(grid))
        for expected in ([100, 15], [100, 0], [100, 100],
                         [100, 10], [100, 40]):
            self.assertTrue(any(np.allclose(row, expected) for row in grid))
        repeated = alfd.common_null_grid_2d(alternatives * 2, [100, 15])
        np.testing.assert_allclose(grid, repeated, rtol=0.0, atol=0.0)
        denser = np.asarray(alfd.common_null_grid_2d(
            alternatives, [100, 15], n_shapes=9))
        self.assertEqual(denser.shape, (64, 2))
        np.testing.assert_array_equal(denser[:len(grid)], grid)

    def test_exact_config_anchor_is_retained_off_the_strength_grid(self):
        grid = alfd.common_null_grid_2d(
            np.empty((0, 2)), [100, 15], n_shapes=2,
            n_strengths=2, max_strength=80)
        self.assertEqual(len(grid), 6)
        self.assertEqual(grid[-1], (100.0, 15.0))

    def test_fallback_adds_unique_shapes_and_invalid_rows_are_rejected(self):
        grid = np.asarray(alfd.common_null_grid_2d(
            np.empty((0, 2)), [100, 15], n_shapes=25))
        self.assertEqual(grid.shape, (176, 2))
        self.assertEqual(len(np.unique(grid, axis=0)), len(grid))
        for config, alternatives in (
                ([100, 15, 0], [[100, 15]]),
                ([100, 15], [[100, 15, 0]]),
                ([15, 100], [[100, 15]]),
                ([100, 15], [[1, -1]])):
            with self.subTest(config=config, alternatives=alternatives):
                with self.assertRaises(ValueError):
                    alfd.common_null_grid_2d(alternatives, config)


class DimensionGenericBankTests(unittest.TestCase):
    def test_native_p3_bank_matches_parallel_and_cache_reload(self):
        grid = [[0.0, 0.0], [0.2, 0.1]]
        with tempfile.TemporaryDirectory() as directory:
            serial = alfd.build_or_load_pooled_is_bank(
                grid, 7, 3, 101, n_workers=1, cache_dir=directory)
            parallel = alfd.build_or_load_pooled_is_bank(
                grid, 7, 3, 101, n_workers=2)
            self.assertEqual(serial.eigs.shape, (6, 3))
            self.assertEqual(serial.log_f.shape, (2, 6))
            np.testing.assert_array_equal(serial.eigs, parallel.eigs)
            np.testing.assert_array_equal(serial.log_f, parallel.log_f)
            self.assertEqual(serial.content_signature, parallel.content_signature)
            self.assertEqual(serial.bank_id, parallel.bank_id)
            with mock.patch.object(
                    alfd, "log_eigval_density_partial",
                    side_effect=AssertionError("cache reload computed densities")):
                cached = alfd.build_or_load_pooled_is_bank(
                    grid, 7, 3, 101, n_workers=2, cache_dir=directory)
            self.assertEqual(serial.bank_id, cached.bank_id)
            np.testing.assert_array_equal(serial.log_f, cached.log_f)

    def test_bank_rejects_invalid_dimension_before_sampling(self):
        with mock.patch.object(
                alfd, "simulate_Xi",
                side_effect=AssertionError("invalid bank sampled")):
            for grid, k_eff in (([[1, 0]], 2), ([[]], 7),
                                ([[1, 0], [1]], 7)):
                with self.subTest(grid=grid, k_eff=k_eff):
                    with self.assertRaises(ValueError):
                        alfd.build_or_load_pooled_is_bank(grid, k_eff, 2, 101)

    def test_native_p3_solver_and_rejection_of_mismatched_alternative(self):
        bank = alfd.build_or_load_pooled_is_bank(
            [[0.0, 0.0], [0.2, 0.1]], 7, 3, 101)
        result = alfd.gkm_eigval_bound_from_pooled_bank(
            [0.3, 0.2, 0.1], bank, 7, n_sim_power=12,
            n_iter=3, seed=102, verbose=False)
        self.assertTrue(0.0 <= result.bound <= result.mixture_power <= 1.0)
        self.assertLessEqual(max(result.grid_rejection_probabilities), 0.05 + 1e-12)
        with mock.patch.object(
                alfd, "log_eigval_density_partial",
                side_effect=AssertionError("invalid alternative computed densities")):
            for eigenvalues in ([0.3, 0.2, 0.1, 0], [[0.3, 0.2, 0.1]]):
                with self.assertRaisesRegex(ValueError, "p=3"):
                    alfd.gkm_eigval_bound_from_pooled_bank(
                        eigenvalues, bank, 7, n_sim_power=2,
                        n_iter=1, seed=102, verbose=False)

    def test_p4_bank_remains_supported_and_has_separate_identity(self):
        banks = [alfd.build_or_load_pooled_is_bank(
            [np.zeros(dimension)], 7, 2, 101) for dimension in (2, 3)]
        self.assertEqual(banks[1].eigs.shape, (2, 4))
        self.assertNotEqual(banks[0].bank_id, banks[1].bank_id)


class DimensionGenericBenchmarkTests(unittest.TestCase):
    def test_native_p3_benchmark_reports_only_null_work_when_requested(self):
        grid = [[0.0, 0.0], [0.2, 0.1]]
        result = alfd._benchmark_adaptive_mhg(
            [[0.3, 0.2, 0.1], [0.2, 0.1, 0], [0.4, 0.2, 0.1]],
            [-1, 0, 1], 100, 7, 20, 20, 300, 1e-10,
            n_workers=1, n_samples=4, fit_grids=[grid] * 3,
            null_bank_only=True)
        self.assertEqual(result["benchmark_scope"], "all_fitted_null_rows")
        self.assertEqual(result["null_samples"], 4)
        self.assertEqual(result["alternative_samples"], 0)
        self.assertEqual(result["omega_rows"], 2)
        self.assertEqual(result["fit_null_rows"], 2)
        self.assertEqual(result["pairs"], 8)


if __name__ == "__main__":
    unittest.main()
