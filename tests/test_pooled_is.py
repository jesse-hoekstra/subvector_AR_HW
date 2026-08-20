import inspect
import os
import tempfile
import unittest
from unittest import mock

import numpy as np

import alfd_eigval as alfd


def rational_two_state_bank(base_weights=None, common_log_factor=None):
    """A balanced two-stratum bank whose empirical frequencies are exact."""
    # Under f0, state frequencies are 8/10 and 2/10; under f1 they are
    # 2/10 and 8/10.  Thus the pooled empirical proposal is exactly uniform.
    states = np.array([0] * 8 + [1] * 2 + [0] * 2 + [1] * 8)
    strata = np.array([0] * 10 + [1] * 10)
    probabilities = np.array([[0.8, 0.2], [0.2, 0.8]])
    log_f = np.log(probabilities[:, states])
    if common_log_factor is not None:
        log_f = log_f + np.asarray(common_log_factor, dtype=float)[None, :]
    log_q = alfd.logsumexp(log_f - np.log(2.0), axis=0)
    if base_weights is None:
        base_weights = np.full(states.size, 1.0 / states.size)
    # The numerical values are only state labels here.  The extra trailing
    # coordinate mirrors a rank-deficient null eigenvalue.
    eigs = np.column_stack((states + 1.0, np.zeros(states.size)))
    return alfd.PooledISBank(
        grid=np.array([[1.0], [2.0]]),
        eigs=eigs,
        log_f=log_f,
        log_q=log_q,
        base_weights=np.asarray(base_weights, dtype=float),
        strata=strata,
        n_per_stratum=10,
        role="gkm",
        bank_id="rational-two-state-gkm",
    )


class RawTailCalibrationTests(unittest.TestCase):
    def test_raw_tail_is_not_self_normalized(self):
        rule = alfd.calibrate_raw_weighted_tail(
            [3.0, 2.0, 1.0], [0.2, 0.2, 0.2], alpha=0.25)
        self.assertEqual(rule.threshold, 2.0)
        self.assertAlmostEqual(rule.tie_probability, 0.25)
        self.assertAlmostEqual(rule.empirical_size, 0.25)

        normalized = alfd.calibrate_raw_weighted_tail(
            [3.0, 2.0, 1.0], np.full(3, 1.0 / 3.0), alpha=0.25)
        self.assertEqual(normalized.threshold, 3.0)
        self.assertAlmostEqual(normalized.tie_probability, 0.75)

    def test_raw_tail_rejects_insufficient_total_mass(self):
        with self.assertRaisesRegex(ValueError, "mass"):
            alfd.calibrate_raw_weighted_tail(
                [2.0, 1.0], [0.01, 0.02], alpha=0.05)

    def test_common_rule_handles_target_specific_tie_mass(self):
        scores = np.array([4.0, 3.0, 2.0, 1.0])
        contributions = np.array([
            [0.03, 0.02, 0.50, 0.45],
            [0.01, 0.08, 0.45, 0.46],
        ])
        rule = alfd.common_grid_raw_is_tail_rule(
            scores, contributions, alpha=0.05)
        self.assertEqual(rule.threshold, 3.0)
        self.assertAlmostEqual(rule.tie_probability, 0.5)
        sizes = contributions @ alfd.tail_rejection_probabilities(scores, rule)
        np.testing.assert_allclose(sizes, [0.04, 0.05], atol=1e-15)


class PooledISBankTests(unittest.TestCase):
    @staticmethod
    def _authenticated_bank():
        grid = np.array([[0.0, 0.0, 0.0], [1.0, 0.5, 0.1]])

        def fake_density(eigs, omegas, *args, **kwargs):
            values = (-0.01 * np.asarray(omegas).sum(axis=1)[:, None]
                      - 0.001 * np.asarray(eigs).sum(axis=1)[None, :])
            diagnostics = {
                "pairs": int(values.size),
                "raw_evaluations": int(values.size),
                "order_counts": {"20": int(values.size)},
                "max_order": 20,
                "max_remainder_ratio": 0.0,
            }
            return values, diagnostics

        with mock.patch.object(
                alfd, "log_eigval_density_partial", side_effect=fake_density):
            bank = alfd.build_or_load_pooled_is_bank(
                grid, 7, 2, 101, role="gkm")
        return bank

    def test_rational_bank_recovers_both_target_probabilities(self):
        bank = rational_two_state_bank()
        rejection = (bank.eigs[:, 0] == 2.0).astype(float)
        actual = alfd.gkm_importance_rejection_probabilities(bank, rejection)
        np.testing.assert_allclose(actual, [0.2, 0.8], atol=2e-15)

        diagnostics = alfd.pooled_is_diagnostics(bank, rejection)
        np.testing.assert_allclose(diagnostics["raw_mass"], [1.0, 1.0])
        self.assertLessEqual(
            float(np.max(diagnostics["observed_max_ratio"])), 2.0)

    def test_observation_common_density_factor_cancels(self):
        base = rational_two_state_bank()
        offset = np.linspace(-500.0, 500.0, base.eigs.shape[0])
        shifted = rational_two_state_bank(common_log_factor=offset)
        np.testing.assert_allclose(
            alfd._pooled_is_ratios(base),
            alfd._pooled_is_ratios(shifted),
            rtol=2e-14, atol=0.0)

        log_g = np.log(np.where(base.eigs[:, 0] == 1.0, 0.4, 0.6))
        fit = alfd.fit_gkm_weights_is(
            base, log_g, alpha=0.25, n_iter=20,
            step_size=2.0)
        shifted_fit = alfd.fit_gkm_weights_is(
            shifted, log_g + offset, alpha=0.25, n_iter=20,
            step_size=2.0)
        np.testing.assert_allclose(fit.weights, shifted_fit.weights, atol=2e-14)
        np.testing.assert_allclose(
            fit.rejection_probabilities,
            shifted_fit.rejection_probabilities,
            atol=2e-14)
        self.assertAlmostEqual(fit.training_rule.threshold,
                               shifted_fit.training_rule.threshold,
                               places=12)
        # A huge common log offset can split an artificial exact tie by a few
        # ulps after subtraction.  The representation of the randomized tie
        # may therefore differ even though all target rejection probabilities
        # and the attained raw size are invariant.
        self.assertAlmostEqual(fit.training_rule.empirical_size, 0.25)
        self.assertAlmostEqual(shifted_fit.training_rule.empirical_size, 0.25)

    def test_nonuniform_base_weights_fail_equal_mixture_contract(self):
        base = np.full(20, 1.0 / 20.0)
        base[0] += 0.01
        base[1] -= 0.01
        bank = rational_two_state_bank(base_weights=base)
        with self.assertRaisesRegex(ValueError, "bank"):
            alfd._validate_pooled_is_bank(bank)

    def test_stratum_order_must_match_common_bank_contract(self):
        bank = rational_two_state_bank()
        bank.strata = np.roll(bank.strata, 1)
        with self.assertRaisesRegex(ValueError, "pooled"):
            alfd._validate_pooled_is_bank(bank)

    def test_noninteger_strata_are_not_silently_truncated(self):
        bank = rational_two_state_bank()
        bank.strata = bank.strata.astype(float)
        with self.assertRaisesRegex(ValueError, "integer"):
            alfd._validate_pooled_is_bank(bank)

    def test_cache_rejects_tampered_canonical_settings(self):
        grid = np.array([[0.0, 0.0, 0.0], [1.0, 0.5, 0.1]])

        def fake_density(eigs, omegas, *args, **kwargs):
            values = (-0.01 * np.asarray(omegas).sum(axis=1)[:, None]
                      - 0.001 * np.asarray(eigs).sum(axis=1)[None, :])
            diagnostics = {
                "pairs": int(values.size),
                "raw_evaluations": int(values.size),
                "order_counts": {"20": int(values.size)},
                "max_order": 20,
                "max_remainder_ratio": 0.0,
            }
            return values, diagnostics

        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
                alfd, "log_eigval_density_partial", side_effect=fake_density):
            alfd.build_or_load_pooled_is_bank(
                grid, 7, 2, 1234, role="gkm", cache_dir=directory)
            paths = [os.path.join(directory, name)
                     for name in os.listdir(directory) if name.endswith(".npz")]
            self.assertEqual(len(paths), 1)
            path = paths[0]
            with np.load(path, allow_pickle=False) as archive:
                payload = {name: np.asarray(archive[name]).copy()
                           for name in archive.files}
            payload["settings_json"] = np.array("{}")
            np.savez(path, **payload)
            with self.assertRaisesRegex(RuntimeError, "settings"):
                alfd.build_or_load_pooled_is_bank(
                    grid, 7, 2, 1234, role="gkm", cache_dir=directory)

    def test_mhg_diagnostics_allow_order_zero_analytic_shortcuts(self):
        diagnostics = {
            "pairs": 8,
            "raw_evaluations": 4,
            "order_counts": {0: 4, 20: 4},
            "max_order": 20,
            "max_remainder_ratio": 0.0,
        }
        canonical = alfd._canonical_pooled_mhg_diagnostics(
            diagnostics, expected_pairs=8)
        self.assertEqual(canonical, (
            '{"max_order":20,"max_remainder_ratio":0.0,'
            '"order_counts":{"0":4,"20":4},"pairs":8,'
            '"raw_evaluations":4}'))

        invalid = {**diagnostics, "raw_evaluations": 3}
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            alfd._canonical_pooled_mhg_diagnostics(
                invalid, expected_pairs=8)

    def test_direct_path_rejects_density_setting_mismatch(self):
        bank = self._authenticated_bank()
        with self.assertRaisesRegex(ValueError, "adaptive-M"):
            alfd.gkm_eigval_bound_from_pooled_bank(
                [3.0, 2.0, 1.0, 0.5], bank, 7,
                M_trunc=21, verbose=False)

    def test_authenticated_bank_rejects_tampered_content_signature(self):
        bank = self._authenticated_bank()
        bank.content_signature = "0" * 64
        with self.assertRaisesRegex(ValueError, "content signature"):
            alfd._authenticated_pooled_bank_settings(bank)

    def test_authenticated_pooled_path_runs_end_to_end(self):
        grid = [(0.0, 0.0, 0.0), (2.0, 1.0, 0.2)]

        def fake_density(samples, omegas, *args, **kwargs):
            samples = np.asarray(samples, dtype=float)
            omegas = np.asarray(omegas, dtype=float)
            values = -0.001 * np.sum(
                (samples[None, :, :] - omegas[:, None, :]) ** 2, axis=2)
            pairs = int(values.size)
            diagnostics = {
                "pairs": pairs, "raw_evaluations": pairs,
                "order_counts": {20: pairs}, "max_order": 20,
                "max_remainder_ratio": 0.0,
            }
            return (values, diagnostics) if kwargs.get(
                "return_diagnostics") else values

        with mock.patch.object(
                alfd, "log_eigval_density_partial",
                side_effect=fake_density):
            bank = alfd.build_or_load_pooled_is_bank(
                grid, 7, 100, 101, role="gkm")
            result = alfd.gkm_eigval_bound_from_pooled_bank(
                [0.1, 0.08, 0.04, 0.01], bank, 7,
                n_sim_power=500, n_iter=20, seed=303, verbose=False)

        self.assertAlmostEqual(result.mixture_rule.empirical_size, 0.05)
        self.assertLessEqual(result.bound, result.mixture_power + 1e-12)
        self.assertAlmostEqual(
            result.epsilon_grid, result.mixture_power - result.bound,
            places=14)
        self.assertEqual(result.weights.shape, (len(grid),))
        self.assertEqual(
            result.fit_rejection_probabilities.shape, (len(grid),))
        self.assertEqual(
            result.grid_rejection_probabilities.shape, (len(grid),))
        self.assertLessEqual(
            float(np.max(result.grid_rejection_probabilities)),
            0.05 + 2e-12)
        self.assertEqual(result.fit_iterations, 20)

    def test_fixed_mu_update_matches_gkm_four_iteration_regression(self):
        bank = rational_two_state_bank()
        log_g = np.where(bank.eigs[:, 0] == 1.0, -0.75, -4.0)

        fit = alfd.fit_gkm_weights_is(
            bank, log_g, alpha=0.25, n_iter=4, step_size=2.0)

        # Starting from (-2, -2), the four rejection vectors are state 0,
        # state 0, empty, empty. Their exact null probabilities are (.8,.2),
        # (.8,.2), (0,0), and (0,0), respectively.
        np.testing.assert_allclose(fit.mu, [-0.8, -3.2],
                                   rtol=0.0, atol=5e-15)
        np.testing.assert_allclose(
            fit.weights, alfd._softmax([-0.8, -3.2]),
            rtol=0.0, atol=5e-15)
        self.assertEqual(fit.iterations, 4)

    def test_full_support_scoring_survives_reported_weight_underflow(self):
        density = np.array([
            [1e3, 1e3, 1e-300, 1e-300],
            [1e-300, 1e-300, 1e3, 1e3],
        ])
        proposal = density.mean(axis=0)
        bank = alfd.PooledISBank(
            grid=np.array([[2.0, 1.0, 0.0], [1.0, 0.5, 0.0]]),
            eigs=np.zeros((4, 4)), log_f=np.log(density),
            log_q=np.log(proposal), base_weights=np.full(4, 0.25),
            strata=np.array([0, 0, 1, 1]), n_per_stratum=2,
            role="gkm", bank_id="underflow-test")
        log_g = np.log(np.array([1e300, 1e300, 1e-300, 1e-300]))

        fit = alfd.fit_gkm_weights_is(
            bank, log_g, alpha=0.05, n_iter=600, step_size=2.0)

        self.assertEqual(fit.weights[1], 0.0)
        self.assertTrue(np.all(np.isfinite(fit.log_weights)))
        self.assertAlmostEqual(float(np.logaddexp.reduce(fit.log_weights)),
                               0.0, places=14)
        # The second null row has a compensatingly large density.  Scoring
        # from exponentiated weights would silently delete it, while the
        # authoritative log weights retain the full-H GKM mixture.
        log_densities = np.array([[0.0], [800.0], [0.0]])
        score = alfd._score_from_log_densities(
            log_densities, log_weights=fit.log_weights)
        dropped_score = alfd._score_from_log_densities(
            log_densities, fit.weights)
        self.assertLess(float(score[0]), -50.0)
        self.assertAlmostEqual(float(dropped_score[0]), 0.0, places=14)


class GridTests(unittest.TestCase):
    def test_common_grid_is_ordered_and_has_declared_size(self):
        grid = np.asarray(alfd.common_null_grid_3d(
            [[10.0, 4.0, 1.0], [8.0, 5.0, 2.0]],
            [10.0, 5.0, 2.0], n_shapes=6, n_strengths=7,
            max_strength=100.0))
        self.assertEqual(grid.shape, (43, 3))
        self.assertTrue(np.all(grid >= 0.0))
        self.assertTrue(np.all(np.diff(grid, axis=1) <= 1e-12))
        np.testing.assert_array_equal(grid[0], np.zeros(3))


class DriverAndAccountingTests(unittest.TestCase):
    def test_shared_pair_accounting_counts_null_tables_once(self):
        budget = dict(
            n_fit=10, n_power=2000, n_iter=5)
        result = alfd._gkm_budget_diagnostics(
            alpha=0.05, common_grid_size=3,
            n_nonnull=2, budget=budget)
        expected = {
            "shared_null_bank": 3 * 3 * 10,
            "beta_training_alternative": 2 * 3 * 10,
            "beta_power": 2 * (3 + 1) * 2000,
        }
        self.assertEqual(result["phase_pairs"], expected)
        self.assertEqual(result["total_pairs"], sum(expected.values()))

    def test_production_driver_calls_pooled_path(self):
        source = inspect.getsource(alfd.main)
        self.assertNotIn("prepared_grids", source)
        self.assertIn("gkm_eigval_bound_from_pooled_bank(", source)
        self.assertNotIn("alfd_eigval_bound_from_pooled_banks(", source)
        self.assertNotIn("compress_mixture(", source)


if __name__ == "__main__":
    unittest.main()
