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
        role="training",
        bank_id="rational-two-state-training",
    )


class RawTailCalibrationTests(unittest.TestCase):
    def test_raw_tail_is_not_self_normalized(self):
        rule = alfd.calibrate_raw_weighted_tail(
            [3.0, 2.0, 1.0], [0.2, 0.2, 0.2], alpha=0.25)
        self.assertEqual(rule.threshold, 2.0)
        self.assertAlmostEqual(rule.tie_probability, 0.25)
        self.assertAlmostEqual(rule.empirical_size, 0.25)

        normalized = alfd.calibrate_weighted_tail(
            [3.0, 2.0, 1.0], [0.2, 0.2, 0.2], alpha=0.25)
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
    def _authenticated_banks():
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
            training = alfd.build_or_load_pooled_is_bank(
                grid, 7, 2, 101, role="training")
            audit = alfd.build_or_load_pooled_is_bank(
                grid, 7, 2, 202, role="audit")
        return training, audit

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
        fit = alfd.fit_emw_weights_is(
            base, log_g, alpha=0.25, n_iter=20,
            convergence_patience=2)
        shifted_fit = alfd.fit_emw_weights_is(
            shifted, log_g + offset, alpha=0.25, n_iter=20,
            convergence_patience=2)
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

    def test_stratum_order_must_match_audit_reshape_contract(self):
        bank = rational_two_state_bank()
        bank.strata = np.roll(bank.strata, 1)
        with self.assertRaisesRegex(ValueError, "strata"):
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
                grid, 7, 2, 1234, role="training", cache_dir=directory)
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
                    grid, 7, 2, 1234, role="training", cache_dir=directory)

    def test_certification_rejects_density_setting_mismatch(self):
        training, audit = self._authenticated_banks()
        with self.assertRaisesRegex(ValueError, "adaptive-M"):
            alfd.alfd_eigval_bound_from_pooled_banks(
                [3.0, 2.0, 1.0, 0.5], training, audit, 7,
                M_trunc=21, verbose=False)

    def test_distinct_metadata_cannot_disguise_duplicate_bank_content(self):
        training, audit = self._authenticated_banks()
        for name in ("grid", "eigs", "log_f", "log_q", "base_weights",
                     "strata"):
            setattr(audit, name, np.asarray(getattr(training, name)).copy())
        audit.mhg_diagnostics = dict(training.mhg_diagnostics)
        diagnostics_json = alfd._canonical_pooled_mhg_diagnostics(
            audit.mhg_diagnostics, audit.log_f.size)
        audit.content_signature = alfd._pooled_bank_content_signature(
            grid=audit.grid, eigs=audit.eigs, log_f=audit.log_f,
            log_q=audit.log_q, base_weights=audit.base_weights,
            strata=audit.strata, diagnostics_json=diagnostics_json)
        with self.assertRaisesRegex(ValueError, "identical numeric content"):
            alfd.alfd_eigval_bound_from_pooled_banks(
                [3.0, 2.0, 1.0, 0.5], training, audit, 7,
                verbose=False)

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
            training = alfd.build_or_load_pooled_is_bank(
                grid, 7, 100, 101, role="training")
            audit = alfd.build_or_load_pooled_is_bank(
                grid, 7, 100, 202, role="audit")
            result = alfd.alfd_eigval_bound_from_pooled_banks(
                [0.1, 0.08, 0.04, 0.01], training, audit, 7,
                n_sim_calibration=500, n_sim_power=500, n_iter=20,
                max_active_support=2, seed=303, confidence_delta=0.4,
                verbose=False)

        self.assertAlmostEqual(result.point_rule.empirical_size, 0.05)
        self.assertGreaterEqual(result.upper_confidence, 0.05)
        self.assertLessEqual(result.lower_grid_point, result.upper_point)
        self.assertLessEqual(result.gkm_grid_lower, result.gkm_point_upper)
        self.assertEqual(result.weights.shape, (2,))
        self.assertEqual(result.full_weights.shape, (2,))


class CompressionAndGridTests(unittest.TestCase):
    def test_compression_retains_indices_and_alignment(self):
        active, retained, dropped = alfd.compress_mixture(
            [0.10, 0.60, 0.05, 0.25], max_active=2)
        np.testing.assert_array_equal(active, [1, 3])
        np.testing.assert_allclose(retained, [0.60 / 0.85, 0.25 / 0.85])
        self.assertAlmostEqual(dropped, 0.15)
        self.assertAlmostEqual(float(retained.sum()), 1.0)

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
            n_fit=10, n_calibration=2000, n_validation=2000,
            n_power=2000, n_iter=5, validation_grid_size=3)
        result = alfd._common_is_budget_diagnostics(
            alpha=0.05, curve_confidence=0.90, common_grid_size=3,
            n_nonnull=2, max_active_support=2, budget=budget)
        expected = {
            "shared_fit_null": 3 * 3 * 10,
            "fit_alternative": 2 * 3 * 10,
            "calibration": 2 * (2 + 1) * 2000,
            "shared_audit_null": 3 * 3 * 2000,
            "audit_alternative": 2 * 3 * 2000,
            "power": 2 * (2 + 1) * 2000,
        }
        self.assertEqual(result["phase_pairs"], expected)
        self.assertEqual(result["total_pairs"], sum(expected.values()))

    def test_production_driver_calls_pooled_path(self):
        source = inspect.getsource(alfd.main)
        self.assertNotIn("prepared_grids", source)
        self.assertIn("alfd_eigval_bound_from_pooled_banks(", source)


if __name__ == "__main__":
    unittest.main()
