import os
import sys
import tempfile
import unittest
import hashlib
import json
from unittest import mock

import numpy as np
from scipy.special import hyp0f1
from scipy.stats import chi2, ncx2

import alfd_eigval as alfd
from new_power_comparison import (
    dgp_cache_path,
    load_compatible_alfd_bound,
    load_compatible_dgp_cache,
    save_dgp_cache,
)


class CalibrationTests(unittest.TestCase):
    def test_weighted_tail_randomizes_ties_to_exact_size(self):
        scores = np.array([4.0, 3.0, 3.0, 2.0, 1.0])
        weights = np.array([0.1, 0.2, 0.3, 0.1, 0.3])
        rule = alfd.calibrate_weighted_tail(scores, weights, alpha=0.35)
        attained = np.dot(
            weights, alfd.tail_rejection_probabilities(scores, rule))
        self.assertAlmostEqual(attained, 0.35, places=14)
        self.assertEqual(rule.threshold, 3.0)
        self.assertAlmostEqual(rule.tie_probability, 0.5)

    def test_confidence_rules_bracket_empirical_alpha_tail(self):
        scores = np.linspace(0.0, 1.0, 20_000, endpoint=False)
        liberal = alfd.confidence_liberal_tail_rule(
            scores, alpha=0.05, delta=0.0025)
        conservative = alfd.confidence_conservative_tail_rule(
            scores, alpha=0.05, delta=0.0025)
        self.assertGreater(liberal.empirical_size, 0.05)
        self.assertLess(conservative.empirical_size, 0.05)
        self.assertLess(liberal.threshold, conservative.threshold)

    def test_confidence_rules_reject_nonfinite_scores(self):
        for function in (alfd.confidence_liberal_tail_rule,
                         alfd.confidence_conservative_tail_rule):
            with self.assertRaises(ValueError):
                function([0.0, np.nan], alpha=0.05, delta=0.01)

    def test_preflight_ranks_match_rules_and_phase_count(self):
        alpha = 0.05
        curve_confidence = 0.99
        budget = dict(
            n_fit=2000, n_calibration=20000, n_validation=2000,
            n_power=50000, n_iter=600, validation_grid_size=32)
        diagnostics = alfd._simulation_budget_diagnostics(
            alpha, curve_confidence, [13] * 20, [32] * 20, budget)

        self.assertEqual(diagnostics['calibration_rejection_count'], 1116)
        self.assertAlmostEqual(
            diagnostics['calibration_empirical_size'], 0.0558)
        self.assertEqual(diagnostics['validation'][0]['rejection_count'], 58)
        self.assertAlmostEqual(
            diagnostics['validation'][0]['empirical_size'], 0.029)
        self.assertEqual(diagnostics['total_pairs'], 44_800_000)

        scores_cal = np.arange(budget['n_calibration'], dtype=float)
        liberal = alfd.confidence_liberal_tail_rule(
            scores_cal, alpha, diagnostics['event_delta'])
        self.assertAlmostEqual(
            liberal.empirical_size,
            diagnostics['calibration_empirical_size'])

        scores_validation = np.arange(budget['n_validation'], dtype=float)
        conservative = alfd.confidence_conservative_tail_rule(
            scores_validation, alpha,
            diagnostics['validation'][0]['per_null_delta'])
        self.assertAlmostEqual(
            conservative.empirical_size,
            diagnostics['validation'][0]['empirical_size'])

    def test_preflight_rejects_infeasible_validation_budget(self):
        budget = dict(
            n_fit=100, n_calibration=2000, n_validation=100,
            n_power=5000, n_iter=100, validation_grid_size=32)
        with self.assertRaisesRegex(ValueError, 'cannot support'):
            alfd._simulation_budget_diagnostics(
                0.05, 0.99, [13] * 20, [32] * 20, budget)


class PublicationWorkflowTests(unittest.TestCase):
    @staticmethod
    def _synthetic_pooled_bank():
        density = np.array([
            [1.0, 3.0, 2.0, 4.0],
            [3.0, 1.0, 2.0, 1.0],
        ])
        proposal = density.mean(axis=0)
        return alfd.PooledISBank(
            grid=np.array([[2.0, 1.0, 0.0], [1.0, 0.5, 0.0]]),
            eigs=np.zeros((4, 4)), log_f=np.log(density),
            log_q=np.log(proposal), base_weights=np.full(4, 0.25),
            strata=np.array([0, 0, 1, 1]), n_per_stratum=2,
            role="training", bank_id="synthetic-bank")

    def test_common_null_grid_is_deterministic_nested_and_beta_invariant(self):
        alternatives = np.array([
            [36.0, 24.0, 4.0],
            [31.0, 23.0, 17.0],
        ])
        first = alfd.common_null_grid_3d(
            alternatives, [35.0, 25.0, 15.0])
        second = alfd.common_null_grid_3d(
            alternatives, [35.0, 25.0, 15.0])

        self.assertEqual(first, second)
        self.assertEqual(len(first), 43)
        grid = np.asarray(first)
        np.testing.assert_array_equal(grid[0], np.zeros(3))
        self.assertTrue(np.all(np.isfinite(grid)))
        self.assertTrue(np.all(grid >= 0.0))
        self.assertTrue(np.all(np.diff(grid, axis=1) <= 1e-12))

        strengths = np.geomspace(0.1, 100.0, 7)
        for shape_index in range(6):
            block = grid[1 + 7 * shape_index:1 + 7 * (shape_index + 1)]
            np.testing.assert_allclose(block[:, 0], strengths)
            normalized = block / block[:, :1]
            np.testing.assert_allclose(
                normalized, np.repeat(normalized[:1], 7, axis=0))

        # Alternative-path information chooses shape directions, but there is
        # still one fixed helper-default 43-row null grid. The production CLI
        # requests three additional stress directions below.
        perturbed_alternatives = alternatives.copy()
        perturbed_alternatives *= 2.0
        scale_only = alfd.common_null_grid_3d(
            perturbed_alternatives, [35.0, 25.0, 15.0])
        self.assertEqual(first, scale_only)

    def test_production_grid_has_nine_shapes_and_retains_stress_directions(self):
        alternatives = np.array([
            [36.0, 24.0, 4.0],
            [31.0, 23.0, 17.0],
        ])
        standard_points = [
            (35.0, 25.0, 15.0),
            (50.0, 25.0, 5.0),
            (15.0, 10.0, 5.0),
            (5.0, 3.0, 1.0),
        ]
        grid = np.asarray(alfd.common_null_grid_3d(
            alternatives, [35.0, 25.0, 15.0],
            standard_points=standard_points, n_shapes=9))

        self.assertEqual(grid.shape, (68, 3))
        ray_grid = grid[:64]
        shape_rows = ray_grid[1::7] / ray_grid[1::7, :1]
        required = [
            np.asarray(row) / row[0] for row in standard_points
        ] + [
            np.array([1.0, 0.0, 0.0]),
            np.array([1.0, 1.0, 0.0]),
            np.array([1.0, 1.0, 1.0]),
            alternatives[0] / alternatives[0, 0],
            alternatives[1] / alternatives[1, 0],
        ]
        for expected in required:
            self.assertTrue(any(np.allclose(row, expected)
                                for row in shape_rows))
        np.testing.assert_allclose(grid[-4:], standard_points)

    def test_default_cli_beta_grid_has_eleven_symmetric_points_and_zero(self):
        observed = []
        grids = []

        def fake_ncp(beta, kappas, k, n):
            observed.append(float(beta))
            return np.array([35.0, 25.0, 15.0, abs(float(beta))])

        real_grid = alfd.common_null_grid_3d

        def capture_grid(*args, **kwargs):
            result = real_grid(*args, **kwargs)
            grids.append((result, kwargs))
            return result

        argv = [
            'alfd_eigval.py', '--version', '352515',
            '--profile', 'production', '--preflight-only', '--workers', '1',
        ]
        with mock.patch.object(sys, 'argv', argv), \
                mock.patch.object(
                    alfd, 'asymptotic_ncp_eigenvalues', side_effect=fake_ncp), \
                mock.patch.object(
                    alfd, 'common_null_grid_3d', side_effect=capture_grid), \
                mock.patch('builtins.print'):
            alfd.main()

        expected = np.linspace(-2.0, 2.0, 11)
        np.testing.assert_allclose(observed, expected, rtol=0.0, atol=0.0)
        self.assertEqual(observed[5], 0.0)
        np.testing.assert_allclose(
            np.asarray(observed), -np.asarray(observed)[::-1],
            rtol=0.0, atol=1e-15)
        self.assertEqual(len(grids), 1)
        self.assertEqual(len(grids[0][0]), 68)
        self.assertEqual(grids[0][1]['n_shapes'], 9)
        self.assertEqual(len(grids[0][1]['standard_points']), 4)

    def test_common_bank_budget_counts_shared_null_tables_only_once(self):
        budget = dict(
            n_fit=2000, n_calibration=20000, n_validation=2000,
            n_power=50000, n_iter=600, validation_grid_size=32)
        diagnostics = alfd._common_is_budget_diagnostics(
            0.05, 0.99, common_grid_size=68, n_nonnull=10,
            max_active_support=8, budget=budget)

        self.assertEqual(diagnostics['phase_pairs'], {
            'shared_fit_null': 9_248_000,
            'fit_alternative': 1_360_000,
            'calibration': 1_800_000,
            'shared_audit_null': 9_248_000,
            'audit_alternative': 1_360_000,
            'power': 4_500_000,
        })
        self.assertEqual(diagnostics['total_pairs'], 27_516_000)
        self.assertEqual(diagnostics['events_per_family'], 2)
        self.assertAlmostEqual(
            diagnostics['grid_bracket_curve_confidence'], 0.98)

    def test_gkm_importance_estimator_is_ordinary_not_self_normalized(self):
        bank = self._synthetic_pooled_bank()
        rejection = np.array([1.0, 0.0, 0.25, 0.5])
        got = alfd.gkm_importance_rejection_probabilities(bank, rejection)
        expected = np.array([0.3875, 0.4875])
        np.testing.assert_allclose(got, expected, rtol=0.0, atol=1e-14)

        raw_mass = alfd.gkm_importance_rejection_probabilities(
            bank, np.ones(4))
        np.testing.assert_allclose(raw_mass, [1.15, 0.85], atol=1e-14)
        self.assertFalse(np.allclose(raw_mass, 1.0))
        self_normalized = expected / raw_mass
        self.assertFalse(np.allclose(got, self_normalized))

        diagnostics = alfd.pooled_is_diagnostics(bank, rejection)
        np.testing.assert_allclose(
            diagnostics['raw_mass'], raw_mass, atol=1e-14)
        np.testing.assert_allclose(
            diagnostics['tail_mass'], expected, atol=1e-14)

    def test_raw_tail_calibration_does_not_renormalize_is_mass(self):
        scores = np.array([4.0, 3.0, 2.0])
        raw = np.array([0.4, 0.4, 0.4])  # realized ordinary-IS mass is 1.2
        rule = alfd.calibrate_raw_weighted_tail(scores, raw, alpha=0.5)
        self.assertEqual(rule.threshold, 3.0)
        self.assertAlmostEqual(rule.tie_probability, 0.25)
        attained = np.dot(
            raw, alfd.tail_rejection_probabilities(scores, rule))
        self.assertAlmostEqual(attained, 0.5, places=14)

        # Normalizing to unit realized mass would instead randomize the tie at
        # one half and is not GKM equation (D.1).
        normalized = alfd.calibrate_weighted_tail(scores, raw, alpha=0.5)
        self.assertAlmostEqual(normalized.tie_probability, 0.5)

    def test_pooled_bank_cache_reuses_one_common_null_table(self):
        grid = [(0.0, 0.0, 0.0), (2.0, 1.0, 0.0), (4.0, 2.0, 1.0)]
        H, n_per = len(grid), 2
        pairs = H * (H * n_per)
        diagnostics = dict(
            pairs=pairs, raw_evaluations=pairs,
            order_counts={20: pairs}, max_order=20,
            max_remainder_ratio=0.0)

        def fake_eigenvalues(xi):
            return np.zeros((len(xi), 4))

        def fake_density(samples, omegas, c, **kwargs):
            values = np.zeros((len(omegas), len(samples)))
            return values, diagnostics

        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(
                    alfd, 'simulate_Xi',
                    side_effect=lambda M, n, rng: np.zeros((n, M.shape[0], 4))
                ) as simulate, \
                mock.patch.object(
                    alfd, 'eigenvalues_descending',
                    side_effect=fake_eigenvalues), \
                mock.patch.object(
                    alfd, 'log_eigval_density_partial',
                    side_effect=fake_density) as density:
            first = alfd.build_or_load_pooled_is_bank(
                grid, k_eff=7, n_per_stratum=n_per, seed=123,
                cache_dir=directory, role='training')
            second = alfd.build_or_load_pooled_is_bank(
                grid, k_eff=7, n_per_stratum=n_per, seed=123,
                cache_dir=directory, role='training')

        self.assertEqual(simulate.call_count, H)
        density.assert_called_once()
        self.assertEqual(first.bank_id, second.bank_id)
        np.testing.assert_array_equal(first.eigs, second.eigs)
        np.testing.assert_array_equal(first.log_f, second.log_f)
        self.assertEqual(first.sampling_scheme,
                         'stratified_null_gkm_is')

    def test_pooled_bank_validation_rejects_non_gkm_sampling_scheme(self):
        bank = self._synthetic_pooled_bank()
        bank.sampling_scheme = 'weighted_but_not_iid'
        with self.assertRaisesRegex(ValueError, 'invalid pooled'):
            alfd.gkm_importance_rejection_probabilities(
                bank, np.ones(4))

    def test_certification_requires_distinct_authenticated_banks(self):
        training = self._synthetic_pooled_bank()
        audit = self._synthetic_pooled_bank()
        audit.role = 'audit'

        with self.assertRaisesRegex(ValueError, 'distinct'):
            alfd.alfd_eigval_bound_from_pooled_banks(
                [3.0, 2.0, 1.0, 0.5], training, audit, k_eff=7,
                verbose=False)

        audit.bank_id = 'independent-audit-bank'
        training.sampling_seed = 1
        audit.sampling_seed = 2
        with self.assertRaisesRegex(ValueError, 'authenticated'):
            alfd.alfd_eigval_bound_from_pooled_banks(
                [3.0, 2.0, 1.0, 0.5], training, audit, k_eff=7,
                verbose=False)

    def test_compressed_support_is_deterministic_and_renormalized(self):
        active, weights, dropped = alfd.compress_mixture(
            [0.1, 0.4, 0.4, 0.1], max_active=2)
        np.testing.assert_array_equal(active, [1, 2])
        np.testing.assert_allclose(weights, [0.5, 0.5])
        self.assertAlmostEqual(dropped, 0.2)


class MHGBuildProvenanceTests(unittest.TestCase):
    def test_current_library_stamp_matches_source(self):
        lib_name = ('libmhg.dylib' if sys.platform == 'darwin'
                    else 'libmhg.so')
        stamped = alfd._verify_mhg_build_provenance(alfd.MHG_DIR, lib_name)
        self.assertEqual(
            stamped, alfd._sha256_file(os.path.join(alfd.MHG_DIR, 'mhg_core.c')))

    def test_setup_rejects_source_changed_after_build(self):
        lib_name = ('libmhg.dylib' if sys.platform == 'darwin'
                    else 'libmhg.so')
        with tempfile.TemporaryDirectory() as directory:
            source = os.path.join(directory, 'mhg_core.c')
            library = os.path.join(directory, lib_name)
            stamp = os.path.join(directory, f'{lib_name}.mhg_core.sha256')
            with open(source, 'wb') as handle:
                handle.write(b'original C source\n')
            with open(library, 'wb') as handle:
                handle.write(b'compiled library placeholder\n')
            with open(stamp, 'w', encoding='ascii') as handle:
                handle.write(alfd._sha256_file(source) + '\n')

            # Simulate editing mhg_core.c without rebuilding the shared object.
            with open(source, 'ab') as handle:
                handle.write(b'unbuilt source change\n')
            with self.assertRaisesRegex(RuntimeError, 'stale relative'):
                alfd.setup_mhg(directory)


class MHGTests(unittest.TestCase):
    def test_target_benchmark_is_deterministic_and_extrapolates(self):
        betas = np.array([-2.0, 0.0, 2.0])
        ncp = np.array([
            [36.0, 26.0, 17.0, 1.0],
            [35.0, 25.0, 15.0, 0.0],
            [34.0, 26.0, 9.0, 3.0],
        ])
        fit_grids = [
            [(36.0, 26.0, 17.0), (35.0, 25.0, 15.0),
             (20.0, 10.0, 5.0)],
            [(35.0, 25.0, 15.0)],
            [(34.0, 26.0, 9.0)],
        ]
        captured_samples = []
        captured_omegas = []

        def fake_batch(c, omegas, samples, **kwargs):
            captured_samples.append(np.asarray(samples).copy())
            captured_omegas.append(np.asarray(omegas).copy())
            pairs = len(omegas) * len(samples)
            diagnostics = dict(
                pairs=pairs, raw_evaluations=pairs,
                order_counts={90: pairs}, max_order=90,
                max_remainder_ratio=1e-12)
            return np.ones((len(omegas), len(samples))), diagnostics

        results = []
        for _ in range(2):
            with mock.patch.object(
                    alfd, 'chunked_mhg_batch', side_effect=fake_batch), \
                    mock.patch.object(
                        alfd.time, 'perf_counter', side_effect=[10.0, 12.0]):
                results.append(alfd._benchmark_adaptive_mhg(
                    ncp, betas, total_logical_pairs=100, k_eff=7,
                    M_start=20, M_step=20, M_max=300, mhg_tol=1e-10,
                    n_workers=8, n_samples=2, fit_grids=fit_grids))

        np.testing.assert_array_equal(captured_samples[0], captured_samples[1])
        self.assertEqual(captured_samples[0].shape, (2, 4))
        expected_omegas = np.array([
            [36.0, 26.0, 17.0, 0.0],
            [35.0, 25.0, 15.0, 0.0],
            [20.0, 10.0, 5.0, 0.0],
            [36.0, 26.0, 17.0, 1.0],
        ])
        np.testing.assert_array_equal(captured_omegas[0], expected_omegas)
        np.testing.assert_array_equal(captured_omegas[1], expected_omegas)
        self.assertEqual(results[0]['workers'], 2)
        self.assertEqual(results[0]['configured_workers'], 8)
        self.assertEqual(results[0]['omega_rows'], 4)
        self.assertEqual(results[0]['fit_null_rows'], 3)
        self.assertEqual(results[0]['null_samples'], 1)
        self.assertEqual(results[0]['alternative_samples'], 1)
        self.assertEqual(
            results[0]['benchmark_scope'],
            'all_fitted_null_rows_plus_alternative')
        self.assertAlmostEqual(results[0]['pairs_per_second'], 4.0)
        self.assertAlmostEqual(results[0]['measured_extrapolated_seconds'], 25.0)
        self.assertAlmostEqual(results[0]['optimistic_configured_seconds'], 6.25)
        self.assertEqual(results[0]['order_counts'], {90: 8})

        with self.assertRaisesRegex(ValueError, 'fit_grids must align'):
            alfd._benchmark_adaptive_mhg(
                ncp, betas, total_logical_pairs=100, k_eff=7,
                M_start=20, M_step=20, M_max=300, mhg_tol=1e-10,
                n_workers=1, n_samples=1, fit_grids=fit_grids[:2])

    def test_benchmark_cli_exits_before_artifact_writes(self):
        benchmark_result = dict(pairs=2)
        argv = [
            'alfd_eigval.py', '--version', '352515',
            '--profile', 'production', '--benchmark-preflight',
            '--benchmark-samples', '1', '--beta-count', '1',
            '--workers', '1',
        ]
        with mock.patch.object(sys, 'argv', argv), \
                mock.patch.object(
                    alfd, '_benchmark_adaptive_mhg',
                    return_value=benchmark_result) as run_benchmark, \
                mock.patch.object(alfd, '_print_mhg_benchmark') as print_result, \
                mock.patch.object(
                    alfd.os, 'makedirs',
                    side_effect=AssertionError('artifact directory created')), \
                mock.patch.object(
                    alfd, '_atomic_savez',
                    side_effect=AssertionError('artifact written')):
            alfd.main()

        run_benchmark.assert_called_once()
        self.assertEqual(run_benchmark.call_args.args[-1], 1)
        self.assertEqual(len(run_benchmark.call_args.kwargs['fit_grids']), 1)
        print_result.assert_called_once_with(benchmark_result)

    def test_trailing_zero_collapse_is_not_convergence(self):
        broken = alfd.assess_mhg_series(
            9.75, [1, 2, 3, 2, 1, 0.5, 0.25, 0, 0],
            tol=1e-10, ratio_window=3)
        self.assertFalse(broken.converged)
        self.assertTrue(broken.numerical_collapse)

        # Historical scalar-core signature: only a few post-peak terms remain
        # before a still-material series becomes exactly zero.
        short_tail = alfd.assess_mhg_series(
            10.0, [1, 2, 3, 4, 3, 2, 1, 0, 0],
            tol=1e-10, ratio_window=5)
        self.assertFalse(short_tail.converged)
        self.assertTrue(short_tail.numerical_collapse)

        harmless = alfd.assess_mhg_series(
            4.101000010000001,
            [1, 2, 1, 0.1, 0.001, 1e-8, 1e-15, 0, 0],
            tol=1e-10, ratio_window=3)
        self.assertTrue(harmless.converged)
        self.assertFalse(harmless.numerical_collapse)

    def test_adaptive_parameters_fail_closed(self):
        invalid_options = (
            dict(M_start=20.5), dict(M_step=0), dict(M_max=0),
            dict(tol=1e-14), dict(tol=np.nan), dict(ratio_window=1),
            dict(trace_margin=-1.0),
        )
        for options in invalid_options:
            with self.subTest(options=options), self.assertRaises(ValueError):
                alfd.mhg_two_matrix_adaptive(
                    3.5, [4.0], [1.0], **options)
    def test_scalar_0f1_large_arguments(self):
        c = 3.5
        for z in (1.0, 1000.0, 4000.0, 5000.0, 10_000.0):
            got, order, assessment, _ = alfd.mhg_two_matrix_adaptive(
                c, [4.0 * z], [1.0], M_start=20, M_max=300,
                tol=1e-10)
            expected = float(hyp0f1(c, z))
            relative_error = abs(got - expected) / expected
            self.assertLessEqual(relative_error, 1e-10, (z, order, got, expected))
            self.assertTrue(assessment.converged)

    def test_scalar_coefficients_match_closed_form(self):
        c = 3.5
        z = 10_000.0
        _, coef = alfd._mhg(
            180, 2.0, [], [c], [z], y=[1.0], want_coef=True)
        term = 1.0
        for degree in range(1, 151):
            term *= z / ((c + degree - 1.0) * degree)
            relative_error = abs(coef[degree] - term) / term
            self.assertLessEqual(relative_error, 2e-11, degree)

    def test_known_matrix_probes_converge_adaptively(self):
        probes = [
            ([35, 25, 15, 1], [70, 50, 30, 10], 75),
            ([100, 95, 90, 27], [132.9, 105.3, 60.4, 1], 150),
        ]
        for omega, sample, start in probes:
            value, order, assessment, _ = alfd.mhg_two_matrix_adaptive(
                3.5, omega, sample, M_start=start, M_max=300,
                tol=1e-10)
            self.assertTrue(np.isfinite(value))
            self.assertGreaterEqual(value, 1.0)
            self.assertTrue(assessment.converged)
            self.assertLessEqual(order, 300)

    def test_two_matrix_permutation_and_reciprocal_scaling(self):
        omega = np.array([35.0, 25.0, 15.0, 1.0])
        sample = np.array([70.0, 50.0, 30.0, 10.0])
        reference = alfd.mhg_two_matrix_adaptive(
            3.5, omega, sample, M_start=75, tol=1e-10)[0]
        permuted = alfd.mhg_two_matrix_adaptive(
            3.5, omega[[2, 0, 3, 1]], sample[[1, 3, 0, 2]],
            M_start=75, tol=1e-10)[0]
        scale = 7.0
        rescaled = alfd.mhg_two_matrix_adaptive(
            3.5, omega * scale, sample / scale,
            M_start=75, tol=1e-10)[0]
        self.assertLess(abs(permuted - reference) / reference, 2e-13)
        self.assertLess(abs(rescaled - reference) / reference, 2e-13)

        later_start = alfd.mhg_two_matrix_adaptive(
            3.5, omega, sample, M_start=110, tol=1e-10)[0]
        self.assertLess(abs(later_start - reference) / reference, 2e-13)


class EndToEndTests(unittest.TestCase):
    def test_scalar_emw_bound_matches_exact_np_power(self):
        result = alfd.alfd_eigval_bound(
            (0.2,), [()], 7, alpha=0.05,
            n_sim=500, n_sim_calibration=10_000,
            n_sim_validation=2_000, n_sim_power=30_000,
            n_iter=150, M_trunc=20, M_max=100, mhg_tol=1e-10,
            seed=42, verbose=False, n_workers=1,
            confidence_delta=0.05, return_result=True)
        exact = float(ncx2.sf(chi2.ppf(0.95, 7), 7, 0.2))
        self.assertAlmostEqual(result.point_rule.empirical_size, 0.05, places=13)
        self.assertLess(abs(result.upper_point - exact), 0.012)
        self.assertGreaterEqual(result.upper_confidence + 1e-12, exact)
        self.assertGreaterEqual(result.upper_confidence, 0.05)
        self.assertGreaterEqual(result.epsilon_grid_point, -1e-12)

    def test_only_exact_rank_deficiency_uses_alpha_shortcut(self):
        exact = alfd.alfd_eigval_bound(
            (0.0,), [()], 7, return_result=True,
            alternative_is_exact_null=True)
        self.assertEqual(exact.upper_point, 0.05)
        self.assertEqual(exact.point_rule.method, "exact_null_randomization")
        with self.assertRaises(ValueError):
            alfd.alfd_eigval_bound(
                (1e-6,), [()], 7, alternative_is_exact_null=True)

    def test_custom_validation_grid_is_augmented_with_fit_support(self):
        fit = [(5.0, 2.0, 1.0), (3.0, 2.0, 0.0)]
        custom = [(9.0, 4.0, 1.0)]
        validated = alfd._validated_null_grid(custom, 3, "test grid")
        union = alfd._union_null_grids(fit, validated)
        self.assertEqual(union[:2], fit)
        self.assertEqual(union[2:], custom)


class ArtifactContractTests(unittest.TestCase):
    def test_loader_accepts_only_new_confidence_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "bound.npz")
            source_hash = alfd._sha256_file(alfd.__file__)
            core_hash = alfd._sha256_file(
                os.path.join(alfd.MHG_DIR, "mhg_core.c"))
            library_hash = alfd._sha256_file(os.path.join(
                alfd.MHG_DIR,
                "libmhg.dylib" if sys.platform == "darwin" else "libmhg.so"))
            common_grid = [
                [0.0, 0.0, 0.0],
                [0.1, 0.05, 0.0],
                [100.0, 50.0, 0.0],
                [35.0, 25.0, 15.0],
            ]
            settings = dict(
                schema_version=3,
                algorithm="emw_eigval_gkm_is_adaptive_v3",
                producer="alfd_eigval.py",
                calibration_method="independent_mixture_quantile",
                version_label="352515",
                kappas=[35.0, 25.0, 15.0],
                k=7, n=250, alpha=0.05,
                curve_confidence=0.99, profile="production", beta_count=3,
                fit_grid_strategy="strength_shape_3d_v1",
                pooled_importance_method=(
                    "gkm_stratified_equal_null_mixture_v1"),
                confidence_allocation_method=(
                    "two_event_upper_separate_two_event_grid_v1"),
                common_grid=common_grid,
                grid_shapes=1, grid_strengths=2, grid_max_strength=100.0,
                grid_anchor_count=1,
                max_active_support=2,
                training_bank_seed=1234, audit_bank_seed=5678,
                source_sha256=source_hash,
                mhg_core_sha256=core_hash,
                mhg_library_sha256=library_hash,
                mhg_build_source_sha256=core_hash,
            )
            settings_json = json.dumps(settings, sort_keys=True)
            run_signature = hashlib.sha256(json.dumps(
                settings, sort_keys=True,
                separators=(',', ':')).encode()).hexdigest()
            payload = dict(
                schema_version=np.array(3),
                algorithm=np.array("emw_eigval_gkm_is_adaptive_v3"),
                producer=np.array("alfd_eigval.py"),
                calibration_method=np.array("independent_mixture_quantile"),
                confidence_allocation_method=np.array(
                    "two_event_upper_separate_two_event_grid_v1"),
                bound_kind=np.array(
                    "simultaneous_mc_confidence_upper_conditional_on_density_accuracy"),
                confidence_scope=np.array("saved_beta_grid_only"),
                density_accuracy_scope=np.array(
                    "adaptive_empirical_tail_criterion"),
                version_label=np.array("352515"),
                kappas=np.array([35.0, 25.0, 15.0]),
                k=np.array(7), n=np.array(250), alpha=np.array(0.05),
                betas=np.array([-0.2, 0.0, 0.2]),
                bounds=np.array([0.08, 0.05, 0.09]),
                bounds_se=np.zeros(3),
                common_null_grid=np.asarray(common_grid),
                common_grid_size=np.array(4),
                grid_shapes=np.array(1), grid_strengths=np.array(2),
                grid_max_strength=np.array(100.0),
                grid_anchor_count=np.array(1),
                max_active_support=np.array(2),
                active_support_count=np.array([2, 0, 2]),
                discarded_weight_mass=np.array([0.1, 0.0, 0.05]),
                gkm_point_upper=np.array([0.075, 0.05, 0.085]),
                gkm_grid_lower=np.array([0.07, 0.05, 0.08]),
                gkm_epsilon=np.array([0.005, 0.0, 0.005]),
                source_sha256=np.array(source_hash),
                mhg_core_sha256=np.array(core_hash),
                mhg_library_sha256=np.array(library_hash),
                mhg_build_source_sha256=np.array(core_hash),
                curve_confidence=np.array(0.99),
                settings_json=np.array(settings_json),
                run_signature=np.array(run_signature),
            )
            np.savez(path, **payload)
            betas, bounds, _ = load_compatible_alfd_bound(
                path, version_label="352515", kappas=[35, 25, 15],
                k=7, n=250, alpha=0.05)
            np.testing.assert_array_equal(betas, payload["betas"])
            np.testing.assert_array_equal(bounds, payload["bounds"])
            _, _, _, metadata = load_compatible_alfd_bound(
                path, version_label="352515", kappas=[35, 25, 15],
                k=7, n=250, alpha=0.05, return_metadata=True)
            self.assertEqual(metadata["curve_confidence"], 0.99)
            self.assertEqual(metadata["profile"], "production")
            self.assertEqual(metadata["common_grid_size"], 4)
            self.assertEqual(metadata["grid_anchor_count"], 1)
            self.assertEqual(metadata["max_active_support"], 2)
            self.assertEqual(
                metadata["pooled_importance_method"],
                "gkm_stratified_equal_null_mixture_v1")
            self.assertEqual(
                metadata["confidence_allocation_method"],
                "two_event_upper_separate_two_event_grid_v1")

            payload["run_signature"] = np.array("not-the-settings-signature")
            np.savez(path, **payload)
            with self.assertRaisesRegex(ValueError, "run_signature"):
                load_compatible_alfd_bound(
                    path, version_label="352515", kappas=[35, 25, 15],
                    k=7, n=250, alpha=0.05)
            payload["run_signature"] = np.array(run_signature)

            wrong_count_settings = {**settings, "beta_count": 4}
            payload["settings_json"] = np.array(json.dumps(
                wrong_count_settings, sort_keys=True))
            payload["run_signature"] = np.array(hashlib.sha256(json.dumps(
                wrong_count_settings, sort_keys=True,
                separators=(',', ':')).encode()).hexdigest())
            np.savez(path, **payload)
            with self.assertRaisesRegex(ValueError, "beta_count"):
                load_compatible_alfd_bound(
                    path, version_label="352515", kappas=[35, 25, 15],
                    k=7, n=250, alpha=0.05)
            payload["settings_json"] = np.array(settings_json)
            payload["run_signature"] = np.array(run_signature)

            wrong_method_settings = {
                **settings, "pooled_importance_method": "self_normalized_is"}
            payload["settings_json"] = np.array(json.dumps(
                wrong_method_settings, sort_keys=True))
            payload["run_signature"] = np.array(hashlib.sha256(json.dumps(
                wrong_method_settings, sort_keys=True,
                separators=(',', ':')).encode()).hexdigest())
            np.savez(path, **payload)
            with self.assertRaisesRegex(ValueError, "pooled_importance_method"):
                load_compatible_alfd_bound(
                    path, version_label="352515", kappas=[35, 25, 15],
                    k=7, n=250, alpha=0.05)
            payload["settings_json"] = np.array(settings_json)
            payload["run_signature"] = np.array(run_signature)

            wrong_anchor_settings = {**settings, "grid_anchor_count": 0}
            payload["settings_json"] = np.array(json.dumps(
                wrong_anchor_settings, sort_keys=True))
            payload["run_signature"] = np.array(hashlib.sha256(json.dumps(
                wrong_anchor_settings, sort_keys=True,
                separators=(',', ':')).encode()).hexdigest())
            np.savez(path, **payload)
            with self.assertRaisesRegex(ValueError, "grid_anchor_count"):
                load_compatible_alfd_bound(
                    path, version_label="352515", kappas=[35, 25, 15],
                    k=7, n=250, alpha=0.05)
            payload["settings_json"] = np.array(settings_json)
            payload["run_signature"] = np.array(run_signature)

            payload["common_null_grid"] = np.asarray(common_grid) + 1.0
            np.savez(path, **payload)
            with self.assertRaisesRegex(ValueError, "common_null_grid"):
                load_compatible_alfd_bound(
                    path, version_label="352515", kappas=[35, 25, 15],
                    k=7, n=250, alpha=0.05)
            payload["common_null_grid"] = np.asarray(common_grid)

            saved_active_count = payload.pop("active_support_count")
            np.savez(path, **payload)
            with self.assertRaisesRegex(ValueError, "active_support_count"):
                load_compatible_alfd_bound(
                    path, version_label="352515", kappas=[35, 25, 15],
                    k=7, n=250, alpha=0.05)
            payload["active_support_count"] = saved_active_count

            payload["schema_version"] = np.array(2)
            payload["algorithm"] = np.array("emw_eigval_adaptive_v2")
            np.savez(path, **payload)
            with self.assertRaisesRegex(ValueError, "schema_version"):
                load_compatible_alfd_bound(
                    path, version_label="352515", kappas=[35, 25, 15],
                    k=7, n=250, alpha=0.05)
            payload["schema_version"] = np.array(3)
            payload["algorithm"] = np.array(
                "emw_eigval_gkm_is_adaptive_v3")

            payload.pop("bound_kind")
            np.savez(path, **payload)
            with self.assertRaises(ValueError):
                load_compatible_alfd_bound(
                    path, version_label="352515", kappas=[35, 25, 15],
                    k=7, n=250, alpha=0.05)

    def test_dgp_cache_requires_full_settings_and_provenance(self):
        self.assertEqual(
            dgp_cache_path("352515"),
            os.path.join("352515", "dgp", "dgp_curves_352515.npz"))
        betas = np.array([-0.2, 0.0, 0.2])
        curves = (
            np.array([0.04, 0.05, 0.06]),
            np.array([0.05, 0.05, 0.07]),
            np.array([0.06, 0.05, 0.08]),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "dgp.npz")
            save_dgp_cache(
                path, version_label="352515", kappas=[35, 25, 15],
                k=7, n=250, alpha=0.05, betas=betas,
                power_chi2=curves[0], power_c1=curves[1],
                power_cp1=curves[2], num_simulations=1000,
                base_seed=123, chunk_size=100, workers_used=2)
            loaded = load_compatible_dgp_cache(
                path, version_label="352515", kappas=[35, 25, 15],
                k=7, n=250, alpha=0.05, betas=betas,
                num_simulations=1000, base_seed=123, chunk_size=100)
            for actual, expected in zip(loaded, (betas,) + curves):
                np.testing.assert_array_equal(actual, expected)

            with self.assertRaisesRegex(ValueError, "incompatible DGP cache"):
                load_compatible_dgp_cache(
                    path, version_label="352515", kappas=[35, 25, 15],
                    k=7, n=250, alpha=0.05, betas=betas,
                    num_simulations=1000, base_seed=124, chunk_size=100)

            # Legacy caches have arrays but no schema/provenance contract.
            np.savez(
                path, betas=betas, power_chi2=curves[0],
                power_c1=curves[1], power_cp1=curves[2])
            with self.assertRaisesRegex(ValueError, "schema_version"):
                load_compatible_dgp_cache(
                    path, version_label="352515", kappas=[35, 25, 15],
                    k=7, n=250, alpha=0.05, betas=betas,
                    num_simulations=1000, base_seed=123, chunk_size=100)


if __name__ == "__main__":
    unittest.main()
