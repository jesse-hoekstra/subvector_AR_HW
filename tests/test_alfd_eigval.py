import os
import sys
import tempfile
import unittest
import io
import json
from unittest import mock

import numpy as np
from scipy.special import hyp0f1

import alfd_eigval as alfd
from new_power_comparison import (
    dgp_cache_path,
    load_compatible_dgp_cache,
    save_dgp_cache,
)


class CalibrationTests(unittest.TestCase):
    def test_raw_weighted_tail_randomizes_ties_to_exact_size(self):
        scores = np.array([4.0, 3.0, 3.0, 2.0, 1.0])
        weights = np.array([0.1, 0.2, 0.3, 0.1, 0.3])
        rule = alfd.calibrate_raw_weighted_tail(scores, weights, alpha=0.35)
        attained = np.dot(
            weights, alfd.tail_rejection_probabilities(scores, rule))
        self.assertAlmostEqual(attained, 0.35, places=14)
        self.assertEqual(rule.threshold, 3.0)
        self.assertAlmostEqual(rule.tie_probability, 0.5)

    def test_raw_weighted_tail_rejects_nonfinite_inputs(self):
        with self.assertRaises(ValueError):
            alfd.calibrate_raw_weighted_tail(
                [0.0, np.nan], [0.5, 0.5], alpha=0.05)

    def test_reference_and_production_pair_totals_are_exact(self):
        production = alfd._gkm_budget_diagnostics(
            0.05, common_grid_size=68, n_nonnull=8,
            budget=dict(n_fit=2_000, n_power=50_000, n_iter=600))
        reference = alfd._gkm_budget_diagnostics(
            0.05, common_grid_size=68, n_nonnull=8,
            budget=dict(n_fit=10_000, n_power=100_000, n_iter=600))

        self.assertEqual(production["phase_pairs"], {
            "shared_null_bank": 9_248_000,
            "beta_training_alternative": 1_088_000,
            "beta_power": 27_600_000,
        })
        self.assertEqual(production["total_pairs"], 37_936_000)
        self.assertEqual(reference["phase_pairs"], {
            "shared_null_bank": 46_240_000,
            "beta_training_alternative": 5_440_000,
            "beta_power": 55_200_000,
        })
        self.assertEqual(reference["total_pairs"], 106_880_000)


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
            role="gkm", bank_id="synthetic-bank")

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

    def test_default_cli_beta_grid_has_nine_symmetric_points_and_zero(self):
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
                mock.patch.object(
                    alfd, '_atomic_savez',
                    side_effect=AssertionError('preflight wrote an artifact')), \
                mock.patch('builtins.print'):
            alfd.main()

        expected = np.linspace(-2.0, 2.0, 9)
        design = np.linspace(-2.0, 2.0, 81)
        np.testing.assert_allclose(
            observed[:9], expected, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(
            observed[9:], design, rtol=0.0, atol=0.0)
        self.assertEqual(observed[4], 0.0)
        np.testing.assert_allclose(
            np.asarray(observed[:9]), -np.asarray(observed[:9])[::-1],
            rtol=0.0, atol=1e-15)
        self.assertEqual(len(grids), 1)
        self.assertEqual(len(grids[0][0]), 68)
        self.assertEqual(grids[0][1]['n_shapes'], 9)
        self.assertEqual(len(grids[0][1]['standard_points']), 4)

    def test_beta_count_does_not_change_common_grid(self):
        real_grid = alfd.common_null_grid_3d
        constructed = []

        def capture_grid(*args, **kwargs):
            result = real_grid(*args, **kwargs)
            constructed.append(np.asarray(result))
            return result

        for count in (5, 9):
            argv = [
                'alfd_eigval.py', '--version', '352515',
                '--profile', 'production', '--preflight-only',
                '--beta-count', str(count), '--workers', '1',
            ]
            with mock.patch.object(sys, 'argv', argv), \
                    mock.patch.object(
                        alfd, 'common_null_grid_3d', side_effect=capture_grid), \
                    mock.patch.object(
                        alfd, '_atomic_savez',
                        side_effect=AssertionError('preflight wrote an artifact')), \
                    mock.patch('builtins.print'):
                alfd.main()

        self.assertEqual(len(constructed), 2)
        np.testing.assert_array_equal(constructed[0], constructed[1])

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

        # Normalizing to unit realized mass instead changes the tie
        # randomization and is not GKM equation (D.1).
        normalized = alfd.calibrate_raw_weighted_tail(
            scores, raw / raw.sum(), alpha=0.5)
        self.assertEqual(normalized.threshold, 3.0)
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
                cache_dir=directory, role='gkm')
            second = alfd.build_or_load_pooled_is_bank(
                grid, k_eff=7, n_per_stratum=n_per, seed=123,
                cache_dir=directory, role='gkm')

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

    def test_direct_path_rejects_non_gkm_bank_role(self):
        bank = self._synthetic_pooled_bank()
        bank.role = 'audit'
        with self.assertRaisesRegex(ValueError, 'invalid pooled'):
            alfd.gkm_importance_rejection_probabilities(
                bank, np.ones(bank.eigs.shape[0]))


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
            '--benchmark-samples', '1', '--beta-count', '3',
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
        self.assertEqual(len(run_benchmark.call_args.kwargs['fit_grids']), 3)
        print_result.assert_called_once_with(benchmark_result)

    def test_cli_rejects_removed_confidence_audit_and_compression_flags(self):
        legacy_flags = (
            '--curve-confidence', '--n-calibration', '--n-validation',
            '--max-active-support', '--audit-seed')
        for flag in legacy_flags:
            argv = [
                'alfd_eigval.py', '--version', '352515',
                '--preflight-only', flag, '2',
            ]
            with self.subTest(flag=flag), \
                    mock.patch.object(sys, 'argv', argv), \
                    mock.patch.object(sys, 'stderr', io.StringIO()), \
                    self.assertRaises(SystemExit) as raised:
                alfd.main()
            self.assertEqual(raised.exception.code, 2)

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
    def test_exact_rank_deficiency_result_is_alpha_on_full_grid(self):
        exact = alfd._exact_gkm_result(alpha=0.05, G=68)
        self.assertEqual(exact.bound, 0.05)
        self.assertEqual(exact.mixture_power, 0.05)
        self.assertEqual(exact.epsilon_grid, 0.0)
        self.assertEqual(exact.mixture_rule.method,
                         "exact_null_randomization")
        self.assertEqual(exact.weights.shape, (68,))
        self.assertEqual(exact.log_weights.shape, (68,))
        self.assertEqual(exact.grid_rejection_probabilities.shape, (68,))
        np.testing.assert_allclose(exact.weights, 1.0 / 68.0)
        np.testing.assert_allclose(exact.log_weights, -np.log(68.0))
        np.testing.assert_allclose(exact.grid_rejection_probabilities, 0.05)

    def test_null_grid_validation_preserves_full_ordered_grid(self):
        grid = [(5.0, 2.0, 1.0), (3.0, 2.0, 0.0)]
        self.assertEqual(
            alfd._validated_null_grid(grid, 3, "test grid"), grid)


class ArtifactContractTests(unittest.TestCase):
    def test_direct_v4_checkpoint_and_artifact_keys(self):
        grid = [(0.0, 0.0, 0.0), (1.0, 0.5, 0.0)]
        fake_bank = mock.Mock(
            bank_id="bank-id",
            content_signature="a" * 64,
            mhg_diagnostics={
                "pairs": 8, "raw_evaluations": 8,
                "order_counts": {20: 8}, "max_order": 20,
                "max_remainder_ratio": 0.0,
            })
        rule = alfd.TailRule(0.0, 0.0, 0.05, "direct-gkm-test")
        direct_result = alfd.GKMDirectResult(
            bound=0.20, bound_se=0.01,
            mixture_power=0.25, mixture_power_se=0.02,
            epsilon_grid=0.05,
            weights=np.array([0.4, 0.6]),
            log_weights=np.log(np.array([0.4, 0.6])),
            fit_rejection_probabilities=np.array([0.04, 0.05]),
            grid_rejection_probabilities=np.array([0.05, 0.049]),
            fit_iterations=1, mixture_rule=rule, grid_rule=rule,
            importance_diagnostics={},
            mhg_diagnostics={
                "pairs": 4, "raw_evaluations": 4,
                "order_counts": {20: 4}, "max_order": 20,
                "max_remainder_ratio": 0.0,
            })
        writes = []
        real_atomic_savez = alfd._atomic_savez

        def record_save(path, **arrays):
            writes.append((path, set(arrays)))
            real_atomic_savez(path, **arrays)

        argv = [
            "alfd_eigval.py", "--version", "352515",
            "--profile", "production", "--acknowledge-expensive",
            "--n-fit", "2", "--n-power", "2", "--n-iter", "1",
            "--grid-shapes", "1", "--grid-strengths", "1",
            "--beta-count", "3", "--workers", "1",
        ]
        previous_directory = os.getcwd()
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                with mock.patch.object(sys, "argv", argv), \
                        mock.patch.object(sys, "stdout", io.StringIO()), \
                        mock.patch.object(sys, "stderr", io.StringIO()), \
                        mock.patch.object(alfd, "verify_mhg"), \
                        mock.patch.object(
                            alfd, "asymptotic_ncp_eigenvalues",
                            side_effect=lambda beta, *args: np.array(
                                [0.35, 0.25, 0.15, 0.01 * abs(beta)])), \
                        mock.patch.object(
                            alfd, "common_null_grid_3d", return_value=grid), \
                        mock.patch.object(
                            alfd, "build_or_load_pooled_is_bank",
                            return_value=fake_bank), \
                        mock.patch.object(
                            alfd, "gkm_eigval_bound_from_pooled_bank",
                            return_value=direct_result) as direct_call, \
                        mock.patch.object(
                            alfd, "_atomic_savez", side_effect=record_save):
                    alfd.main()
                    sys.stdout._streams[-1].close()
            finally:
                os.chdir(previous_directory)

            artifact = os.path.join(
                directory, "352515", "gkm_direct", "gkm_eigval_352515.npz")
            partial = os.path.join(
                directory, "352515", "gkm_direct",
                "gkm_eigval_352515.partial.npz")
            self.assertTrue(os.path.isfile(artifact))
            self.assertFalse(os.path.exists(partial))
            self.assertEqual(direct_call.call_count, 2)

            checkpoint_required = {
                "schema_version", "algorithm", "producer",
                "calibration_method", "bound_kind", "version_label",
                "run_signature", "settings_json", "kappas", "k", "n",
                "alpha", "betas", "ncp", "bounds", "bounds_se",
                "mixture_power", "mixture_power_se", "epsilon_grid",
                "fitted_weights", "fitted_log_weights",
                "fit_rejection_probabilities",
                "grid_rejection_probabilities", "fit_iterations",
                "max_m_used", "diagnostics_json",
            }
            partial_writes = [keys for path, keys in writes
                              if path.endswith(".partial.npz")]
            self.assertTrue(partial_writes)
            for keys in partial_writes:
                self.assertEqual(keys, checkpoint_required)

            removed = {
                "bounds_point", "bounds_confidence", "curve_confidence",
                "confidence_scope", "confidence_allocation_method",
                "n_calibration", "n_validation", "training_bank_id",
                "audit_bank_id", "active_support_count",
                "discarded_weight_mass", "retained_weights",
                "retained_indices", "gkm_point_upper", "gkm_grid_lower",
            }
            with np.load(artifact, allow_pickle=False) as archive:
                keys = set(archive.files)
                self.assertTrue(checkpoint_required.issubset(keys))
                self.assertTrue(removed.isdisjoint(keys))
                self.assertEqual(int(archive["schema_version"]), 4)
                self.assertEqual(str(archive["algorithm"]),
                                 "gkm_eigval_mw3_adaptive_v4")
                self.assertEqual(str(archive["calibration_method"]),
                                 "gkm_step6_reused_pooled_bank")
                self.assertEqual(str(archive["bound_kind"]),
                                 "gkm_d3_2_grid_adjusted_mc_power_bound")
                np.testing.assert_allclose(
                    archive["mixture_power"] - archive["bounds"],
                    archive["epsilon_grid"], atol=1e-15)
                self.assertEqual(archive["fitted_weights"].shape, (3, 2))
                self.assertEqual(
                    archive["fitted_log_weights"].shape, (3, 2))
                self.assertEqual(
                    archive["grid_rejection_probabilities"].shape, (3, 2))
                settings = json.loads(str(archive["settings_json"]))
                self.assertEqual(settings["gkm_initial_mu"], -2.0)
                self.assertEqual(settings["gkm_step_size"], 2.0)
                self.assertTrue(removed.isdisjoint(settings))

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
            with self.assertRaisesRegex(
                    ValueError, "schema_version|metadata key"):
                load_compatible_dgp_cache(
                    path, version_label="352515", kappas=[35, 25, 15],
                    k=7, n=250, alpha=0.05, betas=betas,
                    num_simulations=1000, base_seed=123, chunk_size=100)


if __name__ == "__main__":
    unittest.main()
