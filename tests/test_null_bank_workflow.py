import contextlib
import io
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

import alfd_eigval as alfd


@contextlib.contextmanager
def temporary_working_directory():
    original = os.getcwd()
    with tempfile.TemporaryDirectory() as directory:
        os.chdir(directory)
        try:
            yield Path(directory)
        finally:
            os.chdir(original)


class NullBankWorkflowTests(unittest.TestCase):
    @staticmethod
    def _bank():
        return alfd.PooledISBank(
            grid=np.array([[1.0, 0.2], [0.0, 0.0]]),
            eigs=np.zeros((6, 3)), log_f=np.zeros((2, 6)),
            log_q=np.zeros(6), base_weights=np.full(6, 1.0 / 6.0),
            strata=np.repeat(np.arange(2), 3), n_per_stratum=3,
            role="gkm", bank_id="test-mw2-null-bank")

    def test_mw2_production_grid_is_deterministic_and_contains_target(self):
        kappas = [100.0, 15.0]
        alternatives = np.asarray([
            alfd.asymptotic_ncp_eigenvalues(beta, kappas, 7, 250)[:-1]
            for beta in np.linspace(-2.0, 2.0, alfd.GRID_DESIGN_BETA_COUNT)
        ])
        options = dict(
            standard_points=alfd.ALLOWED_CONFIGS[(100, 15)]["standard"],
            n_strengths=7, max_strength=100.0)
        first = np.asarray(alfd.common_null_grid_2d(
            alternatives, kappas, **options))
        second = np.asarray(alfd.common_null_grid_2d(
            alternatives, kappas, **options))

        self.assertEqual(first.shape, (36, 2))
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(first[0], np.zeros(2))
        self.assertTrue(np.all(np.isfinite(first)))
        self.assertTrue(np.all(first >= 0.0))
        self.assertTrue(np.all(np.diff(first, axis=1) <= 1e-12))
        self.assertTrue(any(np.allclose(row, kappas) for row in first))

    def test_null_only_preflight_counts_only_the_null_bank_and_writes_nothing(self):
        for version, profile, extra_args, grid_size, pairs in (
                ("10015", "production", [], 36, 2_592_000),
                ("10015", "reference", [], 36, 12_960_000),
                ("10015", "production", ["--grid-shapes", "9"], 64, 8_192_000),
                ("352515", "production", [], 68, 9_248_000)):
            with self.subTest(version=version, profile=profile,
                              extra_args=extra_args):
                argv = [
                    "alfd_eigval.py", "--version", version, "--null-bank-only",
                    "--profile", profile, "--preflight-only", "--workers", "2",
                    *extra_args,
                ]
                output = io.StringIO()
                with mock.patch.object(sys, "argv", argv), \
                        mock.patch.object(sys, "stdout", output), \
                        mock.patch.object(
                            alfd, "simulate_Xi", side_effect=AssertionError(
                                "preflight simulated observations")), \
                        mock.patch.object(
                            alfd, "build_or_load_pooled_is_bank",
                            side_effect=AssertionError("preflight requested a bank")), \
                        mock.patch.object(
                            alfd.os, "makedirs", side_effect=AssertionError(
                                "preflight created an artifact directory")), \
                        mock.patch.object(
                            alfd, "_atomic_savez", side_effect=AssertionError(
                                "preflight wrote an artifact")):
                    alfd.main()

                self.assertIn(f"common null grid H={grid_size};", output.getvalue())
                self.assertIn(
                    f"logical density pairs: {pairs:,} over 0 non-null betas",
                    output.getvalue())

    def test_null_only_forwards_workers_and_ignores_power_artifacts(self):
        argv = [
            "alfd_eigval.py", "--version", "10015", "--null-bank-only",
            "--n-fit", "3", "--workers", "3", "--acknowledge-expensive",
        ]
        with temporary_working_directory() as directory:
            out_dir = directory / "10015" / "gkm_direct"
            out_dir.mkdir(parents=True)
            power_paths = [out_dir / "gkm_eigval_10015.npz",
                           out_dir / "gkm_eigval_10015.partial.npz"]
            for path in power_paths:
                path.write_bytes(b"unrelated existing power artifact")
            with mock.patch.object(sys, "argv", argv), \
                    mock.patch.object(sys, "stdout", io.StringIO()), \
                    mock.patch.object(sys, "stderr", io.StringIO()), \
                    mock.patch.object(alfd, "verify_mhg"), \
                    mock.patch.object(
                        alfd, "build_or_load_pooled_is_bank",
                        return_value=self._bank()) as build_bank, \
                    mock.patch.object(
                        alfd, "gkm_eigval_bound_from_pooled_bank",
                        side_effect=AssertionError("null-only evaluated power")), \
                    mock.patch.object(
                        alfd, "_atomic_savez", side_effect=AssertionError(
                            "null-only wrote a power artifact")):
                alfd.main()

            build_bank.assert_called_once()
            args, kwargs = build_bank.call_args
            self.assertEqual(np.asarray(args[0]).shape, (36, 2))
            self.assertEqual(args[1:3], (7, 3))
            self.assertEqual(kwargs["n_workers"], 3)
            self.assertEqual(kwargs["role"], "gkm")
            self.assertEqual(Path(kwargs["cache_dir"]).resolve(), out_dir.resolve())
            for path in power_paths:
                self.assertEqual(path.read_bytes(),
                                 b"unrelated existing power artifact")

    def test_bank_identity_is_independent_of_curve_controls_and_run_mode(self):
        class BankReached(Exception):
            pass

        calls = []

        def capture_bank(*args, **kwargs):
            calls.append((args, kwargs))
            raise BankReached

        def open_without_power_log(path, *args, **kwargs):
            # This test deliberately interrupts the full CLI before it finishes.
            if Path(path).name == "bound_run.log":
                return io.StringIO()
            return open(path, *args, **kwargs)

        with temporary_working_directory():
            for mode, workers, beta_count, n_power, n_iter in (
                    (["--null-bank-only"], 3, 5, 8, 2),
                    ([], 1, 9, 12, 4)):
                argv = [
                    "alfd_eigval.py", "--version", "10015",
                    "--n-fit", "3", "--seed", "123",
                    "--workers", str(workers), "--beta-count", str(beta_count),
                    "--n-power", str(n_power), "--n-iter", str(n_iter),
                    "--acknowledge-expensive", *mode,
                ]
                with mock.patch.object(sys, "argv", argv), \
                        mock.patch.object(sys, "stdout", io.StringIO()), \
                        mock.patch.object(sys, "stderr", io.StringIO()), \
                        mock.patch.object(alfd, "verify_mhg"), \
                        mock.patch.object(alfd, "_atomic_savez"), \
                        mock.patch.object(
                            alfd, "open", side_effect=open_without_power_log,
                            create=True), \
                        mock.patch.object(
                            alfd, "build_or_load_pooled_is_bank",
                            side_effect=capture_bank), \
                        self.assertRaises(BankReached):
                    alfd.main()

        self.assertEqual(len(calls), 2)
        first_args, first_kwargs = calls[0]
        second_args, second_kwargs = calls[1]
        np.testing.assert_array_equal(first_args[0], second_args[0])
        self.assertEqual(first_args[1:], second_args[1:])
        self.assertEqual(first_kwargs.pop("n_workers"), 3)
        self.assertEqual(second_kwargs.pop("n_workers"), 1)
        self.assertEqual(first_kwargs, second_kwargs)


class ParallelNullBankTests(unittest.TestCase):
    def test_parallel_chunks_stay_bounded_and_preserve_sample_order(self):
        samples = np.column_stack((np.arange(1000), np.zeros((1000, 2))))
        chunk_lengths = []

        def completed_in_reverse_order(worker, tasks):
            for task in reversed(tasks):
                chunk_id, _, omegas, sample_chunk, *_ = task
                chunk_lengths.append(len(sample_chunk))
                values = np.repeat(sample_chunk[:, :1].T, len(omegas), axis=0)
                pairs = values.size
                diagnostics = dict(
                    pairs=pairs, raw_evaluations=pairs,
                    order_counts={20: pairs}, max_order=20,
                    max_remainder_ratio=0.0)
                yield chunk_id, values, diagnostics

        with mock.patch("multiprocessing.Pool") as pool_factory, \
                mock.patch("builtins.print"):
            pool = pool_factory.return_value.__enter__.return_value
            pool.imap_unordered.side_effect = completed_in_reverse_order
            values, diagnostics = alfd.chunked_mhg_batch(
                3.5, [[1.0, 0.2, 0.0]], samples, chunk_size=100,
                n_workers=2, return_diagnostics=True)

        pool_factory.assert_called_once_with(processes=2)
        self.assertEqual(sum(chunk_lengths), len(samples))
        self.assertLessEqual(max(chunk_lengths), 100)
        np.testing.assert_array_equal(values[0], samples[:, 0])
        self.assertEqual(diagnostics["pairs"], len(samples))

    def test_real_mw2_bank_is_identical_across_worker_counts_and_reloads(self):
        grid = [(1.0, 0.2), (0.0, 0.0)]
        settings = dict(
            k_eff=7, n_per_stratum=3, seed=12345,
            M_start=10, M_step=10, M_max=80, mhg_tol=1e-8)
        with tempfile.TemporaryDirectory() as directory:
            serial = alfd.build_or_load_pooled_is_bank(
                grid, n_workers=1, cache_dir=os.path.join(directory, "serial"),
                **settings)
            parallel_dir = os.path.join(directory, "parallel")
            parallel = alfd.build_or_load_pooled_is_bank(
                grid, n_workers=2, cache_dir=parallel_dir, **settings)
            with mock.patch.object(
                    alfd, "simulate_Xi", side_effect=AssertionError(
                        "a different worker count invalidated the cache")), \
                    mock.patch.object(
                        alfd, "log_eigval_density_partial",
                        side_effect=AssertionError("cached densities recomputed")):
                reloaded = alfd.build_or_load_pooled_is_bank(
                    grid, n_workers=3, cache_dir=parallel_dir, **settings)

        self.assertEqual(serial.eigs.shape, (6, 3))
        self.assertEqual(serial.log_f.shape, (2, 6))
        self.assertTrue(np.all(np.isfinite(serial.log_f)))
        self.assertEqual(serial.mhg_diagnostics["pairs"], 12)
        for compared in (parallel, reloaded):
            self.assertEqual(serial.bank_id, compared.bank_id)
            self.assertEqual(serial.content_signature, compared.content_signature)
            self.assertEqual(serial.settings_json, compared.settings_json)
            self.assertEqual(alfd._json_safe(serial.mhg_diagnostics),
                             alfd._json_safe(compared.mhg_diagnostics))
            for name in ("grid", "eigs", "log_f", "log_q", "base_weights", "strata"):
                np.testing.assert_array_equal(
                    getattr(serial, name), getattr(compared, name))


if __name__ == "__main__":
    unittest.main()
