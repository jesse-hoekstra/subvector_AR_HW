import contextlib
import fcntl
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

import alfd_eigval as alfd
import refine_power_curve as refine
import watch_power_progress as watcher


VERSION = "352515"
MIDPOINTS = np.array([-1.75, -1.25, -0.75, -0.25, 0.25, 0.75, 1.25, 1.75])


def _density_fixture(samples, omegas, *args, **kwargs):
    pairs = len(samples) * len(omegas)
    return np.zeros((len(omegas), len(samples))), dict(
        pairs=pairs, raw_evaluations=0, order_counts={0: pairs},
        max_order=0, max_remainder_ratio=0.0)


def _result_fixture(*, kappas_alt, **kwargs):
    # Distinguishable, valid rows make accidental replacement of the saved
    # points visible; the numerical estimator itself is tested elsewhere.
    bound = 0.2 + 0.1 * float(kappas_alt[-1]) / (1.0 + float(kappas_alt[-1]))
    weights = np.array([0.4, 0.6])
    rule = alfd.TailRule(0.0, 0.5, 0.05, "refinement-test")
    return alfd.GKMDirectResult(
        bound=bound, bound_se=0.01,
        mixture_power=bound + 0.05, mixture_power_se=0.02,
        epsilon_grid=0.05, weights=weights, log_weights=np.log(weights),
        fit_rejection_probabilities=np.array([0.04, 0.05]),
        grid_rejection_probabilities=np.array([0.03, 0.04]),
        fit_iterations=1, mixture_rule=rule, grid_rule=rule,
        importance_diagnostics={"fixture": True},
        mhg_diagnostics=dict(
            pairs=4, raw_evaluations=4, order_counts={20: 4},
            max_order=20, max_remainder_ratio=0.0))


class RefinementTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        directory = Path(self.temporary.name)
        original_directory = Path.cwd()
        argv = [
            "alfd_eigval.py", "--version", VERSION,
            "--profile", "production", "--acknowledge-expensive",
            "--n-fit", "2", "--n-power", "2", "--n-iter", "1",
            "--grid-shapes", "1", "--grid-strengths", "1",
            "--beta-count", "9", "--workers", "1",
        ]
        try:
            os.chdir(directory)
            with mock.patch.object(sys, "argv", argv), \
                    mock.patch.object(sys, "stdout", io.StringIO()), \
                    mock.patch.object(sys, "stderr", io.StringIO()), \
                    mock.patch.object(alfd, "verify_mhg"), \
                    mock.patch.object(alfd, "common_null_grid_3d", return_value=[
                        (0.0, 0.0, 0.0), (1.0, 0.5, 0.0)]), \
                    mock.patch.object(alfd, "log_eigval_density_partial",
                                      side_effect=_density_fixture), \
                    mock.patch.object(alfd, "gkm_eigval_bound_from_pooled_bank",
                                      side_effect=_result_fixture):
                try:
                    alfd.main()
                finally:
                    if isinstance(sys.stdout, alfd._Tee):
                        sys.stdout._streams[-1].close()
        finally:
            os.chdir(original_directory)

        self.direct = directory / VERSION / "gkm_direct"
        self.source_path = self.direct / f"gkm_eigval_{VERSION}.npz"
        self.source = refine._read_npz(self.source_path)
        self.settings = json.loads(str(self.source["settings_json"].item()))
        bank_id = str(self.source["bank_id"].item())
        self.bank_path = self.direct / f"pooled_gkm_{bank_id[:16]}.npz"
        self.output = self.direct / "refined"
        self.final = self.output / f"gkm_eigval_{VERSION}.npz"
        self.partial = self.output / f"gkm_eigval_{VERSION}.partial.npz"

    @contextlib.contextmanager
    def _forbid_bank_work(self):
        with mock.patch.object(
                alfd, "build_or_load_pooled_is_bank",
                side_effect=AssertionError("refinement must never build a bank")) as build, \
                mock.patch.object(
                    alfd, "log_eigval_density_partial",
                    side_effect=AssertionError("unexpected numerical density work")) as density, \
                contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            yield
            build.assert_not_called()
            density.assert_not_called()

    def _run(self, *extra, side_effect=_result_fixture):
        with self._forbid_bank_work(), mock.patch.object(
                alfd, "gkm_eigval_bound_from_pooled_bank",
                side_effect=side_effect) as calculate:
            refine.main([
                "--source", str(self.source_path), "--workers", "1",
                "--acknowledge-expensive", *extra])
            return calculate

    def _assert_preserved_rows(self, payload):
        old_indices = np.searchsorted(payload["betas"], self.source["betas"])
        for key in refine.POINT_FIELDS:
            with self.subTest(field=key):
                np.testing.assert_array_equal(payload[key][old_indices], self.source[key])
                self.assertEqual(payload[key].dtype, self.source[key].dtype)

    def test_eight_midpoints_preserve_old_rows_files_bank_and_watcher(self):
        original_bytes = self.source_path.read_bytes()
        bank_bytes = self.bank_path.read_bytes()
        calculate = self._run()
        self.assertEqual(calculate.call_count, 8)
        for beta, call in zip(MIDPOINTS, calculate.call_args_list):
            expected_ncp = np.maximum(alfd.asymptotic_ncp_eigenvalues(
                beta, self.source["kappas"], self.settings["k"], self.settings["n"]), 0.0)
            np.testing.assert_array_equal(call.kwargs["kappas_alt"], expected_ncp)
            self.assertEqual(call.kwargs["n_sim_power"], 2)
            self.assertEqual(call.kwargs["n_iter"], 1)
            self.assertEqual(call.kwargs["bank"].bank_id, self.source["bank_id"].item())

        self.assertEqual(self.source_path.read_bytes(), original_bytes)
        self.assertEqual(self.bank_path.read_bytes(), bank_bytes)
        self.assertFalse(self.partial.exists())
        saved = refine._read_npz(self.final)
        np.testing.assert_array_equal(saved["betas"], np.linspace(-2.0, 2.0, 17))
        self._assert_preserved_rows(saved)
        self.assertTrue(np.all(np.isfinite(saved["bounds"])))
        progress = watcher.load_bound_progress(
            VERSION, partial_path=str(self.partial), final_path=str(self.final))
        self.assertTrue(progress.is_final)
        self.assertEqual(len(progress.betas), 17)
        self.assertNotEqual(progress.run_signature, self.source["run_signature"].item())
        self.assertTrue((self.output / f"gkm_bounds_{VERSION}.csv").is_file())

        reserved = {self.settings["bank_seed"]}
        reserved.update(json.loads(str(row))["power_seed"]
                        for row in self.source["diagnostics_json"])
        new_seeds = [call.kwargs["seed"] for call in calculate.call_args_list]
        self.assertEqual(len(set(new_seeds)), 8)
        self.assertTrue(set(new_seeds).isdisjoint(reserved))
        rerun = self._run()
        rerun.assert_not_called()

    def test_interruption_resumes_only_seven_remaining_midpoints(self):
        seen = []

        def interrupt_second(**kwargs):
            seen.append(kwargs)
            if len(seen) == 2:
                raise KeyboardInterrupt("simulated interruption")
            return _result_fixture(**kwargs)

        with self.assertRaises(KeyboardInterrupt):
            self._run(side_effect=interrupt_second)
        self.assertFalse(self.final.exists())
        checkpoint = refine._read_npz(self.partial)
        self._assert_preserved_rows(checkpoint)
        self.assertEqual(np.count_nonzero(np.isfinite(checkpoint["bounds"])), 10)
        progress = watcher.load_bound_progress(
            VERSION, partial_path=str(self.partial), final_path=str(self.final))
        self.assertFalse(progress.is_final)
        self.assertEqual(np.count_nonzero(np.isfinite(progress.bounds)), 10)

        resumed = self._run()
        self.assertEqual(resumed.call_count, 7)
        self.assertEqual(resumed.call_args_list[0].kwargs["seed"], seen[1]["seed"])
        saved = refine._read_npz(self.final)
        self._assert_preserved_rows(saved)
        first_index = int(np.searchsorted(saved["betas"], MIDPOINTS[0]))
        for key in refine.POINT_FIELDS:
            np.testing.assert_array_equal(saved[key][first_index], checkpoint[key][first_index])

    def test_missing_bank_fails_before_calculation_or_output(self):
        self.bank_path.unlink()
        with mock.patch.object(alfd, "gkm_eigval_bound_from_pooled_bank") as calculate, \
                self._forbid_bank_work(), \
                self.assertRaisesRegex(FileNotFoundError, "null bank is missing"):
            refine.main(["--source", str(self.source_path), "--acknowledge-expensive"])
        calculate.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_corrupt_bank_fails_before_calculation_or_output(self):
        bank = refine._read_npz(self.bank_path)
        bank["eigs"][0] += 0.01  # Valid eigenvalues, invalid content signature.
        np.savez(self.bank_path, **bank)
        with mock.patch.object(alfd, "gkm_eigval_bound_from_pooled_bank") as calculate, \
                self._forbid_bank_work(), \
                self.assertRaisesRegex(ValueError, "content signature differs"):
            refine.main(["--source", str(self.source_path), "--acknowledge-expensive"])
        calculate.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_preflight_validates_cache_and_creates_no_outputs(self):
        before = {path: path.read_bytes() for path in self.direct.iterdir() if path.is_file()}
        with mock.patch.object(alfd, "gkm_eigval_bound_from_pooled_bank") as calculate, \
                self._forbid_bank_work():
            refine.main(["--source", str(self.source_path), "--preflight-only"])
        calculate.assert_not_called()
        self.assertFalse(self.output.exists())
        after = {path: path.read_bytes() for path in self.direct.iterdir() if path.is_file()}
        self.assertEqual(after, before)

    def test_resume_rejects_changed_original_row(self):
        with self.assertRaises(KeyboardInterrupt):
            self._run(side_effect=KeyboardInterrupt("stop before first point"))
        checkpoint = refine._read_npz(self.partial)
        checkpoint["mixture_power"][0] += 0.01
        np.savez(self.partial, **checkpoint)
        with self.assertRaisesRegex(ValueError, "changed an original mixture_power row"):
            self._run()
        self.assertFalse(self.final.exists())

    def test_another_writer_lock_prevents_any_beta_calculation(self):
        self.output.mkdir()
        with open(self.output / ".refinement.lock", "a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaisesRegex(RuntimeError, "another refinement is writing"):
                self._run(side_effect=AssertionError("must not calculate while locked"))
        self.assertFalse(self.partial.exists())
        self.assertFalse(self.final.exists())

    def test_valid_zero_weight_log_minus_infinity_is_preserved(self):
        self.source["fitted_weights"][0] = [0.0, 1.0]
        self.source["fitted_log_weights"][0] = [-np.inf, 0.0]
        np.savez(self.source_path, **self.source)
        self._run()
        self._assert_preserved_rows(refine._read_npz(self.final))
        rerun = self._run()
        rerun.assert_not_called()


if __name__ == "__main__":
    unittest.main()
