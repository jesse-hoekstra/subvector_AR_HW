import contextlib
import csv
import fcntl
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

import numpy as np

import alfd_eigval as alfd
import power_bound_cluster as cluster


RESULT_FIELDS = {
    "bounds": "bound",
    "bounds_se": "bound_se",
    "mixture_power": "mixture_power",
    "mixture_power_se": "mixture_power_se",
    "epsilon_grid": "epsilon_grid",
    "fitted_weights": "weights",
    "fitted_log_weights": "log_weights",
    "fit_rejection_probabilities": "fit_rejection_probabilities",
    "grid_rejection_probabilities": "grid_rejection_probabilities",
    "fit_iterations": "fit_iterations",
}


def read_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name].copy() for name in archive.files}


def result_fixture(*, kappas_alt, bank, seed, n_iter, n_sim_power, alpha, **kwargs):
    """Distinct valid rows, independent of machine and local worker count."""
    support = len(bank.grid)
    bound = 0.15 + (int(seed) % 1000) / 10000.0
    bound += 0.01 * float(kappas_alt[-1]) / (1.0 + float(kappas_alt[-1]))
    weights = np.arange(1, support + 1, dtype=float)
    weights /= weights.sum()
    rule = alfd.TailRule(0.0, 0.5, alpha, "power-cluster-test")
    phase_sizes = dict(training_alternative=len(bank.eigs),
                       alternative_power=(support + 1) * n_sim_power)
    phases = {
        phase: dict(pairs=pairs, raw_evaluations=pairs,
                    order_counts={20: pairs}, max_order=20, max_remainder_ratio=0.0)
        for phase, pairs in phase_sizes.items()
    }
    return alfd.GKMDirectResult(
        bound=bound, bound_se=0.01,
        mixture_power=bound + 0.04, mixture_power_se=0.02,
        epsilon_grid=0.04, weights=weights, log_weights=np.log(weights),
        fit_rejection_probabilities=np.full(support, alpha),
        grid_rejection_probabilities=np.full(support, alpha / 2.0),
        fit_iterations=n_iter, mixture_rule=rule, grid_rule=rule,
        importance_diagnostics={"fixture": True},
        mhg_diagnostics=alfd._combine_phase_mhg_diagnostics(phases))


class PowerClusterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(temporary.cleanup)
        cls.fixture_directory = Path(temporary.name)
        kappas = np.array([100.0, 15.0])
        nuisance = np.asarray([
            np.maximum(alfd.asymptotic_ncp_eigenvalues(beta, kappas, 7, 250), 0.0)[:-1]
            for beta in np.linspace(-2.0, 2.0, alfd.GRID_DESIGN_BETA_COUNT)
        ])
        grid = alfd.common_null_grid_2d(
            nuisance, kappas, standard_points=alfd.ALLOWED_CONFIGS[(100, 15)]["standard"],
            n_shapes=5, n_strengths=7, max_strength=100.0)
        bank_seed = int(np.random.SeedSequence([42, 0x474B4D34]).generate_state(
            1, dtype=np.uint32)[0])
        with contextlib.redirect_stdout(io.StringIO()):
            cls.bank = alfd.build_or_load_pooled_is_bank(
                grid, 7, 2, bank_seed,
                M_start=alfd.ALLOWED_CONFIGS[(100, 15)]["M_start"],
                cache_dir=cls.fixture_directory, cache_metadata=alfd._gkm_provenance(2))
        cls.bank_settings = json.loads(cls.bank.settings_json)
        cls.fixture_bank_path = next(cls.fixture_directory.glob("pooled_gkm_*.npz"))

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.directory = self.root / "shared-power"
        self.bank_directory = self.root / "bank"
        self.bank_directory.mkdir()
        self.bank_path = self.bank_directory / self.fixture_bank_path.name
        shutil.copyfile(self.fixture_bank_path, self.bank_path)
        self.bank_bytes = self.bank_path.read_bytes()
        self.output = io.StringIO()

    def _call(self, function, *args, **kwargs):
        with contextlib.redirect_stdout(self.output), contextlib.redirect_stderr(self.output):
            return function(*args, **kwargs)

    def _prepare(self, **overrides):
        options = dict(version="10015", profile="reference", shards=4,
                       seed=42, beta_count=17, bank=self.bank_path,
                       n_power=3, n_iter=2)
        options.update(overrides)
        return self._call(cluster.prepare_power, self.directory, **options)

    @contextlib.contextmanager
    def _forbid_bank_work(self, forbid_density=True):
        with contextlib.ExitStack() as stack:
            names = ["build_or_load_pooled_is_bank", "_sample_pooled_null_eigenvalues"]
            if forbid_density:
                names.extend(["simulate_Xi", "log_eigval_density_partial"])
            for name in names:
                stack.enter_context(mock.patch.object(
                    alfd, name, side_effect=AssertionError(f"unexpected {name}")))
            yield

    def _run_all(self, shards=4, workers=1, side_effect=result_fixture):
        with self._forbid_bank_work(), mock.patch.object(
                alfd, "gkm_eigval_bound_from_pooled_bank", side_effect=side_effect) as solve:
            for shard in reversed(range(shards)):
                self._call(cluster.run_worker, self.directory, shard, workers=workers)
            path = Path(self._call(cluster.merge_power, self.directory))
        return path, solve

    def _assert_bank_unchanged(self):
        self.assertEqual(self.bank_path.read_bytes(), self.bank_bytes)
        self.assertEqual(list(self.bank_directory.glob("*.npz")), [self.bank_path])

    def test_prepare_only_loads_bank_and_assigns_four_balanced_global_seed_streams(self):
        with self._forbid_bank_work(), mock.patch.object(
                alfd, "gkm_eigval_bound_from_pooled_bank",
                side_effect=AssertionError("prepare calculated power")):
            manifest = self._prepare()
            repeat = self._prepare()
        self.assertEqual(manifest, repeat)
        self.assertEqual(manifest["bank"]["bank_id"], self.bank.bank_id)
        points = manifest["points"]
        np.testing.assert_array_equal([point["beta"] for point in points],
                                      np.linspace(-2.0, 2.0, 17))
        self.assertEqual([point["index"] for point in points], list(range(17)))
        self.assertEqual([sum(point["shard"] == shard for point in points)
                          for shard in range(4)], [4, 4, 4, 4])
        zero = points[8]
        self.assertEqual(zero["beta"], 0.0)
        self.assertIsNone(zero["seed"])
        self.assertIsNone(zero["shard"])
        for point in points:
            self.assertEqual(len(point["ncp"]), 3)
            np.testing.assert_array_equal(point["ncp"], np.maximum(
                alfd.asymptotic_ncp_eigenvalues(point["beta"], [100.0, 15.0], 7, 250), 0.0))
            if point["seed"] is not None:
                expected = int(np.random.SeedSequence(
                    [42, 0x42455441, point["index"]]).generate_state(1, dtype=np.uint32)[0])
                if expected == self.bank.sampling_seed:
                    expected = (expected + 1) % 2 ** 32
                self.assertEqual(point["seed"], expected)
                self.assertNotEqual(point["seed"], self.bank.sampling_seed)
        self._assert_bank_unchanged()

    def test_missing_bank_fails_before_creating_output(self):
        self.bank_path.unlink()
        with self._forbid_bank_work(), self.assertRaises(
                (FileNotFoundError, RuntimeError, ValueError)):
            self._prepare()
        self.assertFalse(self.directory.exists())

    def test_incompatible_bank_provenance_fails_before_creating_output(self):
        changed = dict(alfd._gkm_provenance(2), source_sha256="0" * 64)
        with self._forbid_bank_work(), mock.patch.object(
                alfd, "_gkm_provenance", return_value=changed), \
                self.assertRaisesRegex((RuntimeError, ValueError), "provenance|code|environment"):
            self._prepare()
        self.assertFalse(self.directory.exists())
        self._assert_bank_unchanged()

    def test_changed_bank_is_rejected_before_worker_or_new_preparation(self):
        self._prepare()
        payload = read_npz(self.bank_path)
        payload["eigs"][0, 0] += 0.01
        np.savez(self.bank_path, **payload)
        with self._forbid_bank_work(), mock.patch.object(
                alfd, "gkm_eigval_bound_from_pooled_bank",
                side_effect=AssertionError("changed bank reached solver")), \
                self.assertRaisesRegex((RuntimeError, ValueError), "hash|signature|changed|bank"):
            self._call(cluster.run_worker, self.directory, 0, workers=1)
        self.assertFalse((self.directory / "beta_00000.npz").exists())
        self.directory = self.root / "new-preparation"
        with self._forbid_bank_work(), self.assertRaisesRegex(
                (RuntimeError, ValueError), "hash|signature|changed|bank"):
            self._prepare()
        self.assertFalse(self.directory.exists())

    def test_completed_worker_resumes_without_recomputing_and_bank_is_unchanged(self):
        self._prepare()
        with self._forbid_bank_work(), mock.patch.object(
                alfd, "gkm_eigval_bound_from_pooled_bank", side_effect=result_fixture) as solve:
            self._call(cluster.run_worker, self.directory, 0, workers=1)
        self.assertEqual(solve.call_count, 4)
        before = {path.name: path.read_bytes() for path in self.directory.glob("beta_*.npz")}
        with self._forbid_bank_work(), mock.patch.object(
                alfd, "gkm_eigval_bound_from_pooled_bank",
                side_effect=AssertionError("completed beta recomputed")):
            self._call(cluster.run_worker, self.directory, 0, workers=2)
        after = {path.name: path.read_bytes() for path in self.directory.glob("beta_*.npz")}
        self.assertEqual(before, after)
        self._assert_bank_unchanged()

    def test_interrupted_worker_preserves_completed_beta_and_reuses_pending_seed(self):
        self._prepare()
        seen = []

        def interrupt_second(**kwargs):
            seen.append(kwargs)
            if len(seen) == 2:
                raise KeyboardInterrupt("simulated node interruption")
            return result_fixture(**kwargs)

        with self._forbid_bank_work(), mock.patch.object(
                alfd, "gkm_eigval_bound_from_pooled_bank", side_effect=interrupt_second), \
                self.assertRaises(KeyboardInterrupt):
            self._call(cluster.run_worker, self.directory, 0, workers=1)
        saved = (self.directory / "beta_00000.npz").read_bytes()
        with self._forbid_bank_work(), mock.patch.object(
                alfd, "gkm_eigval_bound_from_pooled_bank", side_effect=result_fixture) as solve:
            self._call(cluster.run_worker, self.directory, 0, workers=2)
        self.assertEqual(solve.call_count, 3)
        self.assertEqual(solve.call_args_list[0].kwargs["seed"], seen[1]["seed"])
        self.assertEqual((self.directory / "beta_00000.npz").read_bytes(), saved)
        self._assert_bank_unchanged()

    def test_grouping_and_worker_counts_leave_all_seventeen_results_identical(self):
        results = []
        for shards, workers in ((1, 1), (4, 2)):
            with self.subTest(shards=shards, workers=workers):
                self.directory = self.root / f"power-{shards}"
                self._prepare(shards=shards)
                path, solve = self._run_all(shards=shards, workers=workers)
                self.assertEqual(solve.call_count, 16)
                self.assertTrue(all(call.kwargs["n_workers"] == workers
                                    for call in solve.call_args_list))
                results.append(read_npz(path))
        for key in ["betas", "ncp", *RESULT_FIELDS, "max_m_used", "diagnostics_json"]:
            np.testing.assert_array_equal(results[0][key], results[1][key], err_msg=key)
        self.assertEqual(results[0]["ncp"].shape, (17, 3))
        self.assertEqual(results[0]["fitted_weights"].shape, (17, 36))
        self.assertEqual(float(results[0]["bounds"][8]), 0.05)
        self.assertEqual(float(results[0]["bounds_se"][8]), 0.0)
        self.assertEqual(int(results[0]["fit_iterations"][8]), 0)
        self._assert_bank_unchanged()

    def test_real_beta_results_and_full_metadata_survive_merge_and_csv(self):
        manifest = self._prepare()
        real_seeds = {manifest["points"][index]["seed"] for index in (0, 16)}
        native_solver = alfd.gkm_eigval_bound_from_pooled_bank

        def selective_solver(**kwargs):
            return native_solver(**kwargs) if kwargs["seed"] in real_seeds else result_fixture(**kwargs)

        with self._forbid_bank_work(forbid_density=False), mock.patch.object(
                alfd, "gkm_eigval_bound_from_pooled_bank", side_effect=selective_solver) as solve:
            for shard in (3, 2, 1, 0):
                self._call(cluster.run_worker, self.directory, shard, workers=2)
            path = Path(self._call(cluster.merge_power, self.directory))
        self.assertEqual(solve.call_count, 16)
        saved = read_npz(path)
        self.assertEqual(path.name, "gkm_eigval_10015.npz")
        np.testing.assert_array_equal(saved["betas"], np.linspace(-2.0, 2.0, 17))
        self.assertEqual(saved["ncp"].shape, (17, 3))
        for point in manifest["points"]:
            index = point["index"]
            if point["beta"] == 0.0:
                expected = alfd._exact_gkm_result(0.05, 36)
            else:
                options = dict(
                    kappas_alt=point["ncp"], bank=self.bank, k_eff=7, alpha=0.05,
                    n_sim_power=3, n_iter=2, seed=point["seed"], verbose=False,
                    n_workers=1, M_trunc=self.bank_settings["M_start"],
                    M_step=self.bank_settings["M_step"], M_max=self.bank_settings["M_max"],
                    mhg_tol=self.bank_settings["mhg_tol"])
                expected = self._call(native_solver if point["seed"] in real_seeds
                                      else result_fixture, **options)
            for field, attribute in RESULT_FIELDS.items():
                np.testing.assert_array_equal(saved[field][index], getattr(expected, attribute),
                                              err_msg=f"beta {point['beta']}: {field}")
            self.assertEqual(int(saved["max_m_used"][index]), expected.mhg_diagnostics["max_order"])
            diagnostics = json.loads(str(saved["diagnostics_json"][index]))
            self.assertEqual(diagnostics["power_seed"], point["seed"])
            self.assertEqual(diagnostics["mhg"], alfd._json_safe(expected.mhg_diagnostics))

        self.assertEqual(str(saved["bank_id"].item()), self.bank.bank_id)
        self.assertEqual(str(saved["bank_content_signature"].item()), self.bank.content_signature)
        self.assertEqual(str(saved["version_label"].item()), "10015")
        np.testing.assert_array_equal(saved["common_null_grid"], self.bank.grid)
        for key in ("schema_version", "algorithm", "producer", "calibration_method", "bound_kind",
                    "run_signature", "settings_json", "kappas", "k", "n", "alpha",
                    "density_accuracy_scope", "common_grid_size", "bank_mhg_diagnostics_json",
                    "M_start", "M_step", "M_max", "mhg_rtol", "seed", "n_fit", "n_power", "n_iter"):
            self.assertIn(key, saved)
        with (path.parent / "gkm_bounds_10015.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 17)
        np.testing.assert_array_equal([float(row["beta"]) for row in rows], saved["betas"])
        for csv_field, saved_field in (
                ("bound", "bounds"), ("bound_se", "bounds_se"),
                ("mixture_power", "mixture_power"), ("mixture_power_se", "mixture_power_se"),
                ("epsilon_grid", "epsilon_grid")):
            np.testing.assert_array_equal([float(row[csv_field]) for row in rows], saved[saved_field])
        self._assert_bank_unchanged()

    def test_merge_refuses_missing_result(self):
        self._prepare()
        with self._forbid_bank_work(), mock.patch.object(
                alfd, "gkm_eigval_bound_from_pooled_bank", side_effect=result_fixture):
            self._call(cluster.run_worker, self.directory, 0, workers=1)
        with self.assertRaisesRegex((RuntimeError, ValueError), "missing|incomplete"):
            self._call(cluster.merge_power, self.directory)
        self.assertFalse((self.directory / "gkm_eigval_10015.npz").exists())

    def test_merge_refuses_corrupt_and_wrong_beta_results(self):
        self._prepare()
        with self._forbid_bank_work(), mock.patch.object(
                alfd, "gkm_eigval_bound_from_pooled_bank", side_effect=result_fixture):
            for shard in range(4):
                self._call(cluster.run_worker, self.directory, shard, workers=1)
        first = self.directory / "beta_00000.npz"
        second = self.directory / "beta_00001.npz"
        original = second.read_bytes()
        payload = read_npz(second)
        payload["bound"] += 0.001
        np.savez(second, **payload)
        with self.assertRaisesRegex((RuntimeError, ValueError), "hash|signature|trust|inconsistent"):
            self._call(cluster.merge_power, self.directory)
        second.write_bytes(original)
        shutil.copyfile(first, second)
        with self.assertRaisesRegex((RuntimeError, ValueError), "identity|beta|index|trust"):
            self._call(cluster.merge_power, self.directory)
        self.assertFalse((self.directory / "gkm_eigval_10015.npz").exists())

    def test_duplicate_beta_lock_refuses_work_and_released_lock_allows_resume(self):
        self._prepare()
        with (self.directory / "beta_00000.lock").open("a+") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self._forbid_bank_work(), mock.patch.object(
                    alfd, "gkm_eigval_bound_from_pooled_bank",
                    side_effect=AssertionError("duplicate beta computed")), \
                    self.assertRaisesRegex(RuntimeError, "lock|holds|running|busy"):
                self._call(cluster.run_worker, self.directory, 0, workers=1)
        with self._forbid_bank_work(), mock.patch.object(
                alfd, "gkm_eigval_bound_from_pooled_bank", side_effect=result_fixture) as solve:
            self._call(cluster.run_worker, self.directory, 0, workers=1)
        self.assertEqual(solve.call_count, 4)
        self._assert_bank_unchanged()


if __name__ == "__main__":
    unittest.main()
