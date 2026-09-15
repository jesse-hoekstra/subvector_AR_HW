import contextlib
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
import null_bank_cluster as cluster


class ClusterNullBankTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.directory = self.root / "shared-bank"
        self.output = io.StringIO()

    def _call(self, function, *args, **kwargs):
        with contextlib.redirect_stdout(self.output):
            return function(*args, **kwargs)

    def _prepare(self, **overrides):
        options = dict(version="10015", profile="reference", shards=2,
                       seed=123, n_fit=2)
        options.update(overrides)
        return self._call(cluster.prepare_bank, self.directory, **options)

    @staticmethod
    def _local_bank(settings, **options):
        return alfd.build_or_load_pooled_is_bank(
            settings["grid"], settings["k_eff"], settings["n_per_stratum"],
            settings["seed"], M_start=settings["M_start"],
            M_step=settings["M_step"], M_max=settings["M_max"],
            mhg_tol=settings["mhg_tol"], role=settings["role"],
            cache_metadata=settings["provenance"], **options)

    def _assert_same_bank(self, expected, actual):
        for name in ("bank_id", "content_signature", "settings_json",
                     "experiment_signature", "sampling_seed", "k_eff",
                     "n_per_stratum", "role"):
            self.assertEqual(getattr(expected, name), getattr(actual, name), name)
        self.assertEqual(alfd._json_safe(expected.mhg_diagnostics),
                         alfd._json_safe(actual.mhg_diagnostics))
        for name in ("grid", "eigs", "log_f", "log_q", "base_weights", "strata"):
            np.testing.assert_array_equal(getattr(expected, name),
                                          getattr(actual, name), err_msg=name)

    def test_real_distributed_banks_exactly_match_local_banks_and_reload(self):
        for shards, workers, n_fit in ((2, 1, 2), (3, 2, 3)):
            with self.subTest(shards=shards, workers=workers, n_fit=n_fit):
                self.directory = self.root / f"shared-{shards}"
                manifest = self._prepare(shards=shards, n_fit=n_fit)
                settings = manifest["settings"]
                self.assertEqual(np.asarray(settings["grid"]).shape, (36, 2))
                expected = self._call(self._local_bank, settings, n_workers=1)

                # Jobs may finish in any order, without changing sample order.
                for shard in reversed(range(shards)):
                    self._call(cluster.run_worker, self.directory, shard,
                               workers=workers)
                cache_directory = self.root / f"cache-{shards}"
                actual = self._call(cluster.merge_bank, self.directory,
                                    cache_dir=cache_directory)
                self._assert_same_bank(expected, actual)
                self.assertEqual(len(list(cache_directory.glob("pooled_gkm_*.npz"))), 1)

                with mock.patch.object(
                        alfd, "simulate_Xi",
                        side_effect=AssertionError("merged cache resampled")), \
                        mock.patch.object(
                            alfd, "log_eigval_density_partial",
                            side_effect=AssertionError("merged cache recomputed")):
                    reloaded = self._call(self._local_bank, settings,
                                          n_workers=2, cache_dir=cache_directory)
                self._assert_same_bank(expected, reloaded)

    def test_prepare_does_no_density_work_and_reuses_identical_samples(self):
        with mock.patch.object(
                alfd, "log_eigval_density_partial",
                side_effect=AssertionError("prepare evaluated densities")):
            original = self._prepare()
            sample_bytes = (self.directory / "eigs.npy").read_bytes()
            manifest_bytes = (self.directory / "manifest.json").read_bytes()
            with mock.patch.object(
                    alfd, "simulate_Xi",
                    side_effect=AssertionError("repeated prepare resampled")):
                repeated = self._prepare()

        self.assertEqual(original, repeated)
        self.assertEqual((self.directory / "eigs.npy").read_bytes(), sample_bytes)
        self.assertEqual((self.directory / "manifest.json").read_bytes(), manifest_bytes)
        self.assertFalse(list(self.directory.glob("shard_*.npz")))

    def test_completed_shard_is_validated_and_reused_without_density_work(self):
        self._prepare()
        path = Path(self._call(cluster.run_worker, self.directory, 0, workers=1))
        original = path.read_bytes()
        with mock.patch.object(
                alfd, "log_eigval_density_partial",
                side_effect=AssertionError("completed shard recomputed")), \
                mock.patch.object(
                    alfd, "simulate_Xi",
                    side_effect=AssertionError("worker resampled")):
            reused = self._call(cluster.run_worker, self.directory, 0, workers=2)
        self.assertEqual(Path(reused), path)
        self.assertEqual(path.read_bytes(), original)

    def test_prepare_refuses_changed_options_without_overwriting_samples(self):
        self._prepare()
        sample_bytes = (self.directory / "eigs.npy").read_bytes()
        manifest_bytes = (self.directory / "manifest.json").read_bytes()
        for changed in (dict(n_fit=3), dict(shards=3), dict(seed=124)):
            with self.subTest(changed=changed), mock.patch.object(
                    alfd, "simulate_Xi",
                    side_effect=AssertionError("conflicting prepare resampled")), \
                    self.assertRaisesRegex(RuntimeError, "different settings"):
                self._prepare(**changed)
        self.assertEqual((self.directory / "eigs.npy").read_bytes(), sample_bytes)
        self.assertEqual((self.directory / "manifest.json").read_bytes(), manifest_bytes)

    def test_status_checks_complete_missing_and_invalid_shards_without_computing(self):
        self._prepare(shards=3)
        self._call(cluster.run_worker, self.directory, 0)
        invalid_path = Path(self._call(cluster.run_worker, self.directory, 1))
        truncated = invalid_path.read_bytes()[:32]
        for damaged in (b"invalid shard file", truncated):
            with self.subTest(damaged=damaged):
                invalid_path.write_bytes(damaged)
                with mock.patch.object(
                        alfd, "log_eigval_density_partial",
                        side_effect=AssertionError("status evaluated densities")), \
                        mock.patch.object(
                            alfd, "simulate_Xi",
                            side_effect=AssertionError("status resampled")):
                    status = self._call(cluster.bank_status, self.directory)
                    result = self._call(cluster.main,
                                        ["status", "--directory", str(self.directory)])

                self.assertEqual(status["shard_count"], 3)
                self.assertEqual(status["complete"], [0])
                self.assertEqual(status["missing"], [2])
                self.assertEqual(set(status["invalid"]), {1})
                self.assertIn("trust", status["invalid"][1])
                self.assertEqual(result, 1)

    def test_merge_refuses_missing_shard_without_writing_a_cache(self):
        self._prepare()
        self._call(cluster.run_worker, self.directory, 0)
        cache_directory = self.root / "cache"
        with self.assertRaisesRegex(RuntimeError, "[Mm]issing|[Ii]ncomplete"):
            self._call(cluster.merge_bank, self.directory, cache_dir=cache_directory)
        self.assertFalse(list(cache_directory.glob("*.npz")))

    def test_merge_refuses_corrupt_numeric_payload(self):
        self._prepare()
        paths = [Path(self._call(cluster.run_worker, self.directory, shard))
                 for shard in range(2)]
        with np.load(paths[1], allow_pickle=False) as archive:
            payload = {name: np.asarray(archive[name]).copy() for name in archive.files}
        payload["log_f"][0, 0] += 1.0
        np.savez(paths[1], **payload)

        cache_directory = self.root / "cache"
        with self.assertRaisesRegex(RuntimeError, "hash|signature|corrupt|trust"):
            self._call(cluster.merge_bank, self.directory, cache_dir=cache_directory)
        self.assertFalse(list(cache_directory.glob("*.npz")))

    def test_merge_refuses_a_different_shard_saved_under_the_expected_name(self):
        self._prepare()
        first = Path(self._call(cluster.run_worker, self.directory, 0))
        second = Path(self._call(cluster.run_worker, self.directory, 1))
        shutil.copyfile(first, second)

        cache_directory = self.root / "cache"
        with self.assertRaisesRegex(RuntimeError, "shard|identity|metadata|trust"):
            self._call(cluster.merge_bank, self.directory, cache_dir=cache_directory)
        self.assertFalse(list(cache_directory.glob("*.npz")))

    def test_worker_refuses_modified_samples_before_density_work(self):
        self._prepare()
        sample_path = self.directory / "eigs.npy"
        samples = np.load(sample_path, allow_pickle=False)
        samples[0, 0] += 1.0
        np.save(sample_path, samples, allow_pickle=False)

        with mock.patch.object(
                alfd, "log_eigval_density_partial",
                side_effect=AssertionError("invalid samples reached density code")), \
                self.assertRaisesRegex(RuntimeError, "sample|eigs|hash|signature"):
            self._call(cluster.run_worker, self.directory, 0)
        self.assertFalse(list(self.directory.glob("shard_*.npz")))

    def test_worker_refuses_modified_manifest_provenance(self):
        self._prepare()
        path = self.directory / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["settings"]["provenance"]["source_sha256"] = "0" * 64
        path.write_text(json.dumps(manifest))

        with mock.patch.object(
                alfd, "log_eigval_density_partial",
                side_effect=AssertionError("invalid manifest reached density code")), \
                self.assertRaisesRegex(RuntimeError, "hash|signature|provenance|settings"):
            self._call(cluster.run_worker, self.directory, 0)

    def test_worker_refuses_different_local_code_or_environment(self):
        manifest = self._prepare()
        changed = dict(manifest["settings"]["provenance"], source_sha256="0" * 64)
        with mock.patch.object(alfd, "_gkm_provenance", return_value=changed), \
                mock.patch.object(
                    alfd, "log_eigval_density_partial",
                    side_effect=AssertionError("mismatched worker computed densities")), \
                self.assertRaisesRegex(RuntimeError, "provenance|environment|code|settings"):
            self._call(cluster.run_worker, self.directory, 0)

    def test_duplicate_worker_is_refused_while_shard_lock_is_held(self):
        self._prepare()
        with (self.directory / "shard_00000.lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with mock.patch.object(
                    alfd, "log_eigval_density_partial",
                    side_effect=AssertionError("duplicate worker started")), \
                    self.assertRaisesRegex(RuntimeError, "lock|running|busy|active"):
                self._call(cluster.run_worker, self.directory, 0)

        # An old lock file itself is harmless once the actual lock is released.
        path = self._call(cluster.run_worker, self.directory, 0)
        self.assertTrue(Path(path).is_file())


if __name__ == "__main__":
    unittest.main()
