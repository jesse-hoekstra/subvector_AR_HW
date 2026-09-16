import contextlib
import csv
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

import alfd_eigval as alfd
import new_power_comparison as comparison
import plot_power_comparison as plotter
import power_bound_cluster as cluster


class PowerComparisonPlotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(temporary.cleanup)
        root = Path(temporary.name)
        # A real tiny bank exercises the exact remote driver's output format.
        with contextlib.redirect_stdout(io.StringIO()):
            alfd.build_or_load_pooled_is_bank(
                [(0.0, 0.0), (100.0, 15.0)], 7, 2, 123,
                M_start=20, cache_dir=root, cache_metadata=alfd._gkm_provenance(2))
            directory = root / "power17"
            cluster.prepare_power(directory, bank=next(root.glob("pooled_gkm_*.npz")),
                                  n_power=4, n_iter=2)

            def fake_result(**kwargs):
                result = alfd._exact_gkm_result(kwargs["alpha"], len(kwargs["bank"].grid))
                result.fit_iterations = kwargs["n_iter"]
                result.bound = result.mixture_power = 0.3
                result.bound_se = result.mixture_power_se = 0.01
                result.importance_diagnostics = {"synthetic_test": True}
                return result

            with mock.patch.object(alfd, "gkm_eigval_bound_from_pooled_bank", side_effect=fake_result):
                for shard in range(4):
                    cluster.run_worker(directory, shard, workers=1)
            cls.bound_path = cluster.merge_power(directory)

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.dgp_path = self.root / "dgp.npz"
        self.betas = np.linspace(-2, 2, 17)
        self._save_dgp()

    def _save_dgp(self, **overrides):
        args = dict(version_label="10015", kappas=[100.0, 15.0], k=7, n=250, alpha=0.05,
                    betas=self.betas, power_chi2=np.full(17, 0.1),
                    power_c1=np.full(17, 0.15), power_cp1=np.full(17, 0.2),
                    num_simulations=100, base_seed=23, chunk_size=10, workers_used=1)
        args.update(overrides)
        comparison.save_dgp_cache(str(self.dgp_path), **args)

    def test_real_merged_format_plots_all_four_series_and_exports_values(self):
        dgp_before = self.dgp_path.read_bytes()
        bound_before = self.bound_path.read_bytes()
        png, table = plotter.write_comparison(
            self.dgp_path, self.bound_path, self.root / "comparison.png")
        self.assertTrue(png.read_bytes().startswith(b"\x89PNG"))
        with table.open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 68)
        self.assertEqual({row["series"] for row in rows},
                         {"power_chi2", "power_c1", "power_cp1", "gkm_bound"})
        for row in rows[:17]:
            self.assertAlmostEqual(float(row["mc_se"]), np.sqrt(0.1 * 0.9 / 100))
        self.assertEqual(self.dgp_path.read_bytes(), dgp_before)
        self.assertEqual(self.bound_path.read_bytes(), bound_before)

    def test_dgp_only_plot_does_not_require_a_bound_or_bank(self):
        _, table = plotter.write_comparison(self.dgp_path, output=self.root / "dgp.png")
        with table.open() as handle:
            self.assertEqual(len(list(csv.DictReader(handle))), 51)

    def test_same_experiment_is_required_before_outputs_are_created(self):
        self._save_dgp(n=251)
        path = self.root / "invalid.png"
        with self.assertRaisesRegex(ValueError, "matching"):
            plotter.write_comparison(self.dgp_path, self.bound_path, path)
        self.assertFalse(path.exists())

    def test_tampered_bound_estimate_is_rejected_by_saved_content_hash(self):
        arrays = plotter._read_npz(self.bound_path)
        arrays["bounds"][0] -= 0.01
        arrays["epsilon_grid"][0] += 0.01
        path = self.root / "tampered.npz"
        np.savez(path, **arrays)
        with self.assertRaisesRegex(ValueError, "content signature"):
            plotter.load_bound(path)

    def test_incomplete_and_malformed_bound_arrays_are_refused(self):
        for key, value in (("bounds", np.array([np.nan] * 17)),
                           ("betas", self.betas[::-1]),
                           ("bounds_se", np.array([0.1] * 16))):
            with self.subTest(key=key):
                arrays = plotter._read_npz(self.bound_path)
                arrays[key] = value
                path = self.root / "bad.npz"
                np.savez(path, **arrays)
                with self.assertRaises(ValueError):
                    plotter.load_bound(path)

    def test_bound_can_be_viewed_without_matching_remote_platform_or_native_library(self):
        # The completed artifact was made by the native driver; forbid imports
        # of it in a fresh viewer process, including any native-library loading.
        program = (
            "import sys\n"
            "sys.modules['alfd_eigval'] = None\n"
            "sys.modules['power_bound_cluster'] = None\n"
            "import plot_power_comparison as p\n"
            "import platform\n"
            "platform.platform = lambda: 'different-viewing-machine'\n"
            "b = p.load_bound(sys.argv[1])\n"
            "assert len(b['bounds']) == 17\n"
        )
        subprocess.run([sys.executable, "-c", program, str(self.bound_path)],
                       cwd=Path(__file__).resolve().parents[1], check=True,
                       capture_output=True, text=True)

    def test_cli_refuses_wrong_version(self):
        with self.assertRaises(SystemExit) as failure, contextlib.redirect_stderr(io.StringIO()):
            plotter.main(["--version", "352515", "--dgp-cache", str(self.dgp_path)])
        self.assertEqual(failure.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
