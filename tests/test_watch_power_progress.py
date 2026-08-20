import csv
import hashlib
import io
import json
import os
import sys
import tempfile
import types
import unittest
from unittest import mock

import matplotlib
import numpy as np

matplotlib.use("Agg")

import new_power_comparison as comparison
import watch_power_progress as watcher


VERSION = "352515"
KAPPAS = np.array([35.0, 25.0, 15.0])
K = 7
N = 250
ALPHA = 0.05


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _save_dgp_cache(path):
    betas = np.linspace(-2.0, 2.0, 5)
    curves = (
        np.array([0.07, 0.06, 0.05, 0.07, 0.10]),
        np.array([0.08, 0.07, 0.05, 0.09, 0.12]),
        np.array([0.10, 0.09, 0.05, 0.12, 0.16]),
    )
    comparison.save_dgp_cache(
        path, version_label=VERSION, kappas=KAPPAS, k=K, n=N,
        alpha=ALPHA, betas=betas, power_chi2=curves[0],
        power_c1=curves[1], power_cp1=curves[2], num_simulations=1_000,
        base_seed=123, chunk_size=100, workers_used=2)
    return (betas,) + curves


def _bound_settings(beta_count):
    repository = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    library_name = "libmhg.dylib" if sys.platform == "darwin" else "libmhg.so"
    source_hash = _sha256(os.path.join(repository, "alfd_eigval.py"))
    core_hash = _sha256(os.path.join(repository, "koev", "mhg15", "mhg_core.c"))
    library_hash = _sha256(
        os.path.join(repository, "koev", "mhg15", library_name))
    common_grid = [
        [0.0, 0.0, 0.0],
        [0.1, 0.05, 0.0],
        [100.0, 50.0, 0.0],
        [35.0, 25.0, 15.0],
    ]
    return dict(
        schema_version=watcher.GKM_SCHEMA_VERSION,
        algorithm=watcher.GKM_ALGORITHM,
        producer=watcher.GKM_PRODUCER,
        calibration_method=watcher.GKM_CALIBRATION_METHOD,
        version_label=VERSION,
        kappas=KAPPAS.tolist(), k=K, n=N, alpha=ALPHA,
        profile="production", beta_count=beta_count,
        fit_grid_strategy=watcher.GKM_COMMON_GRID_METHOD,
        pooled_importance_method=watcher.GKM_POOLED_IS_METHOD,
        common_grid=common_grid, grid_shapes=1, grid_strengths=2,
        grid_max_strength=100.0, grid_anchor_count=1,
        bank_seed=1234, n_fit=2_000, n_power=50_000, n_iter=600,
        source_sha256=source_hash,
        mhg_core_sha256=core_hash, mhg_library_sha256=library_hash,
        mhg_build_source_sha256=core_hash,
    )


def _signature(settings):
    canonical = json.dumps(settings, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def _save_partial_arrays(path, betas, bounds, bounds_se):
    betas = np.asarray(betas, dtype=float)
    bounds = np.asarray(bounds, dtype=float)
    bounds_se = np.asarray(bounds_se, dtype=float)
    settings = _bound_settings(betas.size)
    run_signature = _signature(settings)
    np.savez(
        path,
        schema_version=np.array(watcher.GKM_SCHEMA_VERSION),
        algorithm=np.array(watcher.GKM_ALGORITHM),
        producer=np.array(watcher.GKM_PRODUCER),
        calibration_method=np.array(watcher.GKM_CALIBRATION_METHOD),
        bound_kind=np.array(watcher.GKM_BOUND_KIND),
        version_label=np.array(VERSION),
        run_signature=np.array(run_signature),
        settings_json=np.array(json.dumps(settings, sort_keys=True)),
        kappas=KAPPAS, k=np.array(K), n=np.array(N),
        alpha=np.array(ALPHA), betas=betas, bounds=bounds,
        bounds_se=bounds_se,
    )
    return betas, bounds, bounds_se, run_signature


def _save_partial(path):
    return _save_partial_arrays(
        path,
        betas=[-2.0, -1.0, 0.0, 1.0, 2.0],
        bounds=[0.12, np.nan, 0.05, np.nan, 0.18],
        bounds_se=[0.003, np.nan, 0.0, np.nan, 0.004],
    )


def _save_final(path):
    betas = np.array([-1.0, 0.0, 1.0])
    bounds = np.array([0.12, 0.05, 0.15])
    bounds_se = np.array([0.003, 0.0, 0.004])
    settings = _bound_settings(betas.size)
    run_signature = _signature(settings)
    grid = np.asarray(settings["common_grid"], dtype=float)
    np.savez(
        path,
        schema_version=np.array(watcher.GKM_SCHEMA_VERSION),
        algorithm=np.array(watcher.GKM_ALGORITHM),
        producer=np.array(watcher.GKM_PRODUCER),
        calibration_method=np.array(watcher.GKM_CALIBRATION_METHOD),
        bound_kind=np.array(watcher.GKM_BOUND_KIND),
        density_accuracy_scope=np.array("adaptive_empirical_tail_criterion"),
        version_label=np.array(VERSION), run_signature=np.array(run_signature),
        source_sha256=np.array(settings["source_sha256"]),
        mhg_core_sha256=np.array(settings["mhg_core_sha256"]),
        mhg_library_sha256=np.array(settings["mhg_library_sha256"]),
        mhg_build_source_sha256=np.array(
            settings["mhg_build_source_sha256"]),
        settings_json=np.array(json.dumps(settings, sort_keys=True)),
        betas=betas, bounds=bounds, bounds_se=bounds_se,
        mixture_power=np.array([0.13, 0.05, 0.16]),
        mixture_power_se=np.array([0.003, 0.0, 0.004]),
        epsilon_grid=np.array([0.01, 0.0, 0.01]),
        common_null_grid=grid, common_grid_size=np.array(grid.shape[0]),
        grid_shapes=np.array(1), grid_strengths=np.array(2),
        grid_max_strength=np.array(100.0), grid_anchor_count=np.array(1),
        kappas=KAPPAS, k=np.array(K), n=np.array(N),
        alpha=np.array(ALPHA),
    )
    return betas, bounds, bounds_se, run_signature


class DgpLoadingTests(unittest.TestCase):
    def test_loads_only_a_signed_compatible_dgp_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "dgp.npz")
            expected = _save_dgp_cache(path)

            loaded = watcher.load_dgp_curves(VERSION, path)
            self.assertEqual(loaded.path, path)
            for actual, wanted in zip(
                    (loaded.betas, loaded.power_chi2, loaded.power_c1,
                     loaded.power_cp1), expected):
                np.testing.assert_array_equal(actual, wanted)

            with np.load(path, allow_pickle=False) as archive:
                payload = {name: np.asarray(archive[name]).copy()
                           for name in archive.files}
            payload["run_signature"] = np.array("tampered")
            np.savez(path, **payload)
            with self.assertRaisesRegex(ValueError, "run_signature"):
                watcher.load_dgp_curves(VERSION, path)


class BoundProgressLoadingTests(unittest.TestCase):
    def test_partial_preserves_only_completed_finite_points(self):
        with tempfile.TemporaryDirectory() as directory:
            partial = os.path.join(directory, "partial.npz")
            expected = _save_partial(partial)
            progress = watcher.load_bound_progress(
                VERSION, partial_path=partial,
                final_path=os.path.join(directory, "missing-final.npz"),
                expected_run_signature=expected[-1])

            self.assertFalse(progress.is_final)
            np.testing.assert_array_equal(progress.betas, expected[0])
            np.testing.assert_array_equal(
                np.isfinite(progress.bounds),
                np.array([True, False, True, False, True]))
            np.testing.assert_allclose(
                progress.bounds[np.isfinite(progress.bounds)],
                [0.12, 0.05, 0.18])

            with self.assertRaisesRegex(ValueError, "run signature"):
                watcher.load_bound_progress(
                    VERSION, partial_path=partial,
                    final_path=os.path.join(directory, "missing-final.npz"),
                    expected_run_signature="different-run")

    def test_final_artifact_is_used_when_partial_is_absent_or_stale(self):
        with tempfile.TemporaryDirectory() as directory:
            final = os.path.join(directory, "final.npz")
            expected = _save_final(final)
            stale_partial = os.path.join(directory, "partial.npz")
            np.savez(stale_partial, run_signature=np.array("stale"))

            progress = watcher.load_bound_progress(
                VERSION, partial_path=stale_partial, final_path=final,
                expected_run_signature=expected[-1])

            self.assertTrue(progress.is_final)
            self.assertEqual(progress.source_path, final)
            np.testing.assert_array_equal(progress.betas, expected[0])
            np.testing.assert_array_equal(progress.bounds, expected[1])
            np.testing.assert_array_equal(progress.bounds_se, expected[2])

    def test_newest_matching_artifact_wins_force_rerun_and_handoff_race(self):
        with tempfile.TemporaryDirectory() as directory:
            partial = os.path.join(directory, "partial.npz")
            final = os.path.join(directory, "final.npz")
            final_expected = _save_final(final)
            partial_expected = _save_partial_arrays(
                partial,
                betas=final_expected[0],
                bounds=[0.13, np.nan, np.nan],
                bounds_se=[0.003, np.nan, np.nan],
            )
            self.assertEqual(partial_expected[-1], final_expected[-1])

            timestamp = 1_700_000_000_000_000_000
            os.utime(final, ns=(timestamp, timestamp))
            os.utime(partial, ns=(timestamp + 1_000_000,)*2)
            active_rerun = watcher.load_bound_progress(
                VERSION, partial_path=partial, final_path=final,
                expected_run_signature=final_expected[-1])
            self.assertFalse(active_rerun.is_final)
            self.assertEqual(active_rerun.source_path, partial)
            np.testing.assert_array_equal(
                np.isfinite(active_rerun.bounds),
                [True, False, False])

            os.utime(final, ns=(timestamp + 2_000_000,)*2)
            completed_handoff = watcher.load_bound_progress(
                VERSION, partial_path=partial, final_path=final,
                expected_run_signature=final_expected[-1])
            self.assertTrue(completed_handoff.is_final)
            self.assertEqual(completed_handoff.source_path, final)
            np.testing.assert_array_equal(
                completed_handoff.bounds, final_expected[1])

    def test_partial_deleted_during_handoff_returns_loaded_final(self):
        with tempfile.TemporaryDirectory() as directory:
            partial = os.path.join(directory, "partial.npz")
            final = os.path.join(directory, "final.npz")
            final_expected = _save_final(final)
            partial_expected = _save_partial_arrays(
                partial,
                betas=final_expected[0],
                bounds=[0.13, np.nan, np.nan],
                bounds_se=[0.003, np.nan, np.nan],
            )
            self.assertEqual(partial_expected[-1], final_expected[-1])

            real_load_final = watcher._load_final

            def load_final_then_remove_partial(*args, **kwargs):
                loaded = real_load_final(*args, **kwargs)
                os.unlink(partial)
                return loaded

            with mock.patch.object(
                    watcher, "_load_final",
                    side_effect=load_final_then_remove_partial):
                progress = watcher.load_bound_progress(
                    VERSION, partial_path=partial, final_path=final,
                    expected_run_signature=final_expected[-1])

            self.assertFalse(os.path.exists(partial))
            self.assertTrue(progress.is_final)
            self.assertEqual(progress.source_path, final)
            np.testing.assert_array_equal(
                progress.bounds, final_expected[1])


class PlotTests(unittest.TestCase):
    def _inputs(self, directory):
        dgp_path = os.path.join(directory, "dgp.npz")
        partial_path = os.path.join(directory, "partial.npz")
        _save_dgp_cache(dgp_path)
        _save_partial(partial_path)
        return (
            watcher.load_dgp_curves(VERSION, dgp_path),
            watcher.load_bound_progress(
                VERSION, partial_path=partial_path,
                final_path=os.path.join(directory, "missing.npz")),
        )

    def test_figure_contains_all_dgp_curves_and_only_completed_bound_points(self):
        with tempfile.TemporaryDirectory() as directory:
            dgp, progress = self._inputs(directory)
            figure, metrics = watcher.build_progress_figure(dgp, progress)
            self.addCleanup(lambda: matplotlib.pyplot.close(figure))
            axes = figure.axes[0]
            line_data = [(np.asarray(line.get_xdata(), dtype=float),
                          np.asarray(line.get_ydata(), dtype=float))
                         for line in axes.lines]

            self.assertEqual(len(axes.lines), 5)
            self.assertEqual(
                [line.get_label() for line in axes.lines],
                [r"$\chi^2$", r"$c_1$", r"$c_3$",
                 r"GKM power bound ($m_W=3$)",
                 r"$\alpha=0.05$"])
            self.assertEqual(len(axes.collections), 0)
            self.assertEqual(len(axes.texts), 0)

            for curve in (dgp.power_chi2, dgp.power_c1, dgp.power_cp1):
                self.assertTrue(any(
                    x.shape == dgp.betas.shape
                    and np.array_equal(x, dgp.betas)
                    and np.array_equal(y, curve)
                    for x, y in line_data))

            complete = np.isfinite(progress.bounds)
            self.assertTrue(any(
                np.array_equal(x, progress.betas[complete])
                and np.array_equal(y, progress.bounds[complete])
                for x, y in line_data))
            self.assertEqual(metrics["completed_beta_count"], 3)

    def test_png_publish_is_atomic_and_preserves_old_file_on_render_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            dgp, progress = self._inputs(directory)
            output = os.path.join(directory, "live.png")
            with mock.patch.object(
                    watcher.os, "replace", wraps=os.replace) as replace:
                watcher.write_progress_plot(dgp, progress, output)
            self.assertTrue(os.path.isfile(output))
            source, destination = replace.call_args.args
            self.assertEqual(destination, output)
            self.assertEqual(os.path.dirname(source), directory)
            self.assertNotEqual(source, output)

            with open(output, "rb") as handle:
                original = handle.read()
            with mock.patch(
                    "matplotlib.figure.Figure.savefig",
                    side_effect=RuntimeError("render failed")):
                with self.assertRaisesRegex(RuntimeError, "render failed"):
                    watcher.write_progress_plot(dgp, progress, output)
            with open(output, "rb") as handle:
                self.assertEqual(handle.read(), original)

    def test_all_nan_checkpoint_and_no_checkpoint_render_baselines(self):
        with tempfile.TemporaryDirectory() as directory:
            dgp_path = os.path.join(directory, "dgp.npz")
            partial_path = os.path.join(directory, "partial.npz")
            _save_dgp_cache(dgp_path)
            all_nan = np.full(3, np.nan)
            _save_partial_arrays(
                partial_path, betas=[-1.0, 0.0, 1.0],
                bounds=all_nan, bounds_se=all_nan)
            dgp = watcher.load_dgp_curves(VERSION, dgp_path)
            progress = watcher.load_bound_progress(
                VERSION, partial_path=partial_path,
                final_path=os.path.join(directory, "missing-final.npz"))

            figure, metrics = watcher.build_progress_figure(dgp, progress)
            self.addCleanup(lambda: matplotlib.pyplot.close(figure))
            self.assertEqual(metrics["completed_beta_count"], 0)
            self.assertEqual(metrics["total_beta_count"], 3)
            self.assertIn("live: 0/3 betas", figure.axes[0].get_title())
            self.assertEqual(len(figure.axes[0].lines), 5)

            output = os.path.join(directory, "waiting.png")
            with mock.patch("builtins.print"):
                watcher.main([
                    "--version", VERSION,
                    "--dgp-cache", dgp_path,
                    "--partial-path", os.path.join(directory, "missing.npz"),
                    "--final-path", os.path.join(directory, "also-missing.npz"),
                    "--output", output,
                    "--once",
                ])
            self.assertTrue(os.path.isfile(output))

            waiting_figure, waiting_metrics = watcher.build_progress_figure(
                dgp, None)
            self.addCleanup(lambda: matplotlib.pyplot.close(waiting_figure))
            self.assertEqual(waiting_metrics["completed_beta_count"], 0)
            self.assertEqual(waiting_metrics["total_beta_count"], 0)
            self.assertIn(
                "waiting for checkpoint", waiting_figure.axes[0].get_title())


class CsvTests(unittest.TestCase):
    def test_csv_is_atomic_complete_and_carries_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            dgp_path = os.path.join(directory, "dgp.npz")
            partial_path = os.path.join(directory, "partial.npz")
            output = os.path.join(directory, "progress.csv")
            _save_dgp_cache(dgp_path)
            _save_partial(partial_path)
            dgp = watcher.load_dgp_curves(VERSION, dgp_path)
            progress = watcher.load_bound_progress(
                VERSION, partial_path=partial_path,
                final_path=os.path.join(directory, "missing-final.npz"))

            with mock.patch.object(
                    watcher.os, "replace", wraps=os.replace) as replace:
                watcher.write_progress_values(dgp, progress, output)

            self.assertTrue(os.path.isfile(output))
            source, destination = replace.call_args.args
            self.assertEqual(destination, output)
            self.assertEqual(os.path.dirname(source), directory)
            self.assertNotEqual(source, output)
            self.assertFalse(os.path.exists(source))
            with open(output, newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))

            expected_series = {
                "power_chi2", "power_c1", "power_cp1",
                "gkm_power_bound",
            }
            self.assertEqual({row["series"] for row in rows}, expected_series)
            self.assertEqual(len(rows), 20)
            self.assertTrue(all(
                row["dgp_run_signature"] == dgp.run_signature
                for row in rows))
            self.assertTrue(all(
                row["bound_run_signature"] == progress.run_signature
                for row in rows))
            self.assertTrue(all(
                row["synthetic_demo"] == "false" for row in rows))
            self.assertEqual(
                {row["scope"] for row in rows},
                {"finite_sample_dgp", "limit_experiment"})

            incomplete = [row for row in rows
                          if row["series"] == "gkm_power_bound"
                          and row["completed"] == "false"]
            self.assertTrue(incomplete)
            self.assertTrue(all(row["value"] == "" for row in incomplete))


class DemoTests(unittest.TestCase):
    def test_demo_is_synthetic_isolated_and_logs_six_snapshots(self):
        fake_run = types.SimpleNamespace(
            log=mock.Mock(), finish=mock.Mock())
        fake_wandb = types.SimpleNamespace(
            init=mock.Mock(return_value=fake_run),
            Image=mock.Mock(side_effect=lambda value: value),
        )
        plot_metrics = [
            {"completed_beta_count": completed}
            for completed in range(6)
        ]
        with mock.patch.dict(sys.modules, {"wandb": fake_wandb}), \
                mock.patch.object(
                    watcher, "load_dgp_curves",
                    side_effect=AssertionError("strict DGP loader called")), \
                mock.patch.object(
                    watcher, "load_bound_progress",
                    side_effect=AssertionError("strict bound loader called")), \
                mock.patch.object(
                    watcher, "write_progress_plot",
                    side_effect=plot_metrics) as write_plot, \
                mock.patch.object(
                    watcher, "write_progress_values") as write_values, \
                mock.patch("builtins.print"):
            watcher.main([
                "--version", VERSION,
                "--demo",
                "--demo-delay", "0",
                "--wandb-project", "demo-project",
            ])

        self.assertEqual(write_plot.call_count, 6)
        self.assertEqual(write_values.call_count, 6)
        self.assertEqual(fake_run.log.call_count, 6)
        self.assertEqual(
            [call.args[0]["completed_beta_count"]
             for call in fake_run.log.call_args_list],
            list(range(6)))
        self.assertEqual(fake_run.finish.call_count, 1)
        init = fake_wandb.init.call_args.kwargs
        self.assertEqual(init["project"], "demo-project")
        self.assertTrue(init["id"].startswith("gkm-demo-"))
        self.assertIn("[SYNTHETIC DEMO]", init["name"])
        self.assertTrue(init["config"]["synthetic_demo"])
        self.assertEqual(init["config"]["dgp_cache"], "<synthetic-demo>")

        expected_directory = os.path.join(VERSION, "gkm_direct", "demo")
        png_paths = [call.args[2] for call in write_plot.call_args_list]
        csv_paths = [call.args[2] for call in write_values.call_args_list]
        self.assertTrue(all(
            os.path.dirname(path) == expected_directory
            for path in png_paths + csv_paths))
        self.assertEqual(len(set(png_paths)), 1)
        self.assertEqual(len(set(csv_paths)), 1)
        self.assertTrue(os.path.basename(png_paths[0]).startswith(
            "live_power_progress_demo_"))
        self.assertEqual(
            os.path.splitext(png_paths[0])[0] + ".csv", csv_paths[0])

    def test_demo_rejects_real_checkpoint_and_cache_options(self):
        conflicts = [
            ("--dgp-cache", "dgp.npz"),
            ("--partial-path", "partial.npz"),
            ("--final-path", "final.npz"),
            ("--expected-run-signature", "a" * 64),
        ]
        for option, value in conflicts:
            with self.subTest(option=option), \
                    mock.patch("sys.stderr", new=io.StringIO()), \
                    mock.patch.object(watcher, "_run_synthetic_demo") as run:
                with self.assertRaises(SystemExit) as raised:
                    watcher.main([
                        "--version", VERSION, "--demo", option, value])
                self.assertEqual(raised.exception.code, 2)
                run.assert_not_called()


class WandbFailureTests(unittest.TestCase):
    def test_final_snapshot_retries_until_upload_succeeds(self):
        betas = np.array([-1.0, 0.0, 1.0])
        settings = dict(
            version_label=VERSION, kappas=KAPPAS.tolist(),
            k=K, n=N, alpha=ALPHA)
        dgp = watcher.DgpCurves(
            betas=betas,
            power_chi2=np.array([0.06, 0.05, 0.07]),
            power_c1=np.array([0.07, 0.05, 0.08]),
            power_cp1=np.array([0.08, 0.05, 0.09]),
            settings=settings, run_signature="d" * 64, path="dgp.npz")
        progress = watcher.BoundProgress(
            betas=betas, bounds=np.array([0.10, 0.05, 0.11]),
            bounds_se=np.array([0.01, 0.0, 0.01]),
            run_signature="b" * 64, source_path="bound.npz",
            is_final=True, settings=settings)
        logger = types.SimpleNamespace(
            requested=True, start=mock.Mock(),
            log=mock.Mock(side_effect=[False, True]), finish=mock.Mock())

        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(
                    watcher, "WandbLogger", return_value=logger), \
                mock.patch.object(
                    watcher, "load_dgp_curves", return_value=dgp), \
                mock.patch.object(
                    watcher, "load_bound_progress",
                    return_value=progress) as load_progress, \
                mock.patch.object(
                    watcher, "write_progress_plot",
                    return_value={"completed_beta_count": 3,
                                  "total_beta_count": 3,
                                  "below_c3_count": 0}), \
                mock.patch.object(watcher, "write_progress_values"), \
                mock.patch.object(watcher.time, "sleep"), \
                mock.patch("builtins.print"):
            watcher.main([
                "--version", VERSION,
                "--output", os.path.join(directory, "progress.png"),
                "--wandb-project", "test-project",
                "--poll-seconds", "0.001",
            ])

        self.assertEqual(load_progress.call_count, 2)
        self.assertEqual(logger.log.call_count, 2)
        logger.finish.assert_called_once()

    def test_upload_failure_does_not_prevent_local_png(self):
        class FailingRun:
            def __init__(self):
                self.finish = mock.Mock()

            def log(self, *_args, **_kwargs):
                raise RuntimeError("network down")

        run = FailingRun()
        fake_wandb = types.SimpleNamespace(
            init=mock.Mock(return_value=run),
            Image=lambda value: value,
        )
        with tempfile.TemporaryDirectory() as directory:
            dgp_path = os.path.join(directory, "dgp.npz")
            partial_path = os.path.join(directory, "partial.npz")
            output = os.path.join(directory, "live.png")
            _save_dgp_cache(dgp_path)
            _save_partial(partial_path)
            argv = [
                "--version", VERSION,
                "--dgp-cache", dgp_path,
                "--partial-path", partial_path,
                "--final-path", os.path.join(directory, "missing.npz"),
                "--output", output,
                "--once",
                "--wandb-project", "test-project",
            ]
            with mock.patch.dict(sys.modules, {"wandb": fake_wandb}), \
                    mock.patch("builtins.print") as printed:
                watcher.main(argv)

            self.assertTrue(os.path.isfile(output))
            self.assertTrue(fake_wandb.init.called)
            self.assertTrue(run.finish.called)
            messages = " ".join(
                " ".join(str(arg) for arg in call.args)
                for call in printed.call_args_list)
            self.assertIn("network down", messages)

    def test_transient_upload_failure_is_retried_at_next_update(self):
        class FlakyRun:
            def __init__(self):
                self.log_calls = 0

            def log(self, *_args, **_kwargs):
                self.log_calls += 1
                if self.log_calls == 1:
                    raise RuntimeError("temporary outage")

        run = FlakyRun()
        image = mock.Mock(side_effect=lambda value: value)
        logger = watcher.WandbLogger(project="test-project")
        logger.module = types.SimpleNamespace(Image=image)
        logger.run = run

        with mock.patch("builtins.print") as printed:
            self.assertFalse(logger.log("first.png", {"step": 1}))
            self.assertEqual(logger.upload_failures, 1)
            self.assertTrue(logger.log("second.png", {"step": 2}))

        self.assertEqual(run.log_calls, 2)
        self.assertEqual(image.call_count, 2)
        self.assertEqual(logger.upload_failures, 0)
        messages = " ".join(
            " ".join(str(arg) for arg in call.args)
            for call in printed.call_args_list)
        self.assertIn("next watcher poll will retry", messages)


if __name__ == "__main__":
    unittest.main()
