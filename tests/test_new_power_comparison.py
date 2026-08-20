import json
import os
import tempfile
import unittest

import numpy as np

import new_power_comparison as comparison


class DgpOnlyComparisonTests(unittest.TestCase):
    def setUp(self):
        self.version = '352515'
        self.kappas = np.array([35.0, 25.0, 15.0])
        self.betas = np.array([-0.2, 0.0, 0.2])
        self.curves = (
            np.array([0.04, 0.05, 0.06]),
            np.array([0.05, 0.05, 0.07]),
            np.array([0.06, 0.05, 0.08]),
        )

    def _save(self, path):
        comparison.save_dgp_cache(
            path, version_label=self.version, kappas=self.kappas,
            k=7, n=250, alpha=0.05, betas=self.betas,
            power_chi2=self.curves[0], power_c1=self.curves[1],
            power_cp1=self.curves[2], num_simulations=1000,
            base_seed=123, chunk_size=100, workers_used=2)

    def _load(self, path):
        return comparison.load_compatible_dgp_cache(
            path, version_label=self.version, kappas=self.kappas,
            k=7, n=250, alpha=0.05, betas=self.betas,
            num_simulations=1000, base_seed=123, chunk_size=100)

    @staticmethod
    def _resign_with_source_hash(path, source_hash):
        with np.load(path, allow_pickle=False) as archive:
            payload = {name: np.asarray(archive[name]).copy()
                       for name in archive.files}
        settings = json.loads(str(payload['settings_json']))
        settings['source_sha256'] = source_hash
        settings_json, signature = comparison._settings_json_and_signature(
            settings)
        payload['source_sha256'] = np.array(source_hash)
        payload['settings_json'] = np.array(settings_json)
        payload['run_signature'] = np.array(signature)
        np.savez(path, **payload)

    def test_audited_pre_cleanup_schema_one_cache_remains_compatible(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'dgp.npz')
            self._save(path)
            legacy_hash = next(iter(
                comparison._TRUSTED_PRE_DGP_ONLY_SOURCE_SHA256))
            self._resign_with_source_hash(path, legacy_hash)

            loaded = self._load(path)
            for actual, expected in zip(
                    loaded, (self.betas,) + self.curves):
                np.testing.assert_array_equal(actual, expected)

    def test_arbitrary_even_self_consistently_signed_source_hash_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'dgp.npz')
            self._save(path)
            self._resign_with_source_hash(path, '0' * 64)

            with self.assertRaisesRegex(
                    ValueError, 'audited DGP-equivalent predecessor'):
                self._load(path)

    def test_legacy_alfd_plotter_api_is_absent(self):
        for name in (
                'adaptive_alfd_path', 'load_compatible_alfd_bound',
                'ALFD_SCHEMA_VERSION', 'ALFD_ALGORITHM', 'ALFD_BOUND_KIND'):
            self.assertFalse(hasattr(comparison, name), name)


if __name__ == '__main__':
    unittest.main()
