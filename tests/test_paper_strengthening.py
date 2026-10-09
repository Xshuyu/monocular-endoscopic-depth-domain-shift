"""Regression checks for seed isolation and measurement-summary semantics."""
import hashlib
import sys
import unittest
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT.parent))
from lesion_3d_measurement.src.ground_truth_benchmark import generate_benchmark_cohort
from lesion_3d_measurement.tests.paper_strengthening_benchmark import metrics, seed_mean_ci


def fingerprint(samples):
    h=hashlib.sha256()
    for sample in samples:
        for field in ('rgb_image','ground_truth_depth_mm','lesion_mask','tool_mask'):
            h.update(getattr(sample,field).tobytes())
    return h.hexdigest()


class ProtocolTests(unittest.TestCase):
    def test_seed_isolation_and_repetition(self):
        np.random.seed(171)
        before=np.random.get_state()
        a=generate_benchmark_cohort(4,seed=3100)
        after=np.random.get_state()
        self.assertEqual(before[0],after[0])
        np.testing.assert_array_equal(before[1],after[1])
        self.assertEqual(before[2:],after[2:])
        self.assertEqual(fingerprint(a),fingerprint(generate_benchmark_cohort(4,seed=3100)))
        self.assertNotEqual(fingerprint(a),fingerprint(generate_benchmark_cohort(4,seed=3101)))

    def test_default_preserves_legacy_fixture(self):
        # Fingerprint measured from the archived pre-change generator in this environment.
        expected='69c8be2f67d61b181c8b06ae3767ca65a08da82dbe09a37bbfe039808a690b11'
        self.assertEqual(fingerprint(generate_benchmark_cohort(4)),expected)

    def test_measurement_agreement_is_not_correlation(self):
        truth=np.array([3.,7.,14.,25.])
        rows=[{'true_major_mm':float(t),'predicted_major_mm':float(t+2)} for t in truth]
        result=metrics(rows)
        self.assertAlmostEqual(result['mae_mm'],2.)
        self.assertAlmostEqual(result['rmse_mm'],2.)
        self.assertEqual(result['bland_altman_loa_mm'],[2.,2.])
        self.assertLess(result['concordance_ccc'],1.)
        self.assertAlmostEqual(result['predictive_r2'],1-16/np.sum((truth-truth.mean())**2))
        self.assertEqual(result['size_bin_confusion_true_rows_pred_columns'],
                         [[0,1,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
        self.assertEqual(result['size_bin_agreement_pct'],75.)

    def test_seed_block_interval_constant_effect(self):
        self.assertEqual(seed_mean_ci([2.]*10,42,1000),[2.,2.])
        self.assertEqual(seed_mean_ci([1.,2.,3.],42,1000),seed_mean_ci([1.,2.,3.],42,1000))

if __name__=='__main__': unittest.main()
