"""Contract tests for the sigma calibration driver (top-K recall, deciles, a
tiny end-to-end arm)."""

from __future__ import annotations

import argparse
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.reproduction import eval_sigma_calibration as esc


class RecallAndDecileTest(unittest.TestCase):
    def test_top_k_recall(self):
        mu = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([5.0, 1.0, 2.0, 3.0, 4.0])
        self.assertEqual(esc.top_k_recall(mu, y, 1), 0.0)   # best mu != best y
        self.assertEqual(esc.top_k_recall(mu, y, 2), 0.5)
        self.assertEqual(esc.top_k_recall(mu, y, 5), 1.0)
        self.assertEqual(esc.top_k_recall(y, y, 2), 1.0)    # oracle ordering

    def test_decile_errors(self):
        rng = np.random.RandomState(0)
        sigma = np.abs(rng.randn(200))
        err = np.abs(rng.randn(200))
        deciles = esc.decile_errors(sigma, err)
        self.assertEqual(len(deciles), 10)
        self.assertEqual(sum(d["n"] for d in deciles), 200)
        means = [d["mean_abs_error"] for d in deciles]
        self.assertTrue(all(np.isfinite(means)))

    def test_tiny_end_to_end_arm(self):
        a = argparse.Namespace(data_dir="data/proxy2000_v2", n_test=40,
                               n_members=3, screen_k=100)
        row = esc.run_seed_arm(a, seed=301, n_train=40)
        self.assertTrue(np.isfinite(row["spearman_mu"]))
        self.assertTrue(-1.0 <= row["spearman_sigma_error"] <= 1.0)
        self.assertTrue(-1.0 <= row["spearman_sigma_error_shuffled"] <= 1.0)
        self.assertGreater(row["seconds"], 0.0)
        for frac in (0.05, 0.10):
            self.assertGreaterEqual(row[f"recall_greedy_k{frac}"], 0.0)
            for beta in (0.5, 1.0, 2.0):
                self.assertTrue(np.isfinite(row[f"delta_recall_lcb{beta}_k{frac}"]))
        self.assertEqual(len(row["deciles"]), 10)


if __name__ == "__main__":
    unittest.main(verbosity=2)
