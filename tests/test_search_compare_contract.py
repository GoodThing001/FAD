"""Contract tests for the search-policy comparison driver (plan §5 阶段 B).

Covers the summary math (paired deltas, bootstrap CI, Jaccard, completeness,
diversity) and the search_run --model-seed separation (same split, different
surrogate seed).
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.loop import search_compare as sc
from scripts.loop import search_run as search_run_mod


class DriverMathTest(unittest.TestCase):
    def test_actual_prediction_prefix_and_shared_start(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            policies = sc.MATCHED_POLICIES
            values = {policies[0]: [7., 6., 5., -100.],
                      policies[1]: [7., 6., 4.],
                      policies[2]: [7., 6., 3., 2., -200.],
                      policies[3]: [7., 6., 2.5, -300.]}
            values.update({p: [7., 6., 2.25, -400.] for p in policies[4:]})
            for policy, preds in values.items():
                arch = root / f"seed1_{policy}_A" / "checkpoints"
                arch.mkdir(parents=True)
                rows = [{"candidate_id": f"id{i}", "sequence": f"seq{i}",
                         "parent_evidence": "measured" if i == 0 else "none",
                         "operator": "seed" if i == 0 else "one_hop",
                         "pred_gbsa": v} for i, v in enumerate(preds)]
                (arch / "search_evaluated.jsonl").write_text(
                    "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
                (arch / "search_summary.json").write_text(json.dumps({
                    "n_unique_predicted": len(rows)}), encoding="utf-8")
                (arch / "search_snapshot.json").write_text(json.dumps({
                    "data_sha256": "data", "measured_sha256": "measured",
                    "forbidden_sha256": "forbidden", "allowed_n_from_wt": [4, 13],
                    "max_unique_predictions": 5,
                    "model_snapshot": {"model_id": "m", "feature_version": "f"}}),
                    encoding="utf-8")
            aligned, summary = sc.aligned_prediction_comparison(root, [1], policies)
            self.assertEqual(set(aligned["common_actual_predictions"]), {3})
            self.assertEqual(dict(zip(aligned["policy"], aligned[
                "best_pred_gbsa_at_common_n"])),
                {p: preds[2] for p, preds in values.items()})
            self.assertTrue(summary["initial_seeds_identical"])
            path = root / f"seed1_{policies[2]}_A" / "checkpoints" \
                / "search_evaluated.jsonl"
            rows = [json.loads(line) for line in path.read_text("utf-8").splitlines()]
            rows[0]["sequence"] = "different_seed"
            path.write_text("".join(json.dumps(r) + "\n" for r in rows),
                            encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "initial seed"):
                sc.aligned_prediction_comparison(root, [1], policies)
            rows[0]["sequence"] = "seq0"
            path.write_text("".join(json.dumps(r) + "\n" for r in rows),
                            encoding="utf-8")
            snapshot_path = path.parent / "search_snapshot.json"
            snapshot = json.loads(snapshot_path.read_text("utf-8"))
            snapshot["model_snapshot"]["model_id"] = "different_model"
            snapshot_path.write_text(json.dumps(snapshot), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "split/model/constraints"):
                sc.aligned_prediction_comparison(root, [1], policies)

    def test_bootstrap_ci_basic(self):
        vals = np.arange(1.0, 31.0)
        lo, hi = sc.bootstrap_ci(vals, n_boot=2000)
        self.assertLessEqual(lo, np.mean(vals))
        self.assertLessEqual(np.mean(vals), hi)
        lo2, hi2 = sc.bootstrap_ci(vals, n_boot=2000)
        self.assertEqual((lo, hi), (lo2, hi2))       # deterministic
        self.assertTrue(np.isnan(sc.bootstrap_ci(np.array([]))[0]))

    def test_jaccard(self):
        self.assertEqual(sc.jaccard({"a", "b"}, {"b", "c"}), 1 / 3)
        self.assertEqual(sc.jaccard({"a"}, {"a"}), 1.0)
        self.assertTrue(np.isnan(sc.jaccard(set(), set())))

    def test_paired_delta_math(self):
        rows = []
        for seed in range(1, 11):
            rows.append({"seed": seed, "policy": "stratified_random",
                         "arm": "A", "best_pred_gbsa": float(seed),
                         "diversity_top50": 4.0, "ood_fraction_top50": 0.2,
                         "mean_min_train_hamming_top50": 5.0,
                         "n_unique_predicted": 100})
            rows.append({"seed": seed, "policy": "multi_start_beam",
                         "arm": "A", "best_pred_gbsa": float(seed) - 1.0,
                         "diversity_top50": 6.0, "ood_fraction_top50": 0.1,
                         "mean_min_train_hamming_top50": 4.0,
                         "n_unique_predicted": 90})
        df = pd.DataFrame(rows)
        d = sc.paired_delta(df, "multi_start_beam", "best_pred_gbsa", "lower")
        self.assertEqual(d["delta_mean"], -1.0)
        self.assertEqual(d["wins"], 10)
        self.assertEqual((d["ties"], d["losses"]), (0, 0))
        d2 = sc.paired_delta(df, "multi_start_beam", "diversity_top50", "higher")
        self.assertEqual(d2["delta_mean"], 2.0)
        self.assertEqual(d2["wins"], 10)

    def test_completeness_detects_missing_rows(self):
        rows = []
        for seed in (1, 2):
            for policy in sc.POLICIES:
                rows.append({"seed": seed, "policy": policy, "arm": "A",
                             "best_pred_gbsa": 1.0, "n_unique_predicted": 10,
                             "diversity_top50": 3.0})
        df = pd.DataFrame(rows)
        ok = sc.check_completeness(df, [1, 2], [], sc.POLICIES)
        self.assertTrue(ok["ok"])
        bad = sc.check_completeness(df.iloc[:-1], [1, 2], [], sc.POLICIES)
        self.assertFalse(bad["ok"])
        self.assertTrue(any("row count" in p for p in bad["problems"]))

    def test_top50_diversity(self):
        rows = []
        wt = "UAUCCUUCGGGGCAGGGUGGAAAUCCCGACCGGCGGUAGUAAAGCACAUUUGCUUUAGAGCCC" \
             "GUGACCCGUGUGCAUAAGCACGCGGUGGAUUCAGUUUAAGCUGAAGCCGACAGUGAAAGUCUGGA" \
             "UGGGAGAAGGAUG"
        for i in range(3):
            s = list(wt)
            if i == 1:
                s[29] = "G"                    # site 30 mutated
            elif i == 2:
                s[30] = "G"                    # site 31 mutated
            rows.append({"sequence": "".join(s), "pred_gbsa": float(i)})
        d = sc.top50_diversity(pd.DataFrame(rows))
        self.assertEqual(d, 4 / 3)                    # mean of (1,1,2)


class ModelSeedSeparationTest(unittest.TestCase):
    """--model-seed changes the surrogate while the split stays fixed."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _run(self, model_seed: str | None) -> dict:
        argv = [
            "--policy", "stratified_random", "--seed", "201",
            "--max-unique-predictions", "32", "--max-generations", "4",
            "--allowed-n-from-wt", "4:13", "--n-init", "40",
            "--surrogate", "xgb", "--n-members", "3", "--screen-k", "100",
            "--data-dir", "data/proxy2000_v2",
            "--output", str(self.tmpdir / f"metrics_{model_seed or 'a'}.csv"),
            "--log-dir", str(self.tmpdir / f"arch_{model_seed or 'a'}"),
        ] + (["--model-seed", model_seed] if model_seed else [])
        self.assertEqual(search_run_mod.main(argv=argv), 0)
        ckpt = self.tmpdir / f"arch_{model_seed or 'a'}"
        snap = json.loads((ckpt / "search_snapshot.json").read_text("utf-8"))
        cands = pd.read_csv(ckpt / "search_candidates.csv")
        metrics = pd.read_csv(self.tmpdir / f"metrics_{model_seed or 'a'}.csv")
        return {"measured_sha256": snap["measured_sha256"],
                "preds": cands["pred_gbsa"].tolist(),
                "model_seed": int(metrics["model_seed"].iloc[0]),
                "sequences": cands["sequence"].tolist()}

    def test_model_seed_changes_predictions_not_split(self):
        a_run = self._run(None)
        b_run = self._run("1201")
        self.assertEqual(a_run["model_seed"], 201)
        self.assertEqual(b_run["model_seed"], 1201)
        # identical split -> identical measured set digest
        self.assertEqual(a_run["measured_sha256"], b_run["measured_sha256"])
        # identical generation RNG -> identical candidate SET (order differs
        # because the ranking is by the surrogate's predictions)
        self.assertEqual(sorted(a_run["sequences"]), sorted(b_run["sequences"]))
        # different surrogate -> different predictions somewhere
        self.assertNotEqual(a_run["preds"], b_run["preds"])

    def test_rejected_resume_preserves_fitted_weights(self):
        self._run(None)
        weights = self.tmpdir / "arch_a" / "fitted_surrogate.pkl"
        before = weights.read_bytes()
        argv = ["--policy", "stratified_random", "--seed", "201",
                "--model-seed", "1201", "--max-unique-predictions", "32",
                "--max-generations", "4", "--allowed-n-from-wt", "4:13",
                "--n-init", "40", "--surrogate", "xgb", "--n-members", "3",
                "--screen-k", "100", "--data-dir", "data/proxy2000_v2",
                "--output", str(self.tmpdir / "metrics_a.csv"),
                "--log-dir", str(self.tmpdir / "arch_a")]
        with self.assertRaisesRegex(ValueError, "snapshot"):
            search_run_mod.main(argv=argv)
        self.assertEqual(weights.read_bytes(), before)


if __name__ == "__main__":
    unittest.main(verbosity=2)
