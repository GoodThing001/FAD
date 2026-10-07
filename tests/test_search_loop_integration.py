"""Integration tests: the iterative search layer feeding the closed loop (plan step 5).

docs/项目记录/突变序列迭代搜索算法与代理接口实施方案_20261002.md §6 step 5:
search results become the per-round candidate pool; the ordinary select_batch /
Oracle / PARKED / ingest / retrain flow is unchanged.  Search candidates live
outside the canonical table, so search source is prospective-only by design.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.loop import loop as loop_mod
from scripts.loop.interface import MUT_POS, WT_141, Prediction, normalize
from scripts import run_experiment as runner

CANON = ROOT / "data/proxy2000_v2/fad_proxy2000_v2_full.csv"


def _args_for(tmpdir: Path, **overrides) -> list[str]:
    base = [
        "--mode", "prospective",
        "--candidate-source", "search",
        "--rounds", "1",
        "--budget", "10",
        "--n-init", "40",
        "--n-holdout", "20",
        "--pool-size", "30",
        "--surrogate", "xgb",
        "--n-members", "3",
        "--screen-k", "100",
        "--strategy", "greedy",
        "--oracle", "park",
        "--seed", "102",
        "--data-dir", "data/proxy2000_v2",
        "--search-policy", "multi_start_beam",
        "--search-max-unique-predictions", "64",
        "--search-max-generations", "4",
        "--search-allowed-n-from-wt", "4:13",
        "--search-beam-width", "4",
        "--search-offspring-per-parent", "4",
        "--search-one-hop-fraction", "0.8",
        "--search-restart-fraction", "0.1",
        "--search-diversity-pool-factor", "4",
        "--search-diversity-distance", "2",
        "--search-max-stagnant-generations", "2",
        "--search-n-measured-seeds", "2",
        "--search-n-random-seeds", "2",
        "--search-min-mutual-distance", "2",
        "--protocol-id", "gbsa_pending_protocol",
        "--output", str(tmpdir / "metrics.csv"),
        "--log-dir", str(tmpdir / "checkpoints"),
    ]
    for k, v in overrides.items():
        if v is None:                       # flag-style argument (e.g. --resume)
            base += [f"--{k}"]
        else:
            base += [f"--{k}", str(v)]
    return loop_mod.parse_args(base)


def _canonical_seqs() -> set[str]:
    return set(pd.read_csv(CANON)["Sequence"].map(normalize))


def _n_from_wt(seq: str) -> int:
    return sum(1 for p in MUT_POS if seq[p - 1] != WT_141[p - 1])


class SearchLoopIntegrationTest(unittest.TestCase):
    def test_group_batch_quota_one_group_and_shortage(self):
        seqs = pd.read_csv(CANON)["Sequence"].map(normalize).tolist()[:4]
        pred = Prediction(mu=np.array([4., 3., 2., 1.]))
        kw = dict(budget=3, strategy="greedy", labeled_seqs=[],
                  ratios=[1, 1, 1, 1], direction="minimize", seed=1,
                  diverse_multiplier=5)
        one = loop_mod.select_search_group_batch(
            seqs, pred, ["g1"] * 4, **kw)
        self.assertEqual(one["group_quotas"], {"g1": 3})
        self.assertEqual(one["indices"], [1, 2, 3])
        two = loop_mod.select_search_group_batch(
            seqs, pred, ["g1", "g2", "g2", "g2"], **kw)
        self.assertEqual(two["group_quotas"], {"g1": 1, "g2": 2})
        self.assertEqual(len(two["indices"]), 3)

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self._tmp.name)
        self.ckpt = self.tmpdir / "checkpoints"

    def tearDown(self):
        self._tmp.cleanup()

    def test_search_source_is_prospective_only(self):
        with self.assertRaises(ValueError) as cm:
            loop_mod.ClosedLoop(_args_for(self.tmpdir, mode="retrospective"))
        self.assertIn("prospective-only", str(cm.exception))

    def test_search_park_flow_and_archive(self):
        args = _args_for(self.tmpdir)
        loop = loop_mod.ClosedLoop(args)
        rows, code = loop.run()
        self.assertEqual(code, 3)                     # PARKED
        self.assertTrue((self.ckpt / "PARKED").is_file())
        self.assertEqual(len(rows), 1)
        self.assertEqual(len(loop.pool), 30)          # capped at pool-size

        ledger = pd.read_csv(self.ckpt / "ledger.csv")
        batch = pd.read_csv(self.ckpt / "round_001_batch.csv")
        self.assertEqual(len(ledger), 10)
        self.assertEqual(len(batch), 10)
        canon = _canonical_seqs()
        self.assertTrue(set(batch["sequence"]).isdisjoint(canon))
        for s in batch["sequence"]:
            self.assertTrue(4 <= _n_from_wt(s) <= 13)
        self.assertEqual(set(ledger["source"]), {"search_pool"})
        self.assertEqual(list(ledger["status"]), ["awaiting_labels"] * 10)

        arch = self.ckpt / "round_001_search"
        summary = json.loads((arch / "search_summary.json").read_text("utf-8"))
        state = json.loads((arch / "search_state.json").read_text("utf-8"))
        self.assertTrue(state["done"])
        self.assertEqual(summary["policy"], "multi_start_beam")
        self.assertGreaterEqual(summary["n_candidates"], 30)
        # every exported batch sequence is one of the ranked search candidates
        cands = pd.read_csv(arch / "search_candidates.csv")
        self.assertTrue(set(batch["sequence"]).issubset(set(cands["sequence"])))
        groups = json.loads((arch / "search_groups.json").read_text("utf-8"))
        self.assertEqual(set(cands["candidate_id"]),
                         {cid for g in groups for cid in g["member_ids"]})
        self.assertEqual(set(batch["group_id"]),
                         {g["group_id"] for g in groups})
        self.assertEqual(sorted(batch["group_id"].value_counts().tolist()), [5, 5])
        evaluated = {r["candidate_id"]: r for r in
                     (json.loads(line) for line in (arch / "search_evaluated.jsonl")
                      .read_text("utf-8").splitlines())}
        for r in batch.to_dict("records"):
            parent = evaluated.get(r["parent_candidate_id"])
            self.assertIsNotNone(parent)
            self.assertEqual(r["parent_sequence"], parent["sequence"])
            changed = [p for p in MUT_POS if r["sequence"][p-1] !=
                       parent["sequence"][p-1]]
            self.assertEqual(r["n_from_parent"], len(changed))
            self.assertEqual(r["changed_positions_from_parent"],
                             ",".join(map(str, changed)))
        manifest = json.loads((self.ckpt / "round_001_batch_manifest.json")
                              .read_text("utf-8"))
        self.assertEqual(sum(manifest["search_handoff"]["batch_group_counts"].values()), 10)
        handoff = pd.read_csv(self.ckpt / "pending_batches" / "batches"
                              / f"batch_{batch['batch_id'].iloc[0]}"
                              / "batch_handoff.csv")
        self.assertEqual(list(handoff["group_id"]), list(batch["group_id"]))
        self.assertEqual(list(handoff["parent_sequence"]),
                         list(batch["parent_sequence"]))
        # pool provenance + metrics columns
        self.assertEqual(loop.pool_stats["source"], "search")
        row = rows.iloc[0]
        self.assertEqual(row["search_stop_reason"], summary["stop_reason"])
        self.assertEqual(row["search_pool_size"], 30)
        self.assertEqual(row["search_n_unique_predicted"], summary["n_unique_predicted"])

    def test_search_handoff_verifier_detects_lineage_tamper(self):
        self.assertEqual(loop_mod.ClosedLoop(_args_for(self.tmpdir)).run()[1], 3)
        (self.tmpdir / "PARKED").write_text(json.dumps({
            "return_code": 3, "metrics_rows": 1, "acceptance_ok": True}),
            encoding="utf-8")
        cmd = [sys.executable, str(ROOT / "tools/verify_loop_run.py"),
               str(self.tmpdir)]
        good = subprocess.run(cmd, capture_output=True, text=True,
                              cwd=ROOT, encoding="utf-8", errors="replace")
        self.assertEqual(good.returncode, 0, good.stdout + good.stderr)
        ledger_path = self.ckpt / "ledger.csv"
        ledger = pd.read_csv(ledger_path)
        ledger.loc[0, "parent_sequence"] = WT_141
        ledger.to_csv(ledger_path, index=False)
        bad = subprocess.run(cmd, capture_output=True, text=True,
                             cwd=ROOT, encoding="utf-8", errors="replace")
        self.assertEqual(bad.returncode, 1)
        self.assertIn("search lineage/group mismatch", bad.stdout)

    def test_protocol_pin_updates_oracle_search_handoff(self):
        self.assertEqual(loop_mod.ClosedLoop(_args_for(self.tmpdir)).run()[1], 3)
        (self.tmpdir / "PARKED").write_text(json.dumps({"return_code": 3}),
                                             encoding="utf-8")
        (self.tmpdir / "config.json").write_text(json.dumps({
            "arguments": {"protocol-id": "gbsa_pending_protocol",
                          "approved-protocols": ""}}), encoding="utf-8")
        result = runner.pin_protocol(self.tmpdir, "team_gbsa_v1")
        self.assertTrue(result["changed"])
        batch = pd.read_csv(self.ckpt / "round_001_batch.csv")
        handoff = pd.read_csv(self.ckpt / "pending_batches" / "batches"
                              / f"batch_{batch['batch_id'].iloc[0]}"
                              / "batch_handoff.csv")
        self.assertEqual(set(handoff["protocol_id"]), {"team_gbsa_v1"})
        self.assertEqual(list(handoff["group_id"]), list(batch["group_id"]))

    def test_search_resume_is_idempotent(self):
        code1 = loop_mod.ClosedLoop(_args_for(self.tmpdir)).run()[1]
        self.assertEqual(code1, 3)
        ledger1 = pd.read_csv(self.ckpt / "ledger.csv")
        batch1 = pd.read_csv(self.ckpt / "round_001_batch.csv")

        rows2, code2 = loop_mod.ClosedLoop(_args_for(self.tmpdir, resume=None)).run()
        self.assertEqual(code2, 3)
        self.assertEqual(len(rows2), 1)               # no duplicated metric row
        ledger2 = pd.read_csv(self.ckpt / "ledger.csv")
        batch2 = pd.read_csv(self.ckpt / "round_001_batch.csv")
        self.assertEqual(len(ledger2), len(ledger1))
        self.assertEqual(list(batch2["batch_id"]), list(batch1["batch_id"]))
        self.assertEqual(list(batch2["sequence"]), list(batch1["sequence"]))
        self.assertEqual(list(batch2["pred_gbsa"]), list(batch1["pred_gbsa"]))
        batch_dirs = list((self.ckpt / "pending_batches" / "batches").glob("batch_*"))
        self.assertEqual(len(batch_dirs), 1)

    def test_search_config_change_refused_on_resume(self):
        self.assertEqual(loop_mod.ClosedLoop(_args_for(self.tmpdir)).run()[1], 3)
        bad = _args_for(self.tmpdir, resume=None, **{"search-beam-width": "16"})
        with self.assertRaises(ValueError) as cm:
            loop_mod.ClosedLoop(bad).run()
        self.assertIn("fingerprint", str(cm.exception))

    def test_search_two_rounds_ingest_retrain_and_new_pool(self):
        args = _args_for(self.tmpdir, rounds="2")
        self.assertEqual(loop_mod.ClosedLoop(args).run()[1], 3)   # round 1 parked

        batch1 = pd.read_csv(self.ckpt / "round_001_batch.csv")
        ingest_rows = [{
            "batch_id": r["batch_id"], "candidate_id": r["candidate_id"],
            "sequence": r["sequence"], "gbsa": -5.0 + 0.1 * i,
            "measurement_status": "pass", "protocol_id": "test_proto",
        } for i, r in batch1.iterrows()]
        ingest_csv = self.tmpdir / "labels.csv"
        pd.DataFrame(ingest_rows).to_csv(ingest_csv, index=False)

        rows2, code2 = loop_mod.ClosedLoop(
            _args_for(self.tmpdir, rounds="2", resume=None,
                      ingest=str(ingest_csv))).run()
        self.assertEqual(code2, 3)                     # round 2 parked
        snap2 = json.loads((self.ckpt / "round_002_search" / "search_snapshot.json")
                           .read_text("utf-8"))
        self.assertEqual(snap2["measured_count"], 50)  # 40 init + 10 ingested
        ledger2 = pd.read_csv(self.ckpt / "ledger.csv")
        self.assertEqual(len(ledger2), 20)
        self.assertEqual(int(ledger2["measured_gbsa"].notna().sum()), 10)
        batch2 = pd.read_csv(self.ckpt / "round_002_batch.csv")
        self.assertTrue(set(batch2["sequence"]).isdisjoint(set(batch1["sequence"])))
        self.assertTrue(set(batch2["sequence"]).isdisjoint(_canonical_seqs()))
        # round-2 search saw the grown labelled set and produced a fresh pool;
        # the ingested labels show up as the round-2 pre-round labelled size
        rows_with_round2 = [r for _, r in rows2.iterrows() if r.get("round") == 2]
        self.assertTrue(rows_with_round2)
        self.assertEqual(int(rows_with_round2[-1]["n_labeled_before"]), 50)


if __name__ == "__main__":
    unittest.main(verbosity=2)
