"""Interface-contract and acceptance tests for the closed loop (interface v0.2.0).

Covers §11.1 of docs/项目记录/GBSA代理接口与候选闭环实施方案_20260930.md:
cross-round uniqueness, selection boundaries, minimisation direction, prediction
contract, truth isolation, metric time point, park/resume, bad ingest, and the
runner's source/PARKED semantics.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.loop.interface import (INTERFACE_VERSION, MUT_POS, PREDICTION_COLUMN,
                                    Prediction, SEQ_LEN, TARGET_COLUMN, TargetSpec,
                                    WT_141, candidate_id, normalize, validate_sequences)
from scripts.loop.mutate import (CandidatePool, generate_candidates,
                                 mutate_from_parent)
from scripts.loop.oracle import ParkOracle, TableOracle
from scripts.loop.selection import hamming_novelty, select_batch
from scripts.loop.surrogates import XGBEnsembleSurrogate, make_surrogate


def _pool(n=120, seed=3, exclude=()):
    return generate_candidates(n, max_mutations=6, seed=seed, exclude=exclude)


def _csv(path: Path, rows: pd.DataFrame) -> Path:
    rows.to_csv(path, index=False)
    return path


class InterfaceContractTest(unittest.TestCase):
    def test_version_and_target_policy(self):
        self.assertEqual(INTERFACE_VERSION, "0.2.0")
        spec = TargetSpec()
        self.assertEqual(spec.name, TARGET_COLUMN)
        self.assertEqual(spec.label_policy, "gbsa_only")
        self.assertFalse(spec.weighted_label_used)
        self.assertEqual(spec.direction, "minimize")
        self.assertEqual(spec.report_column, "pred_gbsa")

    def test_sequence_contract_includes_fixed_scaffold(self):
        self.assertEqual(validate_sequences([WT_141])[0], WT_141)
        with self.assertRaises(ValueError):
            validate_sequences([WT_141[:-1]])
        with self.assertRaises(ValueError):
            validate_sequences([WT_141[:100] + "N" * 41])
        broken = list(WT_141)          # change a position OUTSIDE the 13 sites
        broken[0] = "A" if WT_141[0] != "A" else "C"
        with self.assertRaises(ValueError):
            validate_sequences(["".join(broken)])

    def test_prediction_contract(self):
        p = Prediction(mu=np.zeros(3), sigma=np.ones(3),
                       uncertainty_kind="ensemble_spread")
        self.assertEqual(len(p), 3)
        self.assertTrue(p.has_sigma)
        # sigma is optional and must never be faked with zeros
        p2 = Prediction(mu=np.zeros(3))
        self.assertFalse(p2.has_sigma)
        with self.assertRaises(ValueError):
            Prediction(mu=np.zeros(3), sigma=np.zeros(3))       # missing kind
        with self.assertRaises(ValueError):
            Prediction(mu=np.array([1.0, np.nan, 2.0]))
        with self.assertRaises(ValueError):
            Prediction(mu=np.zeros(3), sigma=np.array([1.0, np.inf, 1.0]),
                       uncertainty_kind="ensemble_spread")
        with self.assertRaises(ValueError):
            Prediction(mu=np.zeros(3), sigma=-np.ones(3),
                       uncertainty_kind="ensemble_spread")

    def test_candidate_id_is_stable(self):
        seq = _pool(1, seed=1).sequences[0]
        self.assertEqual(candidate_id(seq), candidate_id(seq))
        self.assertNotEqual(candidate_id(seq), candidate_id(WT_141))


class MutationTest(unittest.TestCase):
    def test_generation_contract_and_provenance(self):
        pool = _pool(60, seed=0, )
        self.assertEqual(len(pool), 60)
        self.assertEqual(len(set(pool.sequences)), 60)
        for s, prov in zip(pool.sequences, pool.provenance):
            self.assertEqual(len(s), SEQ_LEN)
            validate_sequences([s])
            self.assertEqual(prov["candidate_id"], candidate_id(s))
            self.assertEqual(prov["n_from_wt"], len(prov["changed_positions_from_parent"]))
            self.assertEqual(prov["parent_sequence"], WT_141)
            self.assertEqual(prov["constraints_version"], "v1")

    def test_quota_stratification(self):
        pool = generate_candidates(quotas={2: 5, 8: 7, 11: 3}, seed=2)
        hist = pool.mutation_histogram()
        self.assertEqual(hist.get(2), 5)
        self.assertEqual(hist.get(8), 7)
        self.assertEqual(hist.get(11), 3)

    def test_exclude_respected(self):
        first = _pool(20, seed=1)
        second = _pool(20, seed=1, exclude=first.sequences)
        self.assertFalse(set(first.sequences) & set(second.sequences))

    def test_local_mutation_counts_parent_and_wt(self):
        parent = _pool(1, seed=7).sequences[0]
        local = mutate_from_parent(parent, 5, n_mutations=1, seed=3)
        for s, prov in zip(local.sequences, local.provenance):
            self.assertEqual(prov["n_from_parent"], 1)
            self.assertEqual(prov["parent_sequence"], normalize(parent))
            self.assertGreaterEqual(prov["n_from_wt"], 0)


class OodRuleTest(unittest.TestCase):
    """§7.1 OOD rule must not degenerate to 'never OOD'."""

    def test_loo_threshold_is_finite_and_flags_far_candidates(self):
        from scripts.loop.loop import compute_ood
        from scripts.loop.selection import _mutation_matrix, loo_min_distance

        train = _pool(40, seed=13).sequences
        T = _mutation_matrix(train)
        d = loo_min_distance(T)
        self.assertTrue(np.isfinite(d).all())          # never the self-distance 0
        self.assertTrue((d > 0).all())

        # a training sequence itself is inside the training hull
        flags_same = compute_ood(train[:5], train)
        self.assertFalse(flags_same.any())
        # candidates far outside both the n_from_wt range and the neighbourhood
        far = _pool(400, seed=99).sequences
        flags_far = compute_ood(far, train)
        self.assertGreater(flags_far.mean(), 0.0)
        self.assertLessEqual(flags_far.mean(), 1.0)


class SelectionTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(0)
        self.pool = _pool(120, seed=3).sequences
        self.labeled = self.pool[:20]
        self.mu = rng.normal(0, 5, len(self.pool))
        self.sigma = np.abs(rng.normal(1, 0.3, len(self.pool)))
        self.pred_sigma = Prediction(mu=self.mu, sigma=self.sigma,
                                     uncertainty_kind="ensemble_spread")
        self.pred_plain = Prediction(mu=self.mu)

    def test_minimisation_direction(self):
        mu = np.array([-10.0, 0.0, 10.0])
        seqs = _pool(3, seed=11).sequences
        sel = select_batch(seqs, Prediction(mu=mu), budget=1, strategy="greedy",
                           direction="minimize")
        self.assertEqual(sel["indices"], [0])
        sel_max = select_batch(seqs, Prediction(mu=mu), budget=1, strategy="greedy",
                               direction="maximize")
        self.assertEqual(sel_max["indices"], [2])

    def test_exclusions_and_dedup(self):
        sel = select_batch(self.pool, self.pred_sigma, budget=30, strategy="greedy",
                           labeled_seqs=self.labeled, seed=1)
        idx = sel["indices"]
        self.assertEqual(len(idx), 30)
        self.assertEqual(len(set(idx)), 30)
        for i in idx:                                # never re-offer labelled rows
            self.assertNotIn(self.pool[i], set(self.labeled))

    def test_zero_budget_and_zero_quota(self):
        self.assertEqual(select_batch(self.pool, self.pred_sigma, budget=0,
                                      strategy="greedy")["indices"], [])
        sel = select_batch(self.pool, self.pred_sigma, budget=10,
                           strategy="mixed_control", ratios=(0, 0, 100, 0),
                           seed=1)
        roles = set(sel["roles"].values())
        self.assertTrue(roles <= {"promising", "deficit_fill"})
        with self.assertRaises(ValueError):
            select_batch(self.pool, self.pred_sigma, budget=10, strategy="greedy",
                         ratios=(45, 25, 20))

    def test_negative_inputs_rejected(self):
        with self.assertRaises(ValueError):
            select_batch(self.pool, self.pred_sigma, budget=-1, strategy="greedy")
        with self.assertRaises(ValueError):
            select_batch(self.pool, self.pred_sigma, budget=10, strategy="mixed_control",
                         ratios=(-1, 1, 1, 1))

    def test_pool_exhaustion_reported(self):
        sel = select_batch(self.pool[:5], Prediction(mu=np.zeros(5)), budget=50,
                           strategy="greedy")
        self.assertEqual(sel["effective_budget"], 5)
        self.assertIn("pool exhausted", sel["note"])

    def test_uncertainty_strategy_requires_sigma(self):
        with self.assertRaises(ValueError):
            select_batch(self.pool, self.pred_plain, budget=10,
                         strategy="uncertainty_mixed")
        # plain greedy/random must still work without sigma
        for strategy in ("greedy", "random", "greedy_diverse", "maxmin_coverage"):
            sel = select_batch(self.pool, self.pred_plain, budget=5, strategy=strategy)
            self.assertEqual(len(sel["indices"]), 5)

    def test_determinism_and_novelty(self):
        a = select_batch(self.pool, self.pred_sigma, budget=25, strategy="mixed_control",
                         labeled_seqs=self.labeled, seed=7)["indices"]
        b = select_batch(self.pool, self.pred_sigma, budget=25, strategy="mixed_control",
                         labeled_seqs=self.labeled, seed=7)["indices"]
        self.assertEqual(a, b)
        nov = hamming_novelty(self.pool, self.labeled)
        self.assertTrue(np.all(nov[:20] == 0))

    def test_chunked_hamming_is_chunk_invariant(self):
        """C package: chunked Hamming must not depend on the block size."""
        from scripts.loop.selection import _mutation_matrix, loo_min_distance
        ref = {
            1: hamming_novelty(self.pool, self.labeled, chunk=1),
            3: hamming_novelty(self.pool, self.labeled, chunk=3),
            7: hamming_novelty(self.pool, self.labeled, chunk=7),
            512: hamming_novelty(self.pool, self.labeled, chunk=512),
        }
        base = ref[512]
        for chunk, values in ref.items():
            np.testing.assert_allclose(values, base, err_msg=f"chunk={chunk} differs")
        self.assertTrue((base > 0).any())              # distances are actually computed

        T = _mutation_matrix(self.pool[:40])
        for chunk in (1, 4, 40):
            np.testing.assert_allclose(loo_min_distance(T, chunk=chunk),
                                       loo_min_distance(T, chunk=40))

    def test_greedy_diverse_beats_greedy_on_spread(self):
        sel_g = select_batch(self.pool, self.pred_sigma, budget=10, strategy="greedy",
                             labeled_seqs=self.labeled, seed=1)
        sel_d = select_batch(self.pool, self.pred_sigma, budget=10,
                             strategy="greedy_diverse", labeled_seqs=self.labeled,
                             seed=1)
        labeled = list(self.labeled)

        def mean_min_dist(idx):
            d = hamming_novelty([self.pool[i] for i in idx], labeled)
            return float(np.mean(d))

        self.assertGreaterEqual(mean_min_dist(sel_d["indices"]),
                                mean_min_dist(sel_g["indices"]))


class SurrogateContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.RandomState(5)
        cls.train = _pool(120, seed=11).sequences
        cls.y = rng.normal(30, 8, len(cls.train))
        cls.test = _pool(15, seed=99).sequences

    def test_fit_predict_and_metadata(self):
        model = make_surrogate("xgb", n_members=3, seed=1)
        model.fit(self.train, self.y)
        pred = model.predict(self.test)
        self.assertEqual(len(pred), len(self.test))
        self.assertTrue(np.isfinite(pred.mu).all())
        self.assertTrue(pred.has_sigma and (pred.sigma >= 0).all())
        self.assertEqual(pred.uncertainty_kind, "ensemble_spread_uncalibrated")
        self.assertTrue(pred.model_id)
        self.assertEqual(pred.feature_version, "seq303")

    def test_sigma_can_be_disabled(self):
        model = XGBEnsembleSurrogate(n_members=2, seed=2, with_uncertainty=False)
        model.fit(self.train, self.y)
        self.assertFalse(model.predict(self.test).has_sigma)

    def test_determinism_and_mismatch(self):
        m1 = XGBEnsembleSurrogate(n_members=2, seed=3).fit(self.train, self.y)
        m2 = XGBEnsembleSurrogate(n_members=2, seed=3).fit(self.train, self.y)
        np.testing.assert_allclose(m1.predict(self.test).mu,
                                   m2.predict(self.test).mu, rtol=1e-9)
        with self.assertRaises(ValueError):
            XGBEnsembleSurrogate(n_members=2).fit(self.train, self.y[:-1])


class TruthIsolationTest(unittest.TestCase):
    def setUp(self):
        seqs = _pool(6, seed=4).sequences
        self.allowed, self.outside = seqs[:3], seqs[3:]
        df = pd.DataFrame({"Sequence": seqs, TARGET_COLUMN: np.arange(6.0)})
        self.oracle = TableOracle(df, pool=self.allowed, holdout=self.outside)

    def test_only_pool_labels_are_visible(self):
        q = self.allowed + self.outside
        values, mask = self.oracle.query("b1", q)
        self.assertTrue(np.isfinite(values[:3]).all())
        self.assertTrue(np.isnan(values[3:]).all())
        self.assertTrue(mask[:3].all() and not mask[3:].any())
        # second query on an out-of-pool sequence still yields nothing
        v2, m2 = self.oracle.query("b2", self.outside)
        self.assertTrue(np.isnan(v2).all())
        self.assertFalse(m2.any())

    def test_eligibility_and_already_labeled(self):
        self.assertTrue(self.oracle.eligible_to_query(self.allowed).all())
        self.assertFalse(self.oracle.eligible_to_query(self.outside).any())
        self.oracle.query("b3", self.allowed)
        self.assertTrue(self.oracle.already_labeled(self.allowed).all())
        self.assertFalse(self.oracle.already_labeled(self.outside).any())

    def test_duplicate_table_rejected(self):
        df = pd.DataFrame({"Sequence": [WT_141, WT_141], TARGET_COLUMN: [1.0, 2.0]})
        with self.assertRaises(ValueError):
            TableOracle(df, pool=[WT_141])

    def test_peek_never_reveals(self):
        v, m = self.oracle.peek(self.allowed)
        self.assertTrue(np.isnan(v).all())
        self.assertFalse(m.any())


class ParkResumeTest(unittest.TestCase):
    def test_park_persists_and_ingest_validates(self):
        with tempfile.TemporaryDirectory() as tmp:
            oracle = ParkOracle(Path(tmp) / "park")
            seqs = _pool(5, seed=6).sequences
            v, ok = oracle.query("abc123", seqs)
            self.assertTrue(np.isnan(v).all())
            self.assertFalse(ok.any())
            batch_file = Path(tmp) / "park/batches/batch_abc123/batch.csv"
            self.assertTrue(batch_file.is_file())
            # re-query never overwrites the parked batch
            oracle.query("abc123", seqs)
            self.assertEqual(len(pd.read_csv(batch_file)), 5)

            good = Path(tmp) / "labels.csv"
            pd.DataFrame({
                "batch_id": ["abc123"] * 5,
                "candidate_id": [candidate_id(s) for s in seqs],
                "sequence": seqs,
                TARGET_COLUMN: [1.0, 2.0, np.nan, 4.0, 5.0],
                "measurement_status": ["pass", "pass", "pending", "pass", "fail"],
                "protocol_id": ["p1"] * 5,
            }).to_csv(good, index=False)
            report = oracle.ingest(good)
            self.assertEqual(report["accepted"], 3)
            self.assertEqual(report["failed"], 1)          # QC fail is persisted, not lost
            self.assertEqual(len(report["rejected"]), 1)   # only the bad status
            self.assertEqual(oracle.batch_status("abc123")["status"], "partially_labeled")
            # idempotent re-ingest
            report2 = oracle.ingest(good)
            self.assertEqual(report2["accepted"], 0)
            self.assertGreaterEqual(report2["duplicate"], 3)

    def test_fail_is_persisted_across_restart_and_closes_batch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "park"
            seqs = _pool(4, seed=12).sequences
            oracle = ParkOracle(root)
            oracle.query("b1", seqs)
            rows = pd.DataFrame({
                "batch_id": ["b1"] * 4,
                "candidate_id": [candidate_id(s) for s in seqs],
                "sequence": seqs,
                TARGET_COLUMN: [1.0, 2.0, np.nan, np.nan],
                "measurement_status": ["pass", "pass", "fail", "fail"],
                "protocol_id": ["p1"] * 4,
            })
            report = oracle.ingest(_csv(Path(tmp) / "r1.csv", rows))
            self.assertEqual((report["accepted"], report["failed"]), (2, 2))
            self.assertEqual(report["closed_batches"], ["b1"])
            self.assertTrue(oracle.batch_status("b1")["closed"])

            # a fresh oracle (process restart) still knows the failures
            restarted = ParkOracle(root)
            self.assertEqual(restarted.failed_sequences(), {seqs[2], seqs[3]})
            self.assertEqual(set(restarted.pending), {seqs[0], seqs[1]})
            self.assertTrue(restarted.batch_status("b1")["closed"])
            # failures cannot be flipped to pass afterwards
            flip = pd.DataFrame({"batch_id": ["b1"], "candidate_id": [candidate_id(seqs[2])],
                                 "sequence": [seqs[2]], TARGET_COLUMN: [7.0],
                                 "measurement_status": ["pass"], "protocol_id": ["p1"]})
            rep = restarted.ingest(_csv(Path(tmp) / "flip.csv", flip))
            self.assertEqual(rep["accepted"], 0)
            self.assertIn("already failed QC", {r["reason"] for r in rep["rejected"]})

    def test_ingest_checks_candidate_id_and_protocol(self):
        with tempfile.TemporaryDirectory() as tmp:
            oracle = ParkOracle(Path(tmp) / "park", approved_protocols=["approved_v1"])
            seqs = _pool(3, seed=21).sequences
            oracle.query("b1", seqs)
            rows = pd.DataFrame({
                "batch_id": ["b1", "b1", "b1", "b1"],
                "candidate_id": [candidate_id(seqs[0]), "deadbeefdeadbeef",
                                 candidate_id(seqs[1]), candidate_id(seqs[2])],
                "sequence": [seqs[0], seqs[1], seqs[1], seqs[2]],
                TARGET_COLUMN: [1.0, 2.0, 3.0, 4.0],
                "measurement_status": ["pass"] * 4,
                "protocol_id": ["approved_v1", "approved_v1", "", "other_protocol"],
            })
            report = oracle.ingest(_csv(Path(tmp) / "mixed.csv", rows))
            self.assertEqual(report["accepted"], 1)
            reasons = [r["reason"] for r in report["rejected"]]
            self.assertTrue(any("candidate_id does not match" in r for r in reasons), reasons)
            self.assertTrue(any("missing protocol_id" in r for r in reasons), reasons)
            self.assertTrue(any("unapproved protocol_id" in r for r in reasons), reasons)
            # the single good row is the only confirmed label
            self.assertEqual(set(oracle.pending), {seqs[0]})
            # no temporary files are left behind by the atomic writer
            self.assertEqual(list((Path(tmp) / "park" / "batches" / "batch_b1")
                                  .glob("*.tmp")), [])

    def test_ingest_duplicate_ids_with_conflicting_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            oracle = ParkOracle(Path(tmp) / "park")
            seqs = _pool(2, seed=22).sequences
            oracle.query("b1", seqs)
            rows = pd.DataFrame({
                "batch_id": ["b1"] * 3,
                "candidate_id": [candidate_id(seqs[0])] * 3,
                "sequence": [seqs[0]] * 3,
                TARGET_COLUMN: [1.0, 9.0, 1.0],
                "measurement_status": ["pass"] * 3,
                "protocol_id": ["p1"] * 3,
            })
            report = oracle.ingest(_csv(Path(tmp) / "dup.csv", rows))
            self.assertEqual(report["accepted"], 1)
            self.assertEqual(report["duplicate"], 1)          # the exact repeat
            self.assertIn("conflicting gbsa", {r["reason"] for r in report["rejected"]})
            self.assertEqual(oracle.pending[seqs[0]], 1.0)    # confirmed value untouched
            # a later file cannot overwrite the confirmed value either
            again = pd.DataFrame({"batch_id": ["b1"], "candidate_id": [candidate_id(seqs[0])],
                                  "sequence": [seqs[0]], TARGET_COLUMN: [-5.0],
                                  "measurement_status": ["pass"], "protocol_id": ["p1"]})
            rep2 = oracle.ingest(_csv(Path(tmp) / "again.csv", again))
            self.assertEqual(rep2["accepted"], 0)
            self.assertIn("conflicting gbsa", {r["reason"] for r in rep2["rejected"]})
            self.assertEqual(oracle.pending[seqs[0]], 1.0)

    def test_ingest_rejects_unknown_batch_and_conflicts(self):
        with tempfile.TemporaryDirectory() as tmp:
            oracle = ParkOracle(Path(tmp) / "park")
            seqs = _pool(3, seed=8).sequences
            oracle.query("b1", seqs)
            bad = Path(tmp) / "bad.csv"
            pd.DataFrame({
                "batch_id": ["nope", "b1", "b1"],
                "candidate_id": [candidate_id(s) for s in seqs],
                "sequence": seqs,
                TARGET_COLUMN: [1.0, np.inf, 3.0],
                "measurement_status": ["pass", "pass", "pass"],
                "protocol_id": ["p"] * 3,
            }).to_csv(bad, index=False)
            report = oracle.ingest(bad)
            reasons = {r["reason"] for r in report["rejected"]}
            self.assertIn("unknown batch", reasons)
            self.assertIn("non-finite gbsa", reasons)
            # a valid row for a still-unlabelled batch member is accepted ...
            ok = Path(tmp) / "ok.csv"
            pd.DataFrame({"batch_id": ["b1"], "candidate_id": [candidate_id(seqs[1])],
                          "sequence": [seqs[1]], TARGET_COLUMN: [7.0],
                          "measurement_status": ["pass"], "protocol_id": ["p"]}).to_csv(ok, index=False)
            self.assertEqual(oracle.ingest(ok)["accepted"], 1)
            # ... and a conflicting value for the same candidate is refused
            clash = Path(tmp) / "clash.csv"
            pd.DataFrame({"batch_id": ["b1"], "candidate_id": [candidate_id(seqs[1])],
                          "sequence": [seqs[1]], TARGET_COLUMN: [99.0],
                          "measurement_status": ["pass"], "protocol_id": ["p"]}).to_csv(clash, index=False)
            rep = oracle.ingest(clash)
            self.assertEqual(rep["accepted"], 0)
            self.assertEqual(rep["rejected"][0]["reason"], "conflicting gbsa")
            # identical repeat submission is idempotent
            rep2 = oracle.ingest(ok)
            self.assertEqual(rep2["accepted"], 0)
            self.assertEqual(rep2["duplicate"], 1)


class LedgerAlignmentTest(unittest.TestCase):
    """P0-1: the ledger must carry the prediction *of the sequence on that row*."""

    def test_ledger_predictions_match_model_for_selected_sequences(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "metrics.csv"
            log = Path(tmp) / "ckpt"
            args = ["--mode", "retrospective", "--rounds", "1", "--budget", "30",
                    "--n-init", "120", "--n-holdout", "60", "--pool-size", "300",
                    "--surrogate", "xgb", "--n-members", "2", "--strategy", "greedy",
                    "--seed", "7"]
            proc = subprocess.run(
                [sys.executable, str(ROOT / "scripts/loop/loop.py"), "--output", str(out),
                 "--log-dir", str(log), *args], cwd=str(ROOT), capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, msg=proc.stdout + proc.stderr)

            ledger = pd.read_csv(log / "ledger.csv")
            state = json.loads((log / "state.json").read_text(encoding="utf-8"))
            batch = pd.read_csv(log / "round_001_batch.csv")

            # the batch order must match the ledger order and the recorded ids
            self.assertEqual(list(batch["sequence"]), list(ledger["sequence"]))
            for seq, cid in zip(ledger["sequence"], ledger["candidate_id"]):
                self.assertEqual(candidate_id(seq), cid)

            # one round means the ledger predictions come from the round-1 model,
            # which was trained on the first n_init labels only
            n_init = 120
            train_seqs = state["labeled_sequences"][:n_init]
            train_vals = np.asarray(state["labeled_values"][:n_init], dtype=float)
            surrogate = make_surrogate("xgb", seed=7, n_members=2, screen_k=100,
                                       target=TargetSpec())
            surrogate.fit(train_seqs, train_vals)
            expected = surrogate.predict(list(ledger["sequence"])).mu
            np.testing.assert_allclose(ledger[PREDICTION_COLUMN].to_numpy(float), expected,
                                       rtol=1e-9, atol=1e-9)

            # and the selection really was non-prefix, i.e. the old bug would differ
            positions = ledger["selection_position"].to_numpy(int)
            self.assertTrue((positions != np.arange(len(positions))).any(),
                            "expected a non-prefix selection to make this test meaningful")
            self.assertEqual(len(set(positions)), len(positions))


class FingerprintResumeTest(unittest.TestCase):
    """P0-3: resume must refuse a changed configuration, data set or pool."""

    PARK_ARGS = ["--mode", "prospective", "--candidate-source", "mutate", "--rounds", "2",
                 "--budget", "10", "--n-init", "120", "--n-holdout", "60",
                 "--pool-size", "300", "--surrogate", "xgb", "--n-members", "2",
                 "--oracle", "park", "--protocol-id", "p1", "--seed", "3"]

    def _launch(self, extra, tmp, expect_code):
        out = Path(tmp) / "metrics.csv"
        log = Path(tmp) / "ckpt"
        proc = subprocess.run(
            [sys.executable, str(ROOT / "scripts/loop/loop.py"), "--output", str(out),
             "--log-dir", str(log), *extra], cwd=str(ROOT), capture_output=True, text=True)
        self.assertEqual(proc.returncode, expect_code,
                         msg=f"stdout=\n{proc.stdout}\nstderr=\n{proc.stderr}")
        return out, log, proc

    def test_park_records_fingerprint_and_resume_rejects_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, log, _ = self._launch(self.PARK_ARGS, tmp, expect_code=3)
            fingerprint = json.loads((log / "fingerprint.json").read_text(encoding="utf-8"))
            for key in ("config", "data_sha256", "pool_sha256", "holdout_sha256",
                        "source_sha256", "pool_stats"):
                self.assertIn(key, fingerprint)
            self.assertEqual(fingerprint["pool_stats"]["excluded_canonical_table"], 2000)

            # same configuration resumes fine
            self._launch(self.PARK_ARGS + ["--resume"], tmp, expect_code=3)
            # a changed budget must be refused before any round runs
            changed = ["--budget" if a == "--budget" else a for a in self.PARK_ARGS]
            changed[changed.index("--budget") + 1] = "11"
            _, _, proc = self._launch(changed + ["--resume"], tmp, expect_code=1)
            self.assertIn("fingerprint mismatch", proc.stderr + proc.stdout)

    def test_resume_without_fingerprint_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, log, _ = self._launch(self.PARK_ARGS, tmp, expect_code=3)
            (log / "fingerprint.json").unlink()
            _, _, proc = self._launch(self.PARK_ARGS + ["--resume"], tmp, expect_code=1)
            self.assertIn("fingerprint", proc.stderr + proc.stdout)

    def test_closing_the_batch_clears_parked(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = [a for a in self.PARK_ARGS]
            args[args.index("--rounds") + 1] = "1"       # no further batch is requested
            _, log, _ = self._launch(args, tmp, expect_code=3)
            self.assertTrue((log / "PARKED").is_file())
            batch = pd.read_csv(log / "round_001_batch.csv")
            labels = Path(tmp) / "labels.csv"
            pd.DataFrame({
                "batch_id": batch["batch_id"],
                "candidate_id": batch["candidate_id"],
                "sequence": batch["sequence"],
                TARGET_COLUMN: np.linspace(-30, -10, len(batch)),
                "measurement_status": ["pass"] * len(batch),
                "protocol_id": ["p1"] * len(batch),
            }).to_csv(labels, index=False)

            _, _, proc = self._launch(args + ["--resume", "--ingest", str(labels)],
                                      tmp, expect_code=0)
            self.assertIn("closed", proc.stdout)
            self.assertFalse((log / "PARKED").is_file(), "stale PARKED must be removed")
            state = json.loads((log / "state.json").read_text(encoding="utf-8"))
            self.assertEqual(state["awaiting"], {})
            self.assertIsNone(state["open_batch"])
            self.assertEqual(len(state["failed"]), 0)
            self.assertGreater(len(state["labeled_sequences"]), 120)

    def test_qc_failure_closes_batch_and_excludes_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = [a for a in self.PARK_ARGS]
            args[args.index("--rounds") + 1] = "1"
            _, log, _ = self._launch(args, tmp, expect_code=3)
            batch = pd.read_csv(log / "round_001_batch.csv")
            statuses = ["pass"] * (len(batch) - 2) + ["fail", "fail"]
            labels = Path(tmp) / "labels.csv"
            pd.DataFrame({
                "batch_id": batch["batch_id"],
                "candidate_id": batch["candidate_id"],
                "sequence": batch["sequence"],
                TARGET_COLUMN: list(np.linspace(-30, -10, len(batch) - 2)) + [np.nan, np.nan],
                "measurement_status": statuses,
                "protocol_id": ["p1"] * len(batch),
            }).to_csv(labels, index=False)

            _, _, proc = self._launch(args + ["--resume", "--ingest", str(labels)],
                                      tmp, expect_code=0)
            state = json.loads((log / "state.json").read_text(encoding="utf-8"))
            self.assertEqual(len(state["failed"]), 2)
            self.assertEqual(state["awaiting"], {})
            self.assertFalse((log / "PARKED").is_file())
            failed = set(state["failed"])
            self.assertEqual(len(failed), 2)
            # the failed candidates can never re-enter a batch
            self.assertTrue(failed <= set(batch["sequence"]))
            self.assertIn("failed", proc.stdout)
            # and the ledger records the QC transition instead of leaving them pending
            ledger = pd.read_csv(log / "ledger.csv")
            failed_rows = ledger[ledger["sequence"].isin(failed)]
            self.assertEqual(set(failed_rows["status"]), {"failed"})


class FullLibraryDedupeTest(unittest.TestCase):
    """P1-a: an exported batch never repeats the canonical table or an earlier batch."""

    def test_export_excludes_canonical_table_and_earlier_batches(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "metrics.csv"
            log = Path(tmp) / "ckpt"
            args = ["--mode", "prospective", "--candidate-source", "mutate", "--rounds", "1",
                    "--budget", "20", "--n-init", "120", "--n-holdout", "60",
                    "--pool-size", "300", "--surrogate", "xgb", "--n-members", "2",
                    "--oracle", "park", "--seed", "4"]
            proc = subprocess.run(
                [sys.executable, str(ROOT / "scripts/loop/loop.py"), "--output", str(out),
                 "--log-dir", str(log), *args], cwd=str(ROOT), capture_output=True, text=True)
            self.assertEqual(proc.returncode, 3, msg=proc.stdout + proc.stderr)
            canonical = set(pd.read_csv(
                ROOT / "data/proxy2000_v2/fad_proxy2000_v2_full.csv")["Sequence"].map(normalize))
            batch1 = set(pd.read_csv(log / "round_001_batch.csv")["sequence"])
            self.assertEqual(batch1 & canonical, set(), "batch overlaps the canonical table")
            stats = json.loads((log / "progress.json").read_text(encoding="utf-8"))["pool_stats"]
            self.assertEqual(stats["excluded_canonical_table"], 2000)
            self.assertEqual(stats["excluded_init_labeled"], 120)
            self.assertEqual(stats["excluded_holdout"], 60)


class OodTimePointTest(unittest.TestCase):
    """P1-b: the recorded OOD threshold comes from the pre-acquisition training set."""

    def test_metrics_threshold_matches_pre_round_training_set(self):
        from scripts.loop.loop import ood_threshold
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "metrics.csv"
            log = Path(tmp) / "ckpt"
            args = ["--mode", "retrospective", "--rounds", "2", "--budget", "25",
                    "--n-init", "120", "--n-holdout", "60", "--pool-size", "300",
                    "--surrogate", "xgb", "--n-members", "2", "--strategy", "greedy",
                    "--seed", "9"]
            proc = subprocess.run(
                [sys.executable, str(ROOT / "scripts/loop/loop.py"), "--output", str(out),
                 "--log-dir", str(log), *args], cwd=str(ROOT), capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, msg=proc.stdout + proc.stderr)
            metrics = pd.read_csv(out)
            state = json.loads((log / "state.json").read_text(encoding="utf-8"))
            train = state["labeled_sequences"]
            for _, row in metrics.iterrows():
                before = int(row["n_labeled_before"])
                expected = ood_threshold(train[:before])
                self.assertAlmostEqual(float(row["ood_threshold_hamming"]),
                                       expected["hamming_threshold"], places=6)
                self.assertEqual(str(row["ood_train_n_from_wt"]),
                                 f"{expected['n_from_wt_range'][0]}-"
                                 f"{expected['n_from_wt_range'][1]}")
            # per-stratum report exists for every round
            for rnd in (1, 2):
                strata = pd.read_csv(log / f"round_{rnd:03d}_ood_strata.csv")
                self.assertGreater(len(strata), 0)
                self.assertEqual(int(strata["selected_count"].sum()),
                                 int(metrics.iloc[rnd - 1]["effective_budget"]))


class LoopAcceptanceTest(unittest.TestCase):
    """Subprocess-level checks of the driver (retrospective + prospective)."""

    def _run(self, extra, tmp, expect_code=0):
        out = Path(tmp) / "metrics.csv"
        log = Path(tmp) / "ckpt"
        cmd = [sys.executable, str(ROOT / "scripts/loop/loop.py"), "--output", str(out),
               "--log-dir", str(log), *extra]
        proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
        self.assertEqual(proc.returncode, expect_code,
                         msg=f"stdout=\n{proc.stdout}\nstderr=\n{proc.stderr}")
        return out, log, proc

    def test_cross_round_uniqueness_and_metric_time_point(self):
        with tempfile.TemporaryDirectory() as tmp:
            out, log, _ = self._run(
                ["--mode", "retrospective", "--rounds", "2", "--budget", "40",
                 "--n-init", "120", "--n-holdout", "80", "--pool-size", "300",
                 "--surrogate", "xgb", "--n-members", "3", "--strategy", "greedy",
                 "--seed", "5"], tmp)
            metrics = pd.read_csv(out)
            self.assertEqual(len(metrics), 2)
            b1 = pd.read_csv(log / "round_001_batch.csv")
            b2 = pd.read_csv(log / "round_002_batch.csv")
            self.assertEqual(len(b1), len(b2))
            self.assertEqual(len(set(b1["sequence"]) & set(b2["sequence"])), 0)
            row = metrics.iloc[0]
            self.assertEqual(int(row["model_n_train"]), int(row["n_labeled_before"]))
            self.assertEqual(int(row["n_new_unique"]),
                             int(row["n_labeled_after"]) - int(row["n_labeled_before"]))
            self.assertEqual(int(row["n_new_unique"]), len(b1))
            ledger = pd.read_csv(log / "ledger.csv")
            self.assertEqual(ledger["candidate_id"].nunique(), len(ledger))
            self.assertEqual(len(ledger), int(metrics["n_new_unique"].sum()))
            self.assertIn("holdout_spearman_pre", metrics.columns)
            self.assertIn("holdout_spearman_post", metrics.columns)

    def test_holdout_never_enters_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, log, _ = self._run(
                ["--mode", "retrospective", "--rounds", "1", "--budget", "20",
                 "--n-init", "100", "--n-holdout", "60", "--pool-size", "200",
                 "--surrogate", "xgb", "--n-members", "2", "--seed", "9"], tmp)
            state = json.loads((log / "state.json").read_text(encoding="utf-8"))
            ledger = pd.read_csv(log / "ledger.csv")
            train = set(state["labeled_sequences"]) - set(ledger["sequence"])
            self.assertEqual(len(train), 100)          # exactly the init set

    def test_prospective_parks_and_resumes_with_same_batch(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = ["--mode", "prospective", "--candidate-source", "mutate",
                    "--rounds", "3", "--budget", "20", "--n-init", "120",
                    "--n-holdout", "60", "--pool-size", "300", "--surrogate", "xgb",
                    "--n-members", "2", "--oracle", "park", "--seed", "4"]
            out, log, proc = self._run(args, tmp, expect_code=3)
            self.assertTrue((log / "PARKED").is_file())
            parked = json.loads((log / "PARKED").read_text(encoding="utf-8"))
            batch_id = parked["batch_id"]
            batch_file = log / "pending_batches" / "batches" / f"batch_{batch_id}" / "batch.csv"
            self.assertTrue(batch_file.is_file())
            batch = pd.read_csv(batch_file)

            # partial ingest: only the first 5 measurements return
            labels = Path(tmp) / "labels.csv"
            part = batch.head(5).copy()
            part[TARGET_COLUMN] = np.linspace(10, 20, len(part))
            part["measurement_status"] = "pass"
            part["protocol_id"] = "p1"
            part.to_csv(labels, index=False)

            args_resume = args + ["--resume", "--ingest", str(labels)]
            out2, log2, proc2 = self._run(args_resume, tmp, expect_code=3)
            self.assertTrue((log2 / "PARKED").is_file())
            state = json.loads((log2 / "state.json").read_text(encoding="utf-8"))
            # labels absorbed, batch still open, no new round selected
            self.assertEqual(state["round"], 1)
            self.assertEqual(len(state["awaiting"]), len(batch) - 5)
            self.assertGreaterEqual(len(state["labeled_sequences"]), 120 + 5)
            # the parked batch file was not rewritten
            self.assertEqual(len(pd.read_csv(batch_file)), len(batch))

    def test_resume_after_rounds_exhausted_still_parks_and_absorbs(self):
        """A resume that only delivers measurements must never look DONE."""
        with tempfile.TemporaryDirectory() as tmp:
            args = ["--mode", "prospective", "--candidate-source", "mutate",
                    "--rounds", "1", "--budget", "20", "--n-init", "120",
                    "--n-holdout", "60", "--pool-size", "300", "--surrogate", "xgb",
                    "--n-members", "2", "--oracle", "park", "--seed", "6"]
            _, log, _ = self._run(args, tmp, expect_code=3)
            parked = json.loads((log / "PARKED").read_text(encoding="utf-8"))
            batch_id = parked["batch_id"]
            batch = pd.read_csv(log / "pending_batches" / "batches"
                                / f"batch_{batch_id}" / "batch.csv")
            labels = Path(tmp) / "labels.csv"
            part = batch.head(5).copy()
            part[TARGET_COLUMN] = np.linspace(10, 20, len(part))
            part["measurement_status"] = "pass"
            part["protocol_id"] = "p1"
            part.to_csv(labels, index=False)

            # rounds are already exhausted (1 of 1) and the batch is still open
            _, log2, _ = self._run(args + ["--resume", "--ingest", str(labels)],
                                   tmp, expect_code=3)
            self.assertTrue((log2 / "PARKED").is_file())
            state = json.loads((log2 / "state.json").read_text(encoding="utf-8"))
            self.assertEqual(len(state["awaiting"]), len(batch) - 5)
            self.assertGreaterEqual(len(state["labeled_sequences"]), 120 + 5)
            metrics = pd.read_csv(Path(tmp) / "metrics.csv")
            statuses = set(metrics["status"].astype(str))
            self.assertNotIn("retrained", statuses)
            self.assertIn("awaiting_labels", statuses)

    def test_resume_without_checkpoint_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "m.csv"
            log = Path(tmp) / "empty"
            proc = subprocess.run(
                [sys.executable, str(ROOT / "scripts/loop/loop.py"), "--resume",
                 "--output", str(out), "--log-dir", str(log), "--rounds", "1"],
                cwd=str(ROOT), capture_output=True, text=True)
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("no checkpoint", proc.stderr + proc.stdout)

    def test_pool_exhaustion_stops_cleanly(self):
        with tempfile.TemporaryDirectory() as tmp:
            out, log, proc = self._run(
                ["--mode", "retrospective", "--rounds", "5", "--budget", "100",
                 "--n-init", "120", "--n-holdout", "60", "--pool-size", "120",
                 "--surrogate", "xgb", "--n-members", "2", "--seed", "2"], tmp)
            metrics = pd.read_csv(out)
            self.assertLessEqual(len(metrics), 5)
            self.assertIn("pool_exhausted", set(metrics["status"].astype(str)))


class RunnerSemanticsTest(unittest.TestCase):
    def test_registry_hash_check_and_source_manifest(self):
        sys.path.insert(0, str(ROOT / "scripts"))
        import run_experiment as runner
        self.assertIsNotNone(runner.registry_hash())
        self.assertTrue(runner.source_manifest(
            ROOT / "scripts/loop/loop.py")["n_files"] > 5)
        self.assertEqual(runner.registry_hash("does_not_exist"), None)

    def test_unregistered_dataset_fails_for_formal_runs(self):
        sys.path.insert(0, str(ROOT / "scripts"))
        import run_experiment as runner
        from unittest import mock
        with mock.patch.object(runner, "registry_hash", return_value=None):
            with self.assertRaises(ValueError):
                runner.require_registry_hash({})
            # an explicit non-formal smoke run may opt out
            self.assertIsNone(runner.require_registry_hash({"allow_missing_registry": True}))
        with mock.patch.object(runner, "registry_hash", return_value="DEADBEEF"):
            self.assertEqual(runner.require_registry_hash({}), "DEADBEEF")

    def test_resume_run_requires_an_existing_checkpointed_run(self):
        sys.path.insert(0, str(ROOT / "scripts"))
        import run_experiment as runner
        with self.assertRaises(FileNotFoundError):
            runner.resume_run("no_such_run_id", None)

    def test_resume_run_reuses_the_run_and_updates_the_marker(self):
        sys.path.insert(0, str(ROOT / "scripts"))
        import run_experiment as runner
        run_id = "test_resume_run_entrypoint"
        run_dir = ROOT / "runs" / run_id
        if run_dir.exists():
            shutil.rmtree(run_dir)
        self.addCleanup(lambda: shutil.rmtree(run_dir, ignore_errors=True))
        created = subprocess.run(
            [sys.executable, str(ROOT / "scripts/run_experiment.py"),
             "configs/experiments/loop_prospective_park.json", "--run-id", run_id],
            cwd=str(ROOT), capture_output=True, text=True)
        self.assertEqual(created.returncode, 3, msg=created.stdout + created.stderr)
        self.assertTrue((run_dir / "PARKED").is_file())

        batch = pd.read_csv(run_dir / "checkpoints" / "round_001_batch.csv")
        labels = Path(tmp_csv := tempfile.mkdtemp()) / "labels.csv"
        pd.DataFrame({
            "batch_id": batch["batch_id"],
            "candidate_id": batch["candidate_id"],
            "sequence": batch["sequence"],
            TARGET_COLUMN: np.linspace(-40, -20, len(batch)),
            "measurement_status": ["pass"] * len(batch),
            "protocol_id": ["unassigned"] * len(batch),
        }).to_csv(labels, index=False)
        self.addCleanup(lambda: shutil.rmtree(tmp_csv, ignore_errors=True))

        resumed = subprocess.run(
            [sys.executable, str(ROOT / "scripts/run_experiment.py"),
             "--resume-run", run_id, "--ingest", str(labels)],
            cwd=str(ROOT), capture_output=True, text=True)
        self.assertEqual(resumed.returncode, 0, msg=resumed.stdout + resumed.stderr)
        self.assertFalse((run_dir / "PARKED").is_file())
        self.assertTrue((run_dir / "DONE").is_file())
        self.assertTrue((run_dir / "resume_history.json").is_file())
        # the resumed run keeps the original config and its manifest hashes
        config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        self.assertEqual(config["name"], "loop_prospective_park")
        self.assertFalse((run_dir / "checkpoints" / "PARKED").is_file())


class DiscoveryMetricsTest(unittest.TestCase):
    """§5.3.2 additional metrics: top-10% pool recall and the pool-split copy."""

    def test_top10_recall_counts_only_true_pool_best(self):
        sys.path.insert(0, str(ROOT))
        from scripts.loop.strategy_compare import top10_recall
        pool_seqs = np.array([f"seq{i:03d}" for i in range(20)])
        pool_values = np.linspace(0, 19, 20)             # seq000 is the best
        self.assertAlmostEqual(top10_recall(set(), pool_seqs, pool_values), 0.0)
        self.assertAlmostEqual(
            top10_recall({"seq000", "seq001", "seq019"}, pool_seqs, pool_values), 1.0)
        # queried set may contain anything; only the pool's true top-10% counts
        self.assertAlmostEqual(
            top10_recall({"seq000", "seq010", "unknown"}, pool_seqs, pool_values), 0.5)
        # empty pool is defined, not an error
        self.assertTrue(np.isnan(top10_recall(set(), np.array([]), np.array([]))))

    def test_pool_truth_matches_loop_split(self):
        sys.path.insert(0, str(ROOT))
        from scripts.loop.strategy_compare import pool_truth

        class Args:
            data_dir = "data/proxy2000_v2"
            n_init, n_holdout, pool_size = 300, 200, 600
        seqs, values = pool_truth(Args(), seed=5)
        self.assertEqual(len(seqs), 600)
        self.assertEqual(len(values), 600)
        # deterministic and disjoint from the init/holdout region by construction
        seqs2, _ = pool_truth(Args(), seed=5)
        self.assertEqual(list(seqs), list(seqs2))


class CompletenessTest(unittest.TestCase):
    """§5.3/§7: an incomplete comparison must not look like a result."""

    def test_incomplete_comparison_is_reported(self):
        sys.path.insert(0, str(ROOT))
        from scripts.loop.strategy_compare import check_completeness

        class Args:
            rounds = 5
            n_init = 300
        seeds = range(1, 4)
        strategies = ["random", "greedy"]
        rows = []
        for seed in seeds:
            for strategy in strategies:
                for rnd in range(1, 6):
                    if seed == 2 and strategy == "greedy" and rnd == 3:
                        continue                      # drop one sub-run round
                    rows.append({"seed": seed, "strategy": strategy, "round": rnd,
                                 "best_selected": -1.0, "regret": 0.5, "pool_best": -5.0,
                                 "n_new_unique": 50, "holdout_spearman_post": 0.2,
                                 "n_labeled_after": 300 + 50 * rnd})
        report = check_completeness(pd.DataFrame(rows), seeds, strategies, Args())
        self.assertFalse(report["ok"])
        self.assertTrue(any("rounds=" in p for p in report["problems"]))

        complete = [r for r in rows if not (r["seed"] == 2 and r["strategy"] == "greedy"
                                            and r["round"] == 3)]
        complete += [{"seed": 2, "strategy": "greedy", "round": 3, "best_selected": -1.0,
                      "regret": 0.5, "pool_best": -5.0, "n_new_unique": 50,
                      "holdout_spearman_post": 0.2, "n_labeled_after": 450}]
        report2 = check_completeness(pd.DataFrame(complete), seeds, strategies, Args())
        self.assertTrue(report2["ok"], report2["problems"])
        self.assertEqual(report2["expected_rows"], 30)
        # observed_runs counts distinct (seed, strategy) sub-runs, NOT round rows
        self.assertEqual(report2["observed_runs"], 6)
        self.assertEqual(report2["observed_rows"], 30)

        dup = complete + [complete[0]]
        self.assertFalse(check_completeness(pd.DataFrame(dup), seeds, strategies,
                                            Args())["ok"])


class TailStatsTest(unittest.TestCase):
    """The worst-10% label must point at the seeds where greedy did WORST."""

    def test_tail_stats_direction(self):
        sys.path.insert(0, str(ROOT))
        from scripts.loop.strategy_compare import tail_stats
        common = list(range(10))
        d = np.array([-8.0, -3.0, -1.0, 0.0, 0.0, 0.0, 0.5, 1.0, 2.0, 3.0])
        out = tail_stats(d, common, n_tail=2)
        # largest (most positive) deltas = where the strategy did worst
        self.assertEqual(out["worst10pct_seeds"], [9, 8])
        self.assertAlmostEqual(out["worst10pct_delta_regret_mean"], 2.5)
        # most negative = where it improved most
        self.assertEqual(out["best10pct_seeds"], [0, 1])
        self.assertAlmostEqual(out["best10pct_delta_regret_mean"], -5.5)
        self.assertAlmostEqual(out["best10pct_share_of_mean"],
                               (-11.0) / d.sum(), places=12)


class ResumeRetrainTest(unittest.TestCase):
    """Full ingest via the runner must retrain, update the ledger and reach DONE."""

    ARGS = {"mode": "prospective", "candidate-source": "mutate", "rounds": 1,
            "budget": 8, "n-init": 120, "n-holdout": 60, "pool-size": 300,
            "surrogate": "xgb", "n-members": 2, "oracle": "park",
            "protocol-id": "gbsa_pending_protocol", "seed": 55}

    def _make_run(self, run_id: str, tmp: str) -> Path:
        import run_experiment as runner
        run_dir = ROOT / "runs" / run_id
        if run_dir.exists():
            shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True)
        (run_dir / "config.json").write_text(json.dumps({
            "name": "test_ingest_flow", "entrypoint": "scripts/loop/loop.py",
            "supports_log_dir": True, "arguments": dict(self.ARGS),
            "acceptance": {"expect_marker": "PARKED", "expect_exit_code": 3}}),
            encoding="utf-8")
        out = run_dir / "metrics.csv"
        log = run_dir / "checkpoints"
        cmd = [sys.executable, str(ROOT / "scripts/loop/loop.py"),
               "--output", str(out), "--log-dir", str(log)]
        for key, value in self.ARGS.items():
            cmd += [f"--{key}", str(value)]
        proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
        self.assertEqual(proc.returncode, 3, msg=proc.stdout + proc.stderr)
        return run_dir

    def _labels(self, batch: pd.DataFrame, values, statuses, protocol) -> Path:
        labels = Path(tempfile.mkdtemp()) / "labels.csv"
        rows = pd.DataFrame({
            "batch_id": batch["batch_id"],
            "candidate_id": batch["candidate_id"],
            "sequence": batch["sequence"],
            TARGET_COLUMN: values,
            "measurement_status": statuses,
            "protocol_id": [protocol] * len(batch),
        })
        rows.to_csv(labels, index=False)
        return labels

    def test_full_pass_ingest_retrains_and_closes(self):
        sys.path.insert(0, str(ROOT / "scripts"))
        import run_experiment as runner
        run_id = "test_full_ingest_retrain"
        run_dir = self._make_run(run_id, tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(run_dir, ignore_errors=True))
        batch = pd.read_csv(run_dir / "checkpoints" / "round_001_batch.csv")
        labels = self._labels(batch, np.linspace(-30, -10, len(batch)),
                              ["pass"] * len(batch), "gbsa_pending_protocol")
        self.addCleanup(lambda: shutil.rmtree(labels.parent, ignore_errors=True))

        rc = runner.resume_run(run_id, str(labels))
        self.assertEqual(rc, 0)
        self.assertTrue((run_dir / "DONE").is_file())
        self.assertFalse((run_dir / "PARKED").is_file())
        ledger = pd.read_csv(run_dir / "checkpoints" / "ledger.csv")
        self.assertEqual(set(ledger["status"]), {"labeled"})
        self.assertFalse(ledger["measured_gbsa"].isna().any())
        np.testing.assert_allclose(ledger["measured_gbsa"].to_numpy(float),
                                   np.linspace(-30, -10, len(batch)))
        state = json.loads((run_dir / "checkpoints" / "state.json").read_text(encoding="utf-8"))
        self.assertEqual(state["status"], "retrained")
        self.assertEqual(state["awaiting"], {})
        metrics = pd.read_csv(run_dir / "metrics.csv")
        retrained = metrics[metrics["status"].astype(str) == "retrained"]
        self.assertEqual(len(retrained), 1)
        self.assertTrue(np.isfinite(float(retrained.iloc[0]["holdout_spearman_post"])))
        self.assertEqual(int(retrained.iloc[0]["n_new_unique"]), len(batch))

    def test_mixed_pass_fail_ingest_closes_and_excludes(self):
        sys.path.insert(0, str(ROOT / "scripts"))
        import run_experiment as runner
        run_id = "test_mixed_ingest_retrain"
        run_dir = self._make_run(run_id, tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(run_dir, ignore_errors=True))
        batch = pd.read_csv(run_dir / "checkpoints" / "round_001_batch.csv")
        n_pass = len(batch) - 2
        values = list(np.linspace(-30, -10, n_pass)) + [np.nan, np.nan]
        statuses = ["pass"] * n_pass + ["fail", "fail"]
        labels = self._labels(batch, values, statuses, "gbsa_pending_protocol")
        self.addCleanup(lambda: shutil.rmtree(labels.parent, ignore_errors=True))

        rc = runner.resume_run(run_id, str(labels))
        self.assertEqual(rc, 0)
        ledger = pd.read_csv(run_dir / "checkpoints" / "ledger.csv")
        self.assertEqual(int((ledger["status"] == "labeled").sum()), n_pass)
        failed_rows = ledger[ledger["status"] == "failed"]
        self.assertEqual(len(failed_rows), 2)
        self.assertTrue(failed_rows["measured_gbsa"].isna().all())
        state = json.loads((run_dir / "checkpoints" / "state.json").read_text(encoding="utf-8"))
        self.assertEqual(len(state["failed"]), 2)
        self.assertEqual(state["status"], "retrained")


class ProtocolPinTest(unittest.TestCase):
    """The audited protocol-pin flow must gate real ingest (P0 follow-up)."""

    def test_pin_protocol_then_ingest_enforces_allow_list(self):
        sys.path.insert(0, str(ROOT / "scripts"))
        import run_experiment as runner
        run_id = "test_protocol_pin"
        run_dir = ResumeRetrainTest()._make_run(run_id, tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(run_dir, ignore_errors=True))

        # 1) pin the lab protocol through the audited flow (no ingest yet)
        rc = runner.resume_run(run_id, None, pin="team_mmgbsa_v1")
        self.assertEqual(rc, 3)                 # still parked: nothing returned
        audit = run_dir / "checkpoints" / "protocol_changes.jsonl"
        self.assertTrue(audit.is_file())
        self.assertIn("team_mmgbsa_v1", audit.read_text(encoding="utf-8"))
        fp = json.loads((run_dir / "checkpoints" / "fingerprint.json").read_text(encoding="utf-8"))
        self.assertEqual(fp["config"]["approved_protocols"], "team_mmgbsa_v1")
        self.assertIn("protocol_pin_audit", fp)
        ledger = pd.read_csv(run_dir / "checkpoints" / "ledger.csv")
        self.assertEqual(set(ledger["protocol_id"]), {"team_mmgbsa_v1"})

        # the oracle-side batch manifest copy AND the embedded fingerprints must
        # agree with the pinned protocol (2026-10-01 audit finding)
        state = json.loads((run_dir / "checkpoints" / "state.json").read_text(encoding="utf-8"))
        oracle_manifest = (run_dir / "checkpoints" / "pending_batches" / "batches"
                           / f"batch_{state['open_batch']}" / "batch_manifest.json")
        self.assertTrue(oracle_manifest.is_file())
        om = json.loads(oracle_manifest.read_text(encoding="utf-8"))
        self.assertEqual(om["protocol_id"], "team_mmgbsa_v1")
        self.assertEqual(om["protocol_pinned_to"], "team_mmgbsa_v1")
        self.assertEqual(om["fingerprint"]["config"]["protocol_id"], "team_mmgbsa_v1")
        self.assertEqual(om["fingerprint"]["config"]["approved_protocols"], "team_mmgbsa_v1")
        ckpt_manifest = json.loads((run_dir / "checkpoints"
                                    / "round_001_batch_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(ckpt_manifest["fingerprint"]["config"]["protocol_id"], "team_mmgbsa_v1")

        # the run-package verifier (which checks both manifest copies, the allow-list
        # and the embedded fingerprints) must accept the pinned-but-parked run
        verify = subprocess.run(
            [sys.executable, str(ROOT / "tools/verify_loop_run.py"), str(run_dir)],
            cwd=str(ROOT), capture_output=True, text=True)
        self.assertEqual(verify.returncode, 0, msg=verify.stdout + verify.stderr)

        # 2) a wrong protocol string is rejected and the batch stays parked
        batch = pd.read_csv(run_dir / "checkpoints" / "round_001_batch.csv")
        rogue = ResumeRetrainTest()._labels(batch, np.linspace(-30, -10, len(batch)),
                                            ["pass"] * len(batch), "rogue_protocol")
        self.addCleanup(lambda: shutil.rmtree(rogue.parent, ignore_errors=True))
        rc = runner.resume_run(run_id, str(rogue))
        self.assertEqual(rc, 3)
        report = json.loads((run_dir / "checkpoints" / "pending_batches"
                             / "ingest_report.json").read_text(encoding="utf-8"))
        self.assertEqual(report["accepted"], 0)
        self.assertTrue(any("unapproved protocol_id" in r["reason"]
                            for r in report["rejected"]))

        # 3) the approved protocol is accepted, the batch closes and the run retrains
        good = ResumeRetrainTest()._labels(batch, np.linspace(-30, -10, len(batch)),
                                           ["pass"] * len(batch), "team_mmgbsa_v1")
        self.addCleanup(lambda: shutil.rmtree(good.parent, ignore_errors=True))
        rc = runner.resume_run(run_id, str(good))
        self.assertEqual(rc, 0)
        self.assertTrue((run_dir / "DONE").is_file())
        manifest = json.loads((run_dir / "checkpoints"
                               / "round_001_batch_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["protocol_pinned_to"], "team_mmgbsa_v1")

    def test_pin_rejects_blank_protocol(self):
        sys.path.insert(0, str(ROOT / "scripts"))
        import run_experiment as runner
        run_id = "test_pin_blank"
        run_dir = ResumeRetrainTest()._make_run(run_id, tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(run_dir, ignore_errors=True))
        with self.assertRaises(ValueError):
            runner.pin_protocol(run_dir, "   ")

    def test_pin_rejected_after_measurements_returned(self):
        sys.path.insert(0, str(ROOT / "scripts"))
        import run_experiment as runner
        run_id = "test_pin_after_labels"
        run_dir = ResumeRetrainTest()._make_run(run_id, tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(run_dir, ignore_errors=True))
        batch = pd.read_csv(run_dir / "checkpoints" / "round_001_batch.csv")
        part = batch.head(2).copy()
        part[TARGET_COLUMN] = [10.0, 11.0]
        part["measurement_status"] = "pass"
        part["protocol_id"] = "gbsa_pending_protocol"
        labels = Path(tempfile.mkdtemp()) / "labels.csv"
        part.to_csv(labels, index=False)
        self.addCleanup(lambda: shutil.rmtree(labels.parent, ignore_errors=True))
        # partial ingest still leaves the run PARKED, but labels now exist
        rc = runner.resume_run(run_id, str(labels))
        self.assertEqual(rc, 3)
        with self.assertRaises(ValueError):
            runner.pin_protocol(run_dir, "team_v2")

    def test_no_post_retrain_eval_still_reaches_retrained(self):
        sys.path.insert(0, str(ROOT / "scripts"))
        import run_experiment as runner
        run_id = "test_no_eval_retrain"
        args = dict(ResumeRetrainTest.ARGS)
        args["no-post-retrain-eval"] = True
        run_dir = ROOT / "runs" / run_id
        if run_dir.exists():
            shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True)
        (run_dir / "config.json").write_text(json.dumps({
            "name": "test_no_eval", "entrypoint": "scripts/loop/loop.py",
            "supports_log_dir": True, "arguments": args,
            "acceptance": {"expect_marker": "PARKED", "expect_exit_code": 3}}),
            encoding="utf-8")
        cmd = [sys.executable, str(ROOT / "scripts/loop/loop.py"),
               "--output", str(run_dir / "metrics.csv"),
               "--log-dir", str(run_dir / "checkpoints"),
               *runner.command_args(args)]
        proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
        self.assertEqual(proc.returncode, 3, msg=proc.stdout + proc.stderr)
        self.addCleanup(lambda: shutil.rmtree(run_dir, ignore_errors=True))

        batch = pd.read_csv(run_dir / "checkpoints" / "round_001_batch.csv")
        labels = ResumeRetrainTest()._labels(
            batch, np.linspace(-30, -10, len(batch)), ["pass"] * len(batch),
            "gbsa_pending_protocol")
        self.addCleanup(lambda: shutil.rmtree(labels.parent, ignore_errors=True))
        rc = runner.resume_run(run_id, str(labels))
        self.assertEqual(rc, 0)                    # DONE, not FAILED
        state = json.loads((run_dir / "checkpoints" / "state.json").read_text(encoding="utf-8"))
        self.assertEqual(state["status"], "retrained")
        metrics = pd.read_csv(run_dir / "metrics.csv")
        retrained = metrics[metrics["status"].astype(str) == "retrained"]
        self.assertEqual(len(retrained), 1)
        # eval skipped: the cell is empty (parsed back as NaN)
        self.assertTrue(pd.isna(retrained.iloc[0]["holdout_spearman_post"]))


class DoneStateGuardTest(unittest.TestCase):
    """finish_run marker semantics: DONE needs retrained, PARKED must not fail."""

    def test_done_requires_retrained_state(self):
        sys.path.insert(0, str(ROOT / "scripts"))
        import run_experiment as runner
        run_dir = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(run_dir, ignore_errors=True))
        pd.DataFrame([{"a": 1}]).to_csv(run_dir / "metrics.csv", index=False)
        ckpt = run_dir / "checkpoints"
        ckpt.mkdir()
        (ckpt / "state.json").write_text(json.dumps({"status": "labels_ingested"}),
                                         encoding="utf-8")
        config = {"arguments": {}}
        rc = runner.finish_run(run_dir, config, 0)
        self.assertEqual(rc, 1)
        self.assertTrue((run_dir / "FAILED").is_file())
        # and a genuinely retrained state passes
        (ckpt / "state.json").write_text(json.dumps({"status": "retrained"}),
                                         encoding="utf-8")
        (run_dir / "FAILED").unlink()
        rc = runner.finish_run(run_dir, config, 0)
        self.assertEqual(rc, 0)
        self.assertTrue((run_dir / "DONE").is_file())

    def test_parked_acceptance_not_poisoned_by_awaiting_state(self):
        """A normal park must keep acceptance_ok=true (2026-10-01 audit finding)."""
        sys.path.insert(0, str(ROOT / "scripts"))
        import run_experiment as runner
        run_dir = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(run_dir, ignore_errors=True))
        pd.DataFrame([{"a": 1}]).to_csv(run_dir / "metrics.csv", index=False)
        ckpt = run_dir / "checkpoints"
        ckpt.mkdir()
        (ckpt / "state.json").write_text(json.dumps({"status": "awaiting_labels"}),
                                         encoding="utf-8")
        (ckpt / "PARKED").write_text(json.dumps({"reason": "awaiting_labels",
                                                 "n_awaiting": 10}), encoding="utf-8")
        config = {"arguments": {}, "acceptance": {"expect_marker": "PARKED",
                                                  "expect_exit_code": 3}}
        rc = runner.finish_run(run_dir, config, 3)
        self.assertEqual(rc, 3)
        marker = json.loads((run_dir / "PARKED").read_text(encoding="utf-8"))
        self.assertTrue(marker["acceptance_ok"])
        self.assertTrue(marker["terminal_state_ok"])
        self.assertEqual(marker["checkpoint_status"], "awaiting_labels")
        self.assertFalse((run_dir / "FAILED").is_file())


if __name__ == "__main__":
    unittest.main()
