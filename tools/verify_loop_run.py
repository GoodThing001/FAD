"""Acceptance check of a completed loop run directory (plan §11.1 witness).

Reads a run directory produced by scripts/run_experiment.py and verifies the
invariants that must hold for the record to be trustworthy:

  * marker is DONE for a completed retrospective run / PARKED for a pending batch
  * every round's batch file has the requested budget and unique candidate ids
  * consecutive batches do not overlap
  * n_labeled_after - n_labeled_before == n_new_unique, cumulative uniqueness holds
  * model_n_train equals the number of labels available *before* the round
  * sigma is either present and positive or absent (never 0)
  * provenance columns are complete for offline-generated candidates

Usage: python tools/verify_loop_run.py runs/verify_smoke_20260930
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.loop.interface import MUT_POS, candidate_id, normalize


def canonical_protocols(value) -> list:
    """Normalize a protocol allow-list (list, comma string, or plain string)."""
    if isinstance(value, list):
        return sorted(str(x).strip() for x in value if str(x).strip())
    return sorted(x.strip() for x in str(value or "").split(",") if x.strip())


def main() -> int:
    run_dir = Path(sys.argv[1])
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    problems: list[str] = []
    notes: list[str] = []

    marker = next((m for m in ("DONE", "PARKED", "FAILED") if (run_dir / m).is_file()), None)
    if marker is None:
        return print(f"[FAIL] {run_dir} has no DONE/PARKED/FAILED marker") or 1
    detail = json.loads((run_dir / marker).read_text(encoding="utf-8"))
    notes.append(f"marker={marker} return_code={detail.get('return_code')} "
                 f"metrics_rows={detail.get('metrics_rows')} "
                 f"acceptance_ok={detail.get('acceptance_ok')}")

    metrics = pd.read_csv(run_dir / "metrics.csv")
    ckpt = run_dir / "checkpoints"
    ledger = pd.read_csv(ckpt / "ledger.csv")

    # --- checkpoint fingerprint (P0-3) ---
    fp_path = ckpt / "fingerprint.json"
    if not fp_path.is_file():
        problems.append("checkpoints/fingerprint.json is missing")
    else:
        fp = json.loads(fp_path.read_text(encoding="utf-8"))
        for key in ("config", "data_sha256", "pool_sha256", "holdout_sha256", "pool_stats"):
            if key not in fp:
                problems.append(f"fingerprint.json lacks {key}")
        data_path = ROOT / "data/proxy2000_v2/fad_proxy2000_v2_full.csv"
        if data_path.is_file():
            import hashlib
            digest = hashlib.sha256(data_path.read_bytes()).hexdigest()
            if fp.get("data_sha256") != digest:
                problems.append("fingerprint data_sha256 does not match the canonical data")
        notes.append(f"fingerprint: pool={str(fp.get('pool_sha256'))[:12]} "
                     f"n_pool={fp.get('pool_size')} "
                     f"excluded_canonical="
                     f"{fp.get('pool_stats', {}).get('excluded_canonical_table')}")

    if marker == "PARKED":
        if not (ckpt / "PARKED").is_file():
            problems.append("PARKED run has no checkpoints/PARKED payload")
        state = json.loads((ckpt / "state.json").read_text(encoding="utf-8"))
        if not state["awaiting"]:
            problems.append("PARKED run has no pending candidates")
        payload = json.loads((ckpt / "PARKED").read_text(encoding="utf-8"))
        if payload.get("n_awaiting") != len(state["awaiting"]):
            problems.append("PARKED n_awaiting does not match the state")
        if payload.get("batch_id") != state.get("open_batch"):
            problems.append("PARKED batch_id does not match state.open_batch")
        notes.append(f"awaiting={len(state['awaiting'])} open_batch={state['open_batch']}")
    else:
        leftover = set(ledger["status"].astype(str)) - {"labeled", "failed"}
        if leftover:
            problems.append(f"DONE run has ledger rows with status {sorted(leftover)}")
        if (ckpt / "PARKED").is_file():
            problems.append("DONE run still carries a stale checkpoints/PARKED marker")
        labeled = ledger[ledger["status"].astype(str) == "labeled"]
        if labeled["measured_gbsa"].isna().any():
            problems.append("DONE run has labeled ledger rows without measured_gbsa")
        state_done = json.loads((ckpt / "state.json").read_text(encoding="utf-8"))
        if state_done.get("status") != "retrained":
            problems.append(f"DONE run state.status={state_done.get('status')!r} "
                            "(expected retrained)")

    # --- hand-off sidecar: manifest + export columns + protocol consistency ---
    fp_config = fp.get("config", {}) if fp_path.is_file() else {}
    for row in metrics.itertuples():
        if getattr(row, "effective_budget", 0) in (0, None):
            continue
        manifest_file = ckpt / f"round_{int(row.round):03d}_batch_manifest.json"
        if not manifest_file.is_file():
            problems.append(f"missing {manifest_file.name}")
            continue
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        for key in ("batch_id", "export_columns", "model_id", "protocol_id",
                    "fingerprint", "ingest_schema"):
            if key not in manifest:
                problems.append(f"{manifest_file.name} lacks {key}")
        for column in ("batch_id", "candidate_id", "sequence", "pred_gbsa",
                       "ood_flag", "protocol_id"):
            if column not in manifest.get("export_columns", []):
                problems.append(f"{manifest_file.name}: export_columns lacks {column}")
        # protocol consistency: manifest ↔ embedded fingerprint ↔ fingerprint.json
        embedded = manifest.get("fingerprint")
        if isinstance(embedded, dict):
            embedded_cfg = embedded.get("config", {})
            if embedded_cfg.get("protocol_id") != manifest.get("protocol_id"):
                problems.append(f"{manifest_file.name}: embedded fingerprint "
                                f"protocol_id {embedded_cfg.get('protocol_id')!r} != "
                                f"manifest {manifest.get('protocol_id')!r}")
            if canonical_protocols(embedded_cfg.get("approved_protocols")) \
                    != canonical_protocols(manifest.get("approved_protocols")):
                problems.append(f"{manifest_file.name}: embedded fingerprint "
                                "approved_protocols does not match the manifest")
        if fp_config and manifest.get("protocol_id") != fp_config.get("protocol_id"):
            problems.append(f"{manifest_file.name}: protocol_id {manifest.get('protocol_id')!r} "
                            f"!= fingerprint.json {fp_config.get('protocol_id')!r}")
        if fp_config and canonical_protocols(manifest.get("approved_protocols")) \
                != canonical_protocols(fp_config.get("approved_protocols")):
            problems.append(f"{manifest_file.name}: approved_protocols "
                            f"{manifest.get('approved_protocols')!r} != fingerprint.json "
                            f"{fp_config.get('approved_protocols')!r}")
        # the oracle-side copy must exist and agree on protocol AND allow-list,
        # including inside its embedded fingerprint
        batch_id = str(manifest.get("batch_id", ""))
        oracle_copy = ckpt / "pending_batches" / "batches" / f"batch_{batch_id}" \
            / "batch_manifest.json"
        if oracle_copy.is_file():
            oc = json.loads(oracle_copy.read_text(encoding="utf-8"))
            for field in ("protocol_id", "protocol_pinned_to"):
                if oc.get(field) != manifest.get(field):
                    problems.append(f"{oracle_copy.name}: {field} {oc.get(field)!r} "
                                    f"!= checkpoint manifest {manifest.get(field)!r}")
            if canonical_protocols(oc.get("approved_protocols")) \
                    != canonical_protocols(manifest.get("approved_protocols")):
                problems.append(f"{oracle_copy.name}: approved_protocols "
                                f"{oc.get('approved_protocols')!r} != checkpoint manifest "
                                f"{manifest.get('approved_protocols')!r}")
            oc_embedded = oc.get("fingerprint")
            if isinstance(oc_embedded, dict):
                oc_cfg = oc_embedded.get("config", {})
                if oc_cfg.get("protocol_id") != manifest.get("protocol_id"):
                    problems.append(f"{oracle_copy.name}: embedded fingerprint "
                                    "protocol_id does not match the checkpoint manifest")
                if canonical_protocols(oc_cfg.get("approved_protocols")) \
                        != canonical_protocols(manifest.get("approved_protocols")):
                    problems.append(f"{oracle_copy.name}: embedded fingerprint "
                                    "approved_protocols does not match the checkpoint "
                                    "manifest")
        else:
            problems.append(f"{oracle_copy.name} is missing")
    if "protocol_id" in ledger.columns:
        if ledger["protocol_id"].isna().any() or (ledger["protocol_id"].astype(str)
                                                 .str.strip() == "").any():
            problems.append("ledger has rows without a protocol_id")

    # --- per-stratum OOD report (P1-b) ---
    for row in metrics.itertuples():
        if getattr(row, "effective_budget", 0) in (0, None):
            continue
        strata_file = ckpt / f"round_{int(row.round):03d}_ood_strata.csv"
        if not strata_file.is_file():
            problems.append(f"missing {strata_file.name}")
            continue
        strata = pd.read_csv(strata_file)
        if int(strata["selected_count"].sum()) != int(row.effective_budget):
            problems.append(f"{strata_file.name}: selected_count sum "
                            f"{int(strata['selected_count'].sum())} != "
                            f"effective_budget {int(row.effective_budget)}")
        if int(strata["pool_count"].sum()) != int(row.pool_available):
            problems.append(f"{strata_file.name}: pool_count sum "
                            f"{int(strata['pool_count'].sum())} != "
                            f"pool_available {int(row.pool_available)}")

    # per-round batch files
    batch_sets: list[set[str]] = []
    for row in metrics.itertuples():
        if getattr(row, "effective_budget", 0) in (0, None):
            continue
        batch_file = ckpt / f"round_{int(row.round):03d}_batch.csv"
        if not batch_file.is_file():
            problems.append(f"missing {batch_file.name}")
            continue
        batch = pd.read_csv(batch_file)
        batch_sets.append(set(batch["candidate_id"]))
        if len(batch) != int(row.effective_budget):
            problems.append(f"{batch_file.name}: {len(batch)} rows != effective_budget "
                            f"{int(row.effective_budget)}")
        if batch["candidate_id"].duplicated().any():
            problems.append(f"{batch_file.name}: duplicated candidate_id")
        if batch["sequence"].duplicated().any():
            problems.append(f"{batch_file.name}: duplicated sequence")
        if int(row.n_labeled_before) != int(row.model_n_train):
            problems.append(f"round {int(row.round)}: model_n_train "
                            f"{int(row.model_n_train)} != n_labeled_before "
                            f"{int(row.n_labeled_before)}")
        if int(row.n_labeled_after) - int(row.n_labeled_before) != int(row.n_new_unique):
            problems.append(f"round {int(row.round)}: label growth "
                            f"{int(row.n_labeled_after) - int(row.n_labeled_before)} != "
                            f"n_new_unique {int(row.n_new_unique)}")
        if "ood_fraction" in metrics.columns:
            notes.append(f"round {int(row.round)}: ood_fraction={row.ood_fraction}")

    for i in range(len(batch_sets)):
        for j in range(i + 1, len(batch_sets)):
            overlap = batch_sets[i] & batch_sets[j]
            if overlap:
                problems.append(f"batches {i + 1} and {j + 1} overlap on {len(overlap)} ids")

    # Search-source batches must preserve actual parent and group provenance.
    if "source" in ledger and (ledger["source"] == "search_pool").any():
        required = ("parent_sequence", "parent_candidate_id", "parent_evidence",
                    "changed_positions_from_parent", "n_from_parent", "search_id",
                    "search_generation", "search_operator", "group_id", "group_role")
        missing = [c for c in required if c not in ledger.columns]
        if missing:
            problems.append(f"search ledger lacks {missing}")
        else:
            for round_no, sub in ledger[ledger["source"] == "search_pool"].groupby("run_round"):
                arch = ckpt / f"round_{int(round_no):03d}_search"
                paths = [arch / n for n in ("search_candidates.csv",
                         "search_evaluated.jsonl", "search_groups.json")]
                if any(not p.is_file() for p in paths):
                    problems.append(f"round {round_no}: search archive incomplete")
                    continue
                ranked = pd.read_csv(paths[0]).set_index("candidate_id")
                evaluated = {r["candidate_id"]: r for r in
                             (json.loads(line) for line in paths[1].read_text(
                                 encoding="utf-8").splitlines() if line.strip())}
                groups = json.loads(paths[2].read_text(encoding="utf-8"))
                all_member_ids = [cid for g in groups for cid in g["member_ids"]]
                member_group = {cid: g for g in groups for cid in g["member_ids"]}
                if (len(member_group) != len(ranked) or
                    len(all_member_ids) != len(ranked) or
                    ranked.index.has_duplicates or len(groups) not in (1, 2)):
                    problems.append(f"round {round_no}: groups do not partition ranked archive")
                for entry in sub.to_dict("records"):
                    cid = str(entry["candidate_id"])
                    if cid not in ranked.index or cid not in evaluated:
                        problems.append(f"{cid}: missing from search archive")
                        continue
                    src = evaluated[cid]
                    parent_id = src.get("parent_candidate_id") or ""
                    parent = evaluated.get(parent_id)
                    parent_seq = parent["sequence"] if parent else ""
                    if parent_id and parent is None:
                        problems.append(f"{cid}: missing parent {parent_id}")
                    changed = [p for p in MUT_POS if parent_seq and
                               normalize(entry["sequence"])[p-1] !=
                               normalize(parent_seq)[p-1]]
                    recorded = [int(x) for x in str(entry[
                        "changed_positions_from_parent"] or "").split(",")
                        if x.strip() and x.lower() != "nan"]
                    group = member_group.get(cid, {})
                    identity = (
                        str(entry["parent_candidate_id"])
                        if pd.notna(entry["parent_candidate_id"]) else "")
                    if (identity != parent_id or
                        str(entry["parent_sequence"])
                            != (parent_seq if parent_seq else "nan") or
                        str(entry["parent_evidence"]) != src["parent_evidence"] or
                        str(entry["search_operator"]) != src["operator"] or
                        int(entry["search_generation"]) != int(src["generation"]) or
                        str(entry["search_id"]) != src["search_id"] or
                        int(entry["n_from_parent"]) != len(changed) or
                        int(src["n_from_parent"]) != len(changed) or
                        recorded != changed or
                        list(src["changed_positions_from_parent"]) != changed or
                        str(entry["group_id"]) != group.get("group_id") or
                        str(entry["group_role"]) != group.get("group_role") or
                        cid != candidate_id(entry["sequence"])):
                        problems.append(f"{cid}: search lineage/group mismatch")
                manifest = json.loads((ckpt / f"round_{int(round_no):03d}_batch_manifest.json")
                                      .read_text(encoding="utf-8"))
                declared = manifest.get("search_handoff", {}).get("batch_group_counts")
                actual = sub["group_id"].value_counts().to_dict()
                if declared != actual:
                    problems.append(f"round {round_no}: search handoff group counts mismatch")
                handoff_name = manifest.get("search_handoff", {}).get("handoff_file")
                if handoff_name != "batch_handoff.csv":
                    problems.append(f"round {round_no}: search handoff file not declared")
                else:
                    batch_id = str(sub["batch_id"].iloc[0])
                    handoff_path = ckpt / "pending_batches" / "batches" \
                        / f"batch_{batch_id}" / handoff_name
                    if not handoff_path.is_file():
                        problems.append(f"round {round_no}: {handoff_name} missing")
                    else:
                        handoff = pd.read_csv(handoff_path)
                        cols = ["candidate_id", "sequence", "parent_sequence",
                                "parent_candidate_id", "parent_evidence",
                                "changed_positions_from_parent", "n_from_parent",
                                "search_id", "search_generation",
                                "search_operator", "group_id", "group_role"]
                        if any(c not in handoff for c in cols) or len(handoff) != len(sub):
                            problems.append(f"round {round_no}: handoff columns/count mismatch")
                        else:
                            left = sub[cols].sort_values("candidate_id").reset_index(drop=True)
                            right = handoff[cols].sort_values("candidate_id").reset_index(drop=True)
                            if not left.equals(right):
                                problems.append(f"round {round_no}: handoff lineage/group mismatch")
                            if ("protocol_id" not in handoff or
                                not (handoff["protocol_id"] == manifest.get(
                                    "protocol_id")).all()):
                                problems.append(f"round {round_no}: handoff protocol mismatch")

    if ledger["candidate_id"].duplicated().any():
        problems.append("ledger contains duplicate candidate_id rows")
    if ledger["sigma_gbsa"].notna().any():
        if (ledger["sigma_gbsa"].dropna() <= 0).any():
            problems.append("ledger has non-positive sigma_gbsa")
    else:
        notes.append("sigma_gbsa empty for every row (surrogate provides no sigma) — allowed")
    if ledger["pred_gbsa"].isna().any() or not ledger["pred_gbsa"].apply(
            lambda v: abs(float(v)) != float("inf")).all():
        problems.append("ledger has missing/infinite pred_gbsa")
    if "n_from_wt" in ledger.columns:
        if ledger["n_from_wt"].isna().any():
            problems.append("ledger has candidates without n_from_wt provenance")
        notes.append(f"n_from_wt range [{int(ledger['n_from_wt'].min())},"
                     f"{int(ledger['n_from_wt'].max())}]")
    if "model_id" in ledger.columns:
        notes.append("model_id(s): " + ", ".join(sorted(set(ledger["model_id"].astype(str)))))

    # The ledger holds one row per *exported* candidate: rows with status
    # 'awaiting_labels' are legitimately not counted in n_new_unique yet.
    revealed = ledger[ledger["status"].astype(str) == "labeled"]
    total_new = int(metrics["n_new_unique"].sum())
    if total_new != len(revealed):
        problems.append(f"sum(n_new_unique) {total_new} != revealed ledger rows {len(revealed)}")
    pending_rows = len(ledger) - len(revealed)
    if marker == "PARKED" and pending_rows == 0:
        problems.append("PARKED run has no awaiting_labels rows in the ledger")
    notes.append(f"ledger: {len(revealed)} revealed + {pending_rows} pending = {len(ledger)} rows")

    for note in notes:
        print(f"[note] {note}")
    if problems:
        for p in problems:
            print(f"[FAIL] {p}")
        return 1
    print(f"[ok] {run_dir.name}: {marker} run satisfies §11.1 invariants "
          f"({len(batch_sets)} batch(es), {len(ledger)} ledger rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
