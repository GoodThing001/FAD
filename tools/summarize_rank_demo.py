"""Summarize rank objectives + candidate demo."""
import json
from pathlib import Path

E = Path(__file__).resolve().parents[1] / "evidence/audit_20260924/ml_squeeze"
d = json.load(open(E / "rank_objectives2.json"))
print("=== RANK OBJECTIVES (30 seeds) ===")
for k, v in d["summary"].items():
    print(f"{k:<22} mean={v['mean']:.4f} std={v['std']:.4f} "
          f"delta={v['paired_delta_vs_mse']:+.4f} wins={v['wins']}/{v['n_seeds']}")
print()
print("=== CANDIDATE DEMO ===")
print(json.dumps(json.load(open(E / "candidate_demo_summary.json")), indent=2))
import pandas as pd
top = pd.read_csv(E / "candidate_top200.csv")
print("\ntop-5 candidates:")
print(top[["rank", "gbsa_fusion_rank", "dock_fusion_rank", "mfe_rank",
           "full_sequence_mfe"]].head(5).to_string(index=False))
