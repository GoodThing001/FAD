"""Summarize the saturated replica-allocation run."""
import json
from pathlib import Path

E = Path(__file__).resolve().parents[1] / "evidence/audit_20260924"
h = json.load(open(E / "replica_alloc_saturated_results.json"))
print("=== SATURATED (n_init=1400, hetero, sigma=5) ===")
for sigma, s in h.items():
    for name, r in s.items():
        tc = r.get("top100_spearman_mean_by_round", [])
        t100 = r.get("final_top100_spearman", float("nan"))
        print(f"{name:<8} final_sp={r['final_spearman']:.4f} "
              f"final_top100={t100:.4f} final_noise={r['final_noise_std']:.2f} "
              f"top100_curve={[round(v, 3) for v in tc]}")
