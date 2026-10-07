"""Summarize the replica-allocation results (homo + hetero) and verify the
formal dock package."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
E = ROOT / "evidence/audit_20260924"

d = json.load(open(E / "formal_dock_fusion_100seed/metrics_summary.json"))
print("FORMAL dock summary:")
for k, v in d.items():
    if isinstance(v, dict):
        if "mean" in v:
            print(f"  {k:<14} mean={v['mean']:.4f} std={v['std']:.4f}")
        else:
            for m, s in v.items():
                print(f"  {k}.{m:<9} mean={s['mean']:.4f} std={s['std']:.4f}")
print()

for tag, fn in [("HOMOSCEDASTIC", "replica_alloc_homo_results.json"),
                ("HETEROSCEDASTIC", "replica_alloc_hetero_results.json")]:
    h = json.load(open(E / fn))
    print(f"=== {tag} (top100 = Spearman within the 100 truly-best holdout seqs) ===")
    for sigma, s in h.items():
        print(f"-- sigma_meas={sigma} --")
        for name, r in s.items():
            tc = r.get("top100_spearman_mean_by_round", [])
            t100 = r.get("final_top100_spearman", float("nan"))
            print(f"  {name:<8} final_sp={r['final_spearman']:.4f} "
                  f"final_top100={t100:.4f} "
                  f"final_noise={r['final_noise_std']:.2f} "
                  f"top100_curve={[round(v, 3) for v in tc]}")
    print()
