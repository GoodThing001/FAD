"""Summarize the ml-squeeze batch results."""
import json
from pathlib import Path

E = Path(__file__).resolve().parents[1] / "evidence/audit_20260924/ml_squeeze"

d = json.load(open(E / "negative_controls.json"))
print("=== NEGATIVE CONTROLS (10 seeds) ===")
for k, v in d.items():
    print(f"  {k:<18} mean={v['mean']:.4f} std={v['std']:.4f}")

d = json.load(open(E / "smoothing_variants.json"))
print("\n=== SMOOTHING VARIANTS (30 seeds, paired vs raw) ===")
for k, v in d["summary"].items():
    if k == "raw":
        print(f"  {k:<12} mean={v['mean']:.4f} std={v['std']:.4f}")
    else:
        print(f"  {k:<12} mean={v['mean']:.4f} delta={v['paired_delta_vs_raw']:+.4f} "
              f"ci=[{v['paired_ci95'][0]:.4f},{v['paired_ci95'][1]:.4f}] "
              f"wins={v['wins']}/{v['n_seeds']}")

card = json.load(open(E / "release_package/model_card.json"))
print("\n=== MODEL CARD ===")
print(json.dumps(card["reference_metrics"], indent=2))

m = json.load(open(E / "server_manifest_summary.json"))
print("\n=== SERVER INVENTORY SUMMARY ===")
print(json.dumps(m, indent=2)[:1500])
