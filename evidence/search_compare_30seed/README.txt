Equal-budget search-policy comparison (plan 阶段 B).
Every policy sees the same train split, surrogate snapshot and hard
unique-prediction budget. best_pred_gbsa comes from the SURROGATE, so it
is a computational diagnostic, not a discovery claim; real GBSA
discovery needs 阶段 C with real measurements. Arm B (retrain
consistency) re-runs the first --consistency-seeds seeds with model
seed +1000. argv: --policies stratified_random,multi_start_hill_climb,multi_start_beam --seeds 30 --seed-start 201 --consistency-seeds 10 --budget 8192 --n-init 300 --surrogate xgb --n-members 5 --screen-k 100 --max-generations 8 --allowed-n-from-wt 4:13 --beam-width 32 --offspring-per-parent 32 --one-hop-fraction 0.8 --restart-fraction 0.1 --diversity-pool-factor 4 --diversity-distance 2 --max-stagnant-generations 2 --n-measured-seeds 8 --n-random-seeds 8 --min-mutual-distance 2 --data-dir data/proxy2000_v2 --out-dir evidence/search_compare_30seed --output runs/srv_search_compare_30seed_20261002_024621/metrics.csv
