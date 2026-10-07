Retrospective proxy-oracle strategy comparison.
Labels come from the proxy table through TableOracle: only pool candidates can
be revealed, the holdout is never queried, and every strategy sees the same
split, pool, budget and seed. These numbers rank strategies on the PROXY target
only; they are not wet-lab evidence.
argv: --mode retrospective --strategies random,greedy,greedy_diverse,maxmin_coverage --seeds 3 --seed-start 1 --rounds 5 --budget 50 --n-init 300 --n-holdout 200 --pool-size 600 --surrogate xgb --n-members 5 --out-dir evidence/loop_20260930/strategy_compare_3seed --output runs\local_strategy_compare_3seed\metrics.csv
