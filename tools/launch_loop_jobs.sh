#!/usr/bin/env bash
# Launch the FAD closed-loop jobs on the shared server, politely.
#   bash tools/launch_loop_jobs.sh                # park + original 30-seed compare
#   bash tools/launch_loop_jobs.sh compare_v2     # fixed-code 30-seed compare
#   bash tools/launch_loop_jobs.sh handoff        # real pending batch (PARKED)
#   bash tools/launch_loop_jobs.sh confirm_100seed  # pre-registered 100-seed confirmation
# Always: 2 BLAS threads, nice 19, detached from the terminal.
set -u
cd /home/hzeng/project/FAD_CLEAN || exit 1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
PY=/home/hzeng/envs/FAD_env/bin/python
STAMP=$(date +%Y%m%d_%H%M%S)
MODE="${1:-default}"

run() {                       # run <run-id> <config> <logfile>
  echo "[launch] $1 <- $2"
  nice -n 19 "$PY" scripts/run_experiment.py "$2" --run-id "$1" >> "$3" 2>&1
  echo "[done] $1 rc=$?" >> "$3"
}

case "$MODE" in
  compare_v2)
    run "srv_strategy_compare_30seed_v2_$STAMP" \
        "configs/experiments/loop_strategy_compare_30seed_v2.json" \
        "/tmp/srv_strategy_compare_v2.log"
    ;;
  handoff)
    run "srv_handoff_$STAMP" "configs/experiments/loop_handoff_96.json" "/tmp/srv_handoff.log"
    ;;
  confirm_100seed)
    run "srv_strategy_compare_100seed_$STAMP" \
        "configs/experiments/loop_strategy_compare_100seed.json" \
        "/tmp/srv_strategy_compare_100seed.log"
    ;;
  search_8192)
    run "srv_search_dev_beam_8192_$STAMP" \
        "configs/experiments/search_dev_beam_8192.json" \
        "/tmp/srv_search_8192.log"
    ;;
  search_compare_30)
    run "srv_search_compare_30seed_$STAMP" \
        "configs/experiments/search_compare_30seed.json" \
        "/tmp/srv_search_compare_30seed.log"
    ;;
  sigma_calib_30)
    run "srv_sigma_calibration_30seed_$STAMP" \
        "configs/experiments/eval_sigma_calibration_30seed.json" \
        "/tmp/srv_sigma_calibration_30seed.log"
    ;;
  *)
    run "srv_loop_park_v2_$STAMP" "configs/experiments/loop_prospective_park.json" "/tmp/srv_park_v2.log"
    run "srv_strategy_compare_30seed_$STAMP" "configs/experiments/loop_strategy_compare_30seed.json" "/tmp/srv_strategy_compare.log"
    ;;
esac
echo "[all done] $MODE $STAMP" >> /tmp/loop_jobs.log
