#!/bin/bash
# Detached watcher: wait until ASRF training writes final_model.prm (50 epochs),
# then run the full ASRF evaluation + refresh the comparison table.
RES=/home/neo/BaFormer/baselines/repos/asrf/result/breakfast/dataset-breakfast_split-1
LOG=/home/neo/BaFormer/baselines/results/asrf_finalize.log
echo "[watcher $(date)] waiting for $RES/final_model.prm" > "$LOG"
for i in $(seq 1 240); do
  if [ -f "$RES/final_model.prm" ]; then
    echo "[watcher $(date)] final_model.prm found, evaluating (50 epochs)" >> "$LOG"
    bash /home/neo/BaFormer/baselines/eval_asrf.sh "$RES/final_model.prm" >> "$LOG" 2>&1
    echo "[watcher $(date)] ASRF finalized." >> "$LOG"
    exit 0
  fi
  sleep 60
done
echo "[watcher $(date)] timed out waiting for final_model.prm" >> "$LOG"
