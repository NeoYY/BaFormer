#!/bin/bash
# Evaluate the trained ASRF Breakfast split1 model once training completes.
# Usage: bash eval_asrf.sh [checkpoint_file]
set -e
cd /home/neo/BaFormer
source .venv-baformer/bin/activate
ASRF=baselines/repos/asrf
RES=$ASRF/result/breakfast/dataset-breakfast_split-1
CKPT=${1:-$RES/final_model.prm}
echo "Using checkpoint: $CKPT"

cd $ASRF
python infer_boundary.py result/breakfast/dataset-breakfast_split-1/config.yaml \
  --checkpoint "$CKPT" \
  --out_dir /home/neo/BaFormer/baselines/predictions/ASRF_s1 \
  --bd_dir /home/neo/BaFormer/baselines/predictions/ASRF_s1_bd

cd /home/neo/BaFormer
# Segmentation metrics (boundary-refined predictions) + seg-transition boundary metric
python baselines/common/eval_tas.py \
  --pred_dir baselines/predictions/ASRF_s1 \
  --gt_dir /home/neo/datasets/data/breakfast/groundTruth \
  --bundle /home/neo/datasets/data/breakfast/splits/test.split1.bundle \
  --dataset breakfast --model "ASRF (WACV21)" \
  --out_json baselines/results/ASRF_s1.json

# Replace boundary block with ASRF's DEDICATED boundary-regression branch
cd baselines/common
python boundary_from_probs.py \
  --bd_dir /home/neo/BaFormer/baselines/predictions/ASRF_s1_bd \
  --gt_dir /home/neo/datasets/data/breakfast/groundTruth \
  --bundle /home/neo/datasets/data/breakfast/splits/test.split1.bundle \
  --threshold 0.5 --model "ASRF (WACV21)" \
  --merge_into /home/neo/BaFormer/baselines/results/ASRF_s1.json

cd /home/neo/BaFormer
python baselines/common/summarize.py
