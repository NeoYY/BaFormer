# Baselines: SOTA temporal-action-segmentation / boundary-detection models on Breakfast

This directory benchmarks **five additional SOTA models** against BaFormer on the
**Breakfast** dataset (split 1, I3D features), using the **same metrics as BaFormer's
`Inference.py`** so results are directly comparable. Everything here is isolated from
the running BaFormer training (BaFormer core code is untouched).

> Note: the referenced `class_agnostic_boundary_detection_summary.pdf` was not present
> on disk, so the candidate models were selected from the temporal-action-segmentation /
> action-boundary-detection literature (all consume the identical Breakfast I3D features,
> so no video re-extraction was needed).

## Models & why they were chosen
| Model | Venue | Weights | Relevance |
|-------|-------|---------|-----------|
| **DiffAct** | ICCV 2023 | shipped in repo | current SOTA diffusion segmentation; has a boundary conditioning |
| **LTContext** | ICCV 2023 | Dropbox | long-term + windowed attention; strong recent model |
| **C2F-TCN** | TPAMI 2023 | Google Drive | coarse-to-fine multi-resolution TCN |
| **ASFormer** | BMVC 2021 | Google Drive | transformer w/ iterative decoder refinement |
| **ASRF** | WACV 2021 | trained here (no public weights) | **dedicated class-agnostic boundary-regression branch** — the most on-topic; directly comparable to BaFormer's boundary head |

Preference was given to models with **pre-trained weights** (4/5). ASRF has no public
weights, so it was trained on Breakfast split1 (the boundary comparison it enables is the
key scientific payoff).

## Layout
```
baselines/
  common/
    eval_tas.py            # shared evaluator = BaFormer metrics (Acc/Edit/F1) + boundary P/R/F1
    boundary_from_probs.py # boundary-detection metric from a model's boundary-prob signal
    summarize.py           # builds results/COMPARISON.md
  repos/                   # cloned upstream repos (+ *_only.py inference wrappers)
    DiffAct/ ASFormer/ LTContext/ C2F-TCN/ asrf/ MS-TCN2/
  weights/                 # downloaded pretrained checkpoints
  predictions/             # per-model recognition files (one per test video)
  results/                 # per-model JSON + COMPARISON.md
  c2f_base/ asrf_dataset/  # isolated data dirs (symlinks) for repos needing special layout
  eval_asrf.sh             # evaluates trained ASRF
  watch_finalize_asrf.sh   # detached: auto-evaluates ASRF at 50 epochs
```

## Metrics
`common/eval_tas.py` reproduces BaFormer exactly:
- **Acc** frame micro-accuracy; **Edit** per-video mean; **F1@{10,25,50}** global tp/fp/fn.
- **Class-agnostic boundary detection**: predicted boundaries = frames where the label
  sequence changes; greedy one-to-one match to GT boundaries within tol {0,1,2,4} frames;
  report P / R / micro-F1 / macro-F1. For ASRF the boundary metric is computed from its
  **dedicated boundary-regression branch** (`boundary_from_probs.py`).

Data: `/home/neo/datasets/data/breakfast`  •  test split1 = 252 videos, 48 classes.
Environment reused: `../.venv-baformer` (torch 2.4.1).

## Reproduce
```bash
source ../.venv-baformer/bin/activate

# DiffAct (weights shipped in repo)
cd repos/DiffAct && python infer_only.py --config configs/Breakfast-Trained-S1.json \
  --model_path trained_models/Breakfast-Trained-S1/release.model \
  --out_dir ../../predictions/DiffAct_s1 --device 0

# ASFormer (weights: weights/asformer_models/models/breakfast/split_1/epoch-120.model)
cd repos/ASFormer && python infer_only.py --dataset breakfast --split 1 \
  --model_file ../../weights/asformer_models/models/breakfast/split_1/epoch-120.model \
  --out_dir ../../predictions/ASFormer_s1

# LTContext (weights: weights/breakfast_split1_ltc.pth) -> saves pred.npy, then convert
cd repos/LTContext && python run_net.py --cfg configs/Breakfast/LTContext.yaml \
  DATA.PATH_TO_DATA_DIR /home/neo/datasets/data/breakfast TRAIN.ENABLE False TEST.ENABLE True \
  TEST.CHECKPOINT_PATH ../../weights/breakfast_split1_ltc.pth DATA.CV_SPLIT_NUM 1 \
  TEST.SAVE_PREDICTIONS True TEST.SAVE_RESULT_PATH ../../predictions/LTContext_s1_raw

# C2F-TCN (weights: weights/best_breakfast_unet.wt ; NOTE uses its OWN class order -> c2f_base/mapping.csv)
cd repos/C2F-TCN && python eval.py --dataset_name breakfast --split 1 --cudad 0 \
  --base_dir ../../c2f_base/ --compile_result \
  --model_checkpoint ../../c2f_base/results/supervised_C2FTCN/split1/best_breakfast_unet.wt

# ASRF (trained here): bash eval_asrf.sh <checkpoint>

# score any recognition dir with the shared harness:
python common/eval_tas.py --pred_dir predictions/<M>_s1 \
  --gt_dir /home/neo/datasets/data/breakfast/groundTruth \
  --bundle /home/neo/datasets/data/breakfast/splits/test.split1.bundle \
  --dataset breakfast --model <M> --out_json results/<M>_s1.json

python common/summarize.py   # -> results/COMPARISON.md
```

See `results/COMPARISON.md` for the aggregated table.
