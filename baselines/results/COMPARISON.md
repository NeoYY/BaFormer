# Breakfast split1 — baseline comparison

## Segmentation metrics (standard MS-TCN eval)

| Model                  |    Acc |   Edit |  F1@10 |  F1@25 |  F1@50 |
|------------------------|--------|--------|--------|--------|--------|
| DiffAct (ICCV23)       |  77.40 |  77.94 |  80.27 |  76.95 |  66.36 |
| ASFormer (BMVC21)      |  73.59 |  75.50 |  75.47 |  70.29 |  58.71 |
| LTContext (ICCV23)     |  76.54 |  76.73 |  77.07 |  72.58 |  61.96 |
| C2F-TCN (TPAMI23)      |  74.12 |  73.30 |  75.16 |  70.09 |  58.06 |
| ASRF (WACV21)          |  65.49 |  71.65 |  72.08 |  65.52 |  50.93 |

## Class-agnostic boundary detection (micro-F1 %)

| Model                  |  tol=0 |  tol=1 |  tol=2 |  tol=4 |
|------------------------|--------|--------|--------|--------|
| DiffAct (ICCV23)       |   2.54 |   7.16 |  12.31 |  19.94 |
| ASFormer (BMVC21)      |   1.34 |   4.23 |   7.11 |  12.87 |
| LTContext (ICCV23)     |   1.02 |   2.86 |   5.15 |  10.61 |
| C2F-TCN (TPAMI23)      |   1.80 |   3.80 |   6.06 |   8.86 |
| ASRF (WACV21) (BR-branch) |   2.08 |   6.47 |   9.89 |  14.67 |
