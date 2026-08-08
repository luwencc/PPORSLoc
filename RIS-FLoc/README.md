# RIS-FLoc

Sparse path–phase RSSI fingerprinting for reference-point (RP) classification and indoor localization.

## Overview

RIS-FLoc trains a dual-branch **CNN–LSTM–SE** network on a sparse `phase × path` RSS grid:

- **Classification**: RP classes (loaded from `ordinate.csv` column names)
- **Regression**: planar `(x, y)` head fused with Top-K soft RP coordinates at test time

## Repository layout

```
RIS-FLoc/
├── RISFLoc.py                 # entry point & public import API
├── requirements.txt
└── risfloc/
    ├── config.py           # hyperparameters
    ├── model.py            # MixedModel
    ├── normalize.py        # z-score helpers
    ├── data.py             # datasets / caches / grids
    ├── train.py            # training loop
    ├── evaluate.py         # test evaluation & plots
    └── utils.py            # AMP / DataParallel helpers
```

## Requirements

- Python 3.9+
- CUDA-capable GPU recommended

```bash
pip install -r requirements.txt
```

## Data

Download archives from the repository Releases page:

**https://github.com/luwencc/ARISLoc/releases/tag/Data**

Extract `Trained_Data/`, `Validation_Data/`, `Test_Data/`, and place `ordinate.csv` next to `RISFLoc.py` (or set paths in `risfloc/config.py`).

## Usage

```bash
cd RIS-FLoc
python RISFLoc.py
```

Set `RETRAIN = False` in `RISFLoc.py` to evaluate with existing `best_model_DRL.pth`, `mean_DRL.npy`, and `std_DRL.npy`.

Tune batch size, GPUs, and AMP flags in `risfloc/config.py`.

## Outputs

| File | Description |
|------|-------------|
| `best_model_DRL.pth` | Best validation checkpoint |
| `mean_DRL.npy` / `std_DRL.npy` | Train z-score stats |
| `Test_picture_DRL/` | Curves, CDF, confusion matrix |

## Import from ARISLoc

```python
import RISFLoc as loc
model = loc.MixedModel()
```

Ensure the `RIS-FLoc` directory is on `sys.path` (ARISLoc does this automatically).

## License

Add your license here before publishing.
