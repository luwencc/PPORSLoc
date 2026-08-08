# ARISLoc

PPO-based deep reinforcement learning for **path–phase** selection on a RIS / metasurface localization pipeline.

Uses the frozen **RIS-FLoc** localizer (`RISFLoc.MixedModel`) as the oracle for localization error.

## Overview

- **Actions**: phase `0…5` (extend next path segment) + **stop**
- **Observation**: flattened sparse phase–path RSS grid + meta features
- **Reward**: MADRL-CIL style (step cost + terminal normalized error)

## Repository layout

```
ARISLoc/
├── ARISLoc.py             # CLI entry point
├── requirements.txt
└── arisloc/
    ├── config.py           # PPO / env hyperparameters
    ├── caches.py           # traj & inference LRU caches
    ├── traj.py             # folder keys / path-phase helpers
    ├── localizer.py        # RISFLoc inference engine
    ├── env.py              # Gymnasium environment
    ├── policy.py           # Actor-Critic
    ├── ppo.py              # GAE + PPO update
    ├── plots.py            # curves / CDF / stats
    └── train.py            # training loop
```

## Requirements

```bash
pip install -r requirements.txt
```

Also install and train **RIS-FLoc** first so these files exist (paths in `arisloc/config.py`):

- `best_model_DRL.pth`
- `mean_DRL.npy` / `std_DRL.npy`
- `ordinate.csv` (place next to the scripts; download with the dataset release if needed)

## Data

Download datasets from **https://github.com/luwencc/ARISLoc/releases/tag/Data**.

For ARISLoc training, point `--drl-root` at your trajectory tree (default in config is `Test_Data`):

```bash
python ARISLoc.py --drl-root /path/to/DRL_RP
```

## Usage

```bash
cd ARISLoc
python ARISLoc.py --device cuda:0 --episodes 2500
```

Useful flags: `--synthetic`, `--cache-gb`, `--loc-batch`, `--eval-greedy`.

## Dependency on RIS-FLoc

ARISLoc imports the sibling package:

```python
import RISFLoc as loc
```

Keep this layout when uploading to GitHub:

```
GITHUB/
├── RIS-FLoc/
│   └── RISFLoc.py
└── ARISLoc/
    └── ARISLoc.py
```

## Outputs

| File | Description |
|------|-------------|
| `drl_ppo_path_phase.pt` | Trained policy |
| `drl_ppo_train_curves_panel.png` | Training curves |
| `drl_ppo_error_cdf.png` | Eval CDF |
| `drl_ppo_episode_returns.csv` | Episode logs |

## License

Add your license here before publishing.
