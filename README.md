# ARISLoc: RIS-Assisted Single Mobile Site Localization via Adaptive Phase Reconfiguration

This repository contains two complementary projects for RIS / metasurface-aided indoor localization from RSSI fingerprints.

| Project | Role |
|---------|------|
| [RIS-FLoc](RIS-FLoc/) | Supervised CNN–LSTM–SE localizer (RP classification + xy) |
| [ARISLoc](ARISLoc/) | PPO agent that chooses path/phase measurement sequences |

## Data

Download datasets from **[Releases](https://github.com/luwencc/ARISLoc/releases/tag/Data)**:

1. Open the link above (or: repo → **Releases** → **Dataset** / tag `Data`).
2. Download `Trained_Data.tar.zip`, `Validation_Data.tar.zip`, and `Test_Data.tar.zip`.
3. Extract them at the repository root (or paths set in config):

```bash
tar -xf Trained_Data.tar.zip
tar -xf Validation_Data.tar.zip
tar -xf Test_Data.tar.zip
```

See also [data/README.md](data/README.md).

## Quick start

1. Train the localizer:

```bash
cd RIS-FLoc
pip install -r requirements.txt
python RISFLoc.py
```

2. Train the PPO policy (needs RIS-FLoc weights + DRL trajectory data):

```bash
cd ../ARISLoc
pip install -r requirements.txt
python ARISLoc.py --drl-root /path/to/DRL_RP --device cuda:0
```

## Relationship

```
RSS CSV ──► RIS-FLoc (MixedModel) ──► localization error / coords
                ▲
                │ frozen oracle
ARISLoc (PPO) ─┘ chooses next phase or stop
```

## License

Add a license before publishing to GitHub.
