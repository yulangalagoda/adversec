# AdverSec

Adversarial robustness of ML-based intrusion detection for in-vehicle CAN-bus
networks, across two datasets — **CICIoV2024** and **ROAD** — run through one
identical, leakage-controlled pipeline. The study asks *when* adversarial training
helps: it aids the signature-scarce dataset (CICIoV2024) and fails on the
signature-rich one (ROAD), a contingency governed by data structure rather than by
the defence itself.

## Design in one rule

The only code that knows which dataset it is, is a thin **adapter**. Everything
downstream receives a canonical CAN table (`ID, DATA_0..DATA_7, true_class`) and
cannot tell CICIoV2024 from ROAD. That is the study's claim expressed as
architecture: one pipeline, only the data differs.

```
adversec/
├── contract.py          canonical schema + validate()   (the keystone)
├── config.py            shared knobs only                (per-dataset -> configs/*.yaml)
├── datasets/            the ONLY dataset-aware code: CANDataset -> CICIoVDataset, ROADDataset
├── pipeline/            dataset-agnostic: dedup -> split -> augment -> encode
├── models/              rf.py, cnn.py
├── experiments/         baseline, attack (robustness+distance+threat-sizing), defended, crossval, realism
├── evaluation.py        shared metrics (macro-F1, robust-support, per-class)
└── cli.py               entry point
configs/                 ciciov2024.yaml, road.yaml   (paths, class maps, defence shape)
notebooks/               01..06, step-by-step drivers; each stage saves its own results/*.json
results/                 citable JSON reports, one file per stage per dataset (tracked in git)
tests/                   migration gates (numbers must reproduce the committed artifacts)
```

## Install

```bash
pip install -e .
```

Exact reproduction uses the pins in `requirements.txt` (torch 2.5.1+cu121,
adversarial-robustness-toolbox 1.20.1, scikit-learn 1.8.0, numpy 2.4.4,
pandas 3.0.3) on a CUDA GPU. **Never install the PyPI package `art`** (an ASCII-art
library) — it shadows the Adversarial Robustness Toolbox and breaks imports.

## Data

- **CICIoV2024** (decimal): six per-class CSVs in `datasets/raw/ciciov2024_decimal/`.
- **ROAD**: candump captures under `datasets/raw/road/` (`ambient/`, `attacks/` +
  `capture_metadata.json`).

Raw data is gitignored — it's multi-GB (ROAD raw alone is ~3GB) and must be
sourced separately, placed under `datasets/raw/`. `datasets/processed/` (the
output of notebooks 01+02, or `adversec prep`) is small (~8MB total) and **is**
tracked in git, so notebooks 03–06 can run on a fresh clone without needing the
raw data at all.

## Reproduce

Each stage writes a citable JSON/artifact; run per dataset (`ciciov2024` or `road`):

```bash
adversec prep     --dataset road                 # canonical -> dedup -> split -> augment -> arrays
adversec baseline --dataset road --device cuda    # RF + CNN clean-data reference
adversec attack   --dataset road --device cuda    # CV per-class robustness + distance + threat-sizing
adversec defend   --dataset road --device cuda    # defended-model cross-validation
```

`adversec defend` runs the per-dataset-shaped comparison declared in
`configs/*.yaml` (robust-support CV for CICIoV2024, per-class comparison for
ROAD). `notebooks/05_defence` runs a different, newer methodology instead — see
[Notebooks](#notebooks) below — so the two paths are not interchangeable and
currently produce different `<name>_defence_results.json` schemas.

## Notebooks

`notebooks/01..06` are step-by-step drivers over the same package code as the CLI,
meant for interactive/report use. Each stage saves a citable artifact, so any
notebook can be run on its own once its inputs exist on disk:

| Notebook | Reads | Saves |
|---|---|---|
| `01_cleaning` | raw data | `<name>_{strict,train,test}.csv`, `results/<name>_dedup_audit.json` |
| `02_preprocessing` | 01's `train.csv`/`test.csv` | `<name>_stage2_arrays.npz`, `<name>_train_dup.csv`, scaler/encoder, `<name>_prep_summary.json` |
| `03_baselines` | 02's arrays + encoder | `results/<name>_baseline_metrics.json` |
| `04_adversarial_attacks` | 02's arrays + encoder, 01's `strict.csv` | `results/<name>_adversarial_results.json` (per-class PGD CV + distance-to-benign) |
| `05_defence` | 02's arrays + encoder | `results/<name>_defence_results.json` (`two_threat_model_averaged` schema — see below) |
| `06_threat_sizing` | 02's arrays/scaler/encoder, 01/02's `train_dup.csv`/`test.csv` | `results/<name>_adversarial_results.json` (adds `threat_sizing`; merge-safe with 04, either order) |

Run `01` then `02` once per dataset; `03`–`06` only depend on `02`'s output, not
on each other, so any one of them can be re-run standalone.

**`05_defence`'s methodology differs from `adversec defend`.** The CLI/`experiments/defended.py`
path runs the per-dataset-shaped comparison from `configs/*.yaml` (robust-support
CV for CICIoV2024, per-class comparison for ROAD — the older design). The notebook
instead treats both datasets identically: it evaluates clean / transfer-attack /
white-box-attack robust-support macro-F1 for the baseline CNN, a PGD-adversarially-trained
CNN, and the Random Forest, under PGD eps=0.10, averaged (mean ± std) over
`N_REPEATS` seeded runs. This is the current, decisive result; the CLI path's
`robust_support_cv`/`perclass_comparison` schemas predate it and are kept for the
migration-gate tests, not as the primary defence claim.

## Verify (migration gates)

Every stage must reproduce the committed numbers:

```bash
python tests/test_adapters.py       # loaders -> canonical table matches the legacy loader
python tests/test_pipeline.py       # CIC dedup/split/augment/encode match committed artifacts
python tests/test_pipeline_road.py  # same for ROAD
python tests/test_baseline.py       # RF exact, CNN within tolerance
```

Threat-sizing numbers intentionally differ from older runs: the realism integer
clip is now per-feature, so the arbitration ID (0–2047) is no longer crushed to a
byte range.

## Key findings

- The ~99.75% duplication in CICIoV2024 inflates accuracy; on honestly
  de-duplicated data the picture is very different (the "accuracy trap").
- Adversarial training's benefit is contingent on data structure: it helps the
  signature-scarce CICIoV2024 attack classes and fails/degrades on the
  signature-rich ROAD classes.
- A cheap per-ID envelope validator rejects most naive gradient attacks on ROAD —
  a known constrained-domain result, not a novel defence.
