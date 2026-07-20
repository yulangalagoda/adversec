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
| `06_threat_sizing` | 02's arrays/scaler/encoder, 01/02's `train_dup.csv`/`test.csv` | `results/<name>_adversarial_results.json` (adds `threat_sizing`, `adaptive_envelope_aware_attack`, `blackbox_hopskipjump_attack`; merge-safe with 04, either order) |

Run `01` then `02` once per dataset; `03`–`06` only depend on `02`'s output, not
on each other, so any one of them can be re-run standalone.

**`05_defence`'s methodology differs from `adversec defend`.** The CLI/`experiments/defended.py`
path runs the per-dataset-shaped comparison from `configs/*.yaml` (robust-support
CV for CICIoV2024, per-class comparison for ROAD — the older design). The notebook
instead treats both datasets identically: it evaluates clean / transfer-attack /
white-box-attack robust-support macro-F1 for the baseline CNN, a PGD-adversarially-trained
CNN, and the Random Forest, under PGD eps=0.10, averaged (mean ± std) over
`N_REPEATS=10` seeded runs, plus a paired t-test (`defended[r] - baseline[r]` per seed)
confirming which differences are real rather than noise. This is the current, decisive
result; the CLI path's `robust_support_cv`/`perclass_comparison` schemas predate it and
are kept for the migration-gate tests, not as the primary defence claim.

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

- **The accuracy trap**: ~99.75% duplication in CICIoV2024 (1,408,219 rows → 3,588
  unique signatures) inflates accuracy; on honestly de-duplicated data both baselines'
  macro-F1 collapses well below their near-perfect accuracy (RF: 0.997 acc / 0.776
  macro-F1). ROAD is far less duplicated (39.8%) and its clean baselines are genuinely
  near-perfect (RF macro-F1 = 1.0), not just accuracy-trapped.
- **Adversarial training's benefit is contingent on data structure, confirmed by a
  paired t-test over 10 repeats, not just by comparing means**: AT helps significantly
  under transfer attack on both datasets (CICIoV2024: p=0.0001; ROAD: p<0.0001). Under
  white-box attack, AT **significantly backfires on ROAD** (defended model worse than
  the undefended baseline's own robustness, p=0.0035) but has **no significant effect
  either way on CICIoV2024** (p=0.158) — a genuine null result, not "fails the same way."
- **The per-ID envelope validator's rejection rate is now false-positive-audited, not
  just asserted**: it rejects the large majority of naive PGD attacks on both datasets
  (CICIoV2024: 100% at every epsilon tested; ROAD: 90.1–90.2%) at a low, credible
  false-positive cost on held-out legitimate traffic (CICIoV2024: 3.2%; ROAD: 1.3%).
  This confirms CICIoV2024's 100% figure is a real result, not an artifact of an
  over-narrow envelope — it was checked, not assumed.
- **The distance-to-benign robustness mechanism inverts on CICIoV2024** (pearson_r =
  -0.59, vs +0.706 on ROAD — classes *closer* to benign are the *more* robust ones,
  the opposite pattern to ROAD). An ablation removing the light-duplication
  convergence crutch shows the inversion survives (r = -0.295) — same sign, roughly
  half the magnitude — so it is a real, if amplified-by-duplication, data-structure
  effect, not an artifact of the padding.
- **In progress**: an adaptive envelope-aware attacker (ID frozen to a real value,
  bytes clipped into that ID's own observed range) and a non-gradient black-box
  attack (HopSkipJump, reduced query budget) are implemented in `notebooks/06` but
  not yet run to completion — see [Limitations](#limitations).

## Limitations

- **CICIoV2024's statistical power is thin.** After strict de-duplication, 3 of 6
  classes have exactly 1 test signature (`spoofing-GAS`, `spoofing-SPEED`,
  `spoofing-STEERING_WHEEL`); `DoS` has 4, `spoofing-RPM` has 2. Per-class F1s and
  the distance-to-benign correlation (n=5 classes) on this dataset should be read as
  indicative, not statistically definitive — neither the with- nor without-duplication
  pearson_r clears the conventional significance bar at that sample size. ROAD's
  statistics are much better-powered (test classes range 118–4,238 signatures) and
  should be weighted more heavily in any claim that needs to generalise.
- **ROAD excludes masquerade attack variants and the correlated-signal attack**
  (`configs/road.yaml`): correlated-signal collapses to a single unique signature
  under strict dedup and isn't viable to train/test; masquerade variants duplicate
  fabrication signatures rather than adding diversity. Masquerade attacks are
  arguably the more realistic stealthy attack vector in the original ROAD paper, so
  this is a real scope limitation, not just a data-cleaning convenience.
- **Only one defence family is tested**: PGD-adversarial training, at one epsilon
  (CICIoV2024) or two epsilon ranges (ROAD). The "AT helps here, backfires there"
  claim is about PGD-based adversarial training specifically — it has not been
  checked against other defence families (TRADES, randomized smoothing, certified
  defences), so "adversarial training in general" would be an overreach of what's
  actually been shown.
- **`threat_sizing()` (and its adaptive/black-box extensions) is a single run, not
  cross-validated** — unlike the per-class robustness CV or the 10-repeat defence
  result. CUDA/cuDNN is not forced deterministic in this codebase, so re-running it
  produces somewhat different F1s run to run (observed swings of several hundredths
  between runs at the same epsilon). Treat single-run threat-sizing numbers as
  approximate; the qualitative pattern (rejection rate holds across epsilon, rounding
  doesn't recover F1) has reproduced consistently across runs even though the exact
  decimals haven't.
- **HopSkipJump uses a deliberately reduced query budget** (`max_iter=15,
  max_eval=300` vs ART's defaults of `50`/`10000`) to stay tractable on more than a
  handful of samples, and runs on a stratified subsample (≤20 signatures/class), not
  the full test set. Its F1 is a lower bound on black-box attacker capability, not
  an upper bound — a well-resourced attacker with the full query budget would likely
  do better.
