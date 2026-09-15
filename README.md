# AdverSec

Adversarial robustness of ML-based intrusion detection for in-vehicle CAN-bus
networks, across two datasets — **CICIoV2024** and **ROAD** — run through one
identical, leakage-controlled pipeline. The study asks *when and how* adversarial
training actually helps: a naive, static implementation (train once against a frozen
baseline) gives an inconsistent, sometimes actively harmful defence under a white-box
attacker — but the canonical iterative (Madry-style) formulation delivers a
statistically significant robustness gain against that same worst-case attacker, on
**both** datasets. The contingency turns out to be less about data structure and more
about whether adversarial training is implemented the way the literature actually
prescribes.

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
notebooks/               01..09, step-by-step drivers; each stage saves its own results/*.json
results/                 citable JSON reports, one file per stage per dataset (tracked in git)
figures/                 write-up figures (PNG + PDF), regenerated from results/ by notebook 09
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
tracked in git, so notebooks 03–07 can run on a fresh clone without needing the
raw data at all. `08_accuracy_trap` is the exception: it needs `datasets/raw/`,
because the duplicated rows it measures are exactly what de-duplication discards.

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

`notebooks/01..09` are step-by-step drivers over the same package code as the CLI,
meant for interactive/report use. Each stage saves a citable artifact, so any
notebook can be run on its own once its inputs exist on disk:

| Notebook | Reads | Saves |
|---|---|---|
| `01_cleaning` | raw data | `<name>_{strict,train,test}.csv`, `results/<name>_dedup_audit.json` |
| `02_preprocessing` | 01's `train.csv`/`test.csv` | `<name>_stage2_arrays.npz`, `<name>_train_dup.csv`, scaler/encoder, `<name>_prep_summary.json` |
| `03_baselines` | 02's arrays + encoder | `results/<name>_baseline_metrics.json` |
| `04_adversarial_attacks` | 02's arrays + encoder, 01's `strict.csv` | `results/<name>_adversarial_results.json` (per-class PGD CV + distance-to-benign) |
| `05_defence` | 02's arrays + encoder | `results/<name>_defence_results.json` (`two_threat_model_averaged` schema — see below) |
| `06_threat_sizing` | 02's arrays/scaler/encoder, 01/02's `train_dup.csv`/`test.csv` | `results/<name>_adversarial_results.json` (adds `threat_sizing`, `adaptive_envelope_aware_attack`, `blackbox_hopskipjump_attack`, `envelope_clean_control`; merge-safe with 04, either order) |
| `07_attack_grid` | 02's arrays + encoder/scaler, 01/02's `train_dup.csv` | `results/<name>_attack_grid_results.json` (FGSM/PGD/HopSkipJump × baseline/static/Madry/RF × transfer/white-box) |
| `08_accuracy_trap` | **raw data** + 01's dedup | `results/<name>_accuracy_trap.json` (leaky row-level CV vs honest signature-level CV) |
| `09_figures` | `results/*.json` only | `figures/<name>_<plot>.png` + `.pdf` — every figure for the write-up |

Run `01` then `02` once per dataset; `03`–`07` only depend on `02`'s output, not
on each other, so any one of them can be re-run standalone. `08` is the only
notebook after `01` that reads `datasets/raw/`.

**`06`'s clean-frame control.** `envelope_clean_control` measures what the per-ID
envelope does to *unperturbed* test frames. Without it, an adversarial rejection
rate is uninterpretable: attack frames are unusual traffic by construction, so many
sit outside the benign envelope before any perturbation is applied. The same section
scores the envelope as a standalone, no-ML detector.

**`07`'s black-box column reports two F1s.** `f1` is measured on continuous frames;
`f1_rounded` re-measures after rounding to legal integers. HopSkipJump minimises L2
and can return perturbations smaller than one byte value — which cannot be injected on
a real bus — so **`f1_rounded` is the number to quote** as the physically real threat.

⚠ **`06` and `07` report HopSkipJump on different metrics — don't mix them in one table.**
`06` uses plain macro-F1 over *all* classes (CICIoV2024: 0.069); `07` uses robust-support
macro-F1, i.e. only classes with ≥2 test frames (CICIoV2024 baseline: 0.053). They also
draw independent subsamples. The numbers differ for those reasons, not because either is
wrong — quote `07`'s for anything compared against the defence grid, and `06`'s only
alongside its own PGD-on-the-same-subsample reference.

**`08` measures the accuracy trap directly.** Every other notebook infers it from the
duplication rate; `08` runs the same models under leaky row-level CV and honest
signature-level CV and reports the difference as `inflation_macro_f1`.

**`09` needs neither GPU nor raw data.** It reads `results/*.json` and renders all nine
figures for both datasets into `figures/` as PNG (200 dpi) and PDF (vector, for the
write-up), so figures can be regenerated on a laptop without re-running any experiment.

**`05_defence`'s methodology differs from `adversec defend`.** The CLI/`experiments/defended.py`
path runs the per-dataset-shaped comparison from `configs/*.yaml` (robust-support
CV for CICIoV2024, per-class comparison for ROAD — the older design). The notebook
instead treats both datasets identically, and compares **two forms of adversarial
training** (`adversec/experiments/defense.py`):

- **Static** (`adversarial_train_cnn`): PGD examples crafted *once* from a frozen,
  undefended baseline, then appended to the clean training set. Cheap, but — as
  Kurakin et al. warn — the attack the model trains against never adapts to the model
  actually being defended.
- **Madry-style / iterative** (`madry_adversarial_train_cnn`): PGD examples crafted
  fresh, every batch, against the model's *current* weights — the canonical
  inner-maximisation/outer-minimisation formulation (Madry et al., 2018).

Both are evaluated under clean / transfer-attack / white-box-attack robust-support
macro-F1 — for the baseline CNN, both defended CNNs, and the Random Forest — under PGD
eps=0.10, averaged (mean ± std) over `N_REPEATS=10` seeded runs, with paired t-tests
(matched by seed, so `defended[r] - baseline[r]` is a real paired comparison, not two
independent samples) confirming which differences are real rather than noise. This is
the current, decisive result; the CLI path's `robust_support_cv`/`perclass_comparison`
schemas predate it and are kept for the migration-gate tests, not as the primary
defence claim.

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

*All numbers below are generated from `results/*.json` and match the committed notebook
outputs. Robust-support macro-F1 unless stated otherwise.*

- **Static adversarial training is inconsistent and sometimes actively harmful — the
  canonical iterative (Madry-style) form fixes this, on both datasets.** Trained the
  cheap/static way (a fixed adversarial set crafted once from a frozen baseline), AT
  helps significantly under transfer attack on both datasets (CICIoV2024: +0.442,
  p<0.0001; ROAD: +0.499, p<0.0001), but under white-box it has **no significant effect
  on CICIoV2024** (−0.020, p=0.255 — a genuine null result) and **significantly backfires
  on ROAD** (−0.098, p=0.015: the defended model ends up *less* robust than the
  undefended baseline). Retrained the proper Madry-style way (PGD crafted fresh against
  the model's current weights, every batch), white-box robustness improves
  **significantly over the undefended baseline on both datasets** (CICIoV2024: +0.104,
  p=0.003; ROAD: +0.265, p<0.0001) and **significantly over the static approach on both**
  (CICIoV2024: +0.122, p=0.0007; ROAD: +0.318, p<0.0001). Two costs come with it: Madry
  AT's *transfer* robustness is well below static AT's (CICIoV2024 0.471 vs 0.728; ROAD
  0.513 vs 0.829), and on ROAD it costs a great deal of **clean** accuracy (0.766 vs the
  baseline's 0.997 — a 23-point drop), whereas on CICIoV2024 clean performance is
  untouched (0.791 vs 0.793).
- **The defence ranking does not depend on which attack you use** (notebook 07). Madry AT
  beats static AT white-box under PGD (CICIoV2024 +0.119, p=0.0004; ROAD +0.356,
  p=0.0009) *and* under FGSM (CICIoV2024 +0.207, p=0.037; ROAD +0.337, p=0.0015), while
  PGD-vs-FGSM against the Madry model is not significantly different on either dataset
  (p=0.105, p=0.169). The static-vs-iterative finding is therefore a property of the
  defence, not an artifact of the attack chosen to test it.
- **Under attack, an adversarially trained CNN beats the Random Forest — on clean traffic
  it does not.** Across the full grid the best AT-CNN wins **9 of 10 attack conditions**
  (the exception is HopSkipJump on CICIoV2024), by margins up to +0.62 macro-F1; the RF
  leads only on clean data (CICIoV2024 0.859 vs 0.848; ROAD 1.000 vs 0.976).
- **The accuracy trap, measured rather than asserted** (notebook 08). Running the *same
  models* under leaky row-level CV and honest signature-level CV: on CICIoV2024 a leaky
  evaluation reports a **perfect macro-F1 of 1.000 for both RF and CNN**, while the
  honest measurement is 0.676 (RF) and 0.764 (CNN) — an invented **+0.324 / +0.236**. On
  ROAD the same contrast produces **essentially nothing** (−0.000 RF, +0.004 CNN). The
  trap tracks duplication exactly: CICIoV2024 is 99.75% duplicate rows (1,408,219 → 3,588
  unique signatures), ROAD only 39.8% (66,252 → 39,858). This is the single cleanest
  demonstration in the study that leakage control is a prerequisite, not a nicety.
- **The per-ID envelope is a *detector*, not a perturbation filter — and it is evaded by
  an attacker who knows it's there.** The clean-frame control (notebook 06) is what makes
  the adversarial rejection rates readable: applied to *unperturbed* frames the envelope
  already rejects **100% of every attack class on CICIoV2024** and 100% of `fuzzing` and
  `max-speedometer` on ROAD, while missing both `reverse-light` classes entirely (they
  are masquerade-style attacks on legitimate IDs with in-range bytes). So the headline
  "the envelope rejects ~all adversarial frames" is mostly detecting *attacks*, not
  *perturbations*. As a standalone, no-ML detector it scores precision 0.281 / recall
  1.000 / F1 0.439 on CICIoV2024 (9/9 attacks caught, 23/709 benign false alarms) and
  precision 0.976 / recall 0.597 / F1 0.741 on ROAD. Against naive PGD it rejects 100%
  (CICIoV2024) and 91.1–91.2% (ROAD) at a measured benign false-positive cost of 3.24%
  and 1.32%. But an attacker who keeps a real arbitration ID and clips payload bytes into
  that ID's own observed range drops rejection to **2.9%** and **1.4%** — i.e. converts
  frames that were 100% rejected into frames that pass, while still degrading the model
  (F1 0.241 and 0.317 at eps=0.10).
- **Black-box attacks are severe, and remain so after the physical-realisability check.**
  HopSkipJump (reduced query budget, ≤20 signatures/class) is more damaging than white-box
  PGD at eps=0.10 on both datasets (CICIoV2024 F1 0.069 vs 0.171; ROAD 0.068 vs 0.226)
  without ever touching a gradient. Because it minimises L2 it can return frames perturbed
  by *less than one integer unit* — against the ROAD Random Forest, mean L2 = 0.003 — so
  the grid reports both a raw and an integer-rounded F1. Rounding recovers some of the
  Random Forest's score (ROAD 0.067 → 0.113; CICIoV2024 0.320 → 0.402) and leaves the CNNs
  unchanged (their perturbations are far above integer granularity), so **the black-box
  threat is real but was overstated for the tree model** by the continuous-space number.
  `f1_rounded` is the figure to quote.
- **The distance-to-benign robustness mechanism inverts on CICIoV2024** (pearson_r =
  −0.531, n=5, vs +0.728 on ROAD, n=4 — classes *closer* to benign are the *more* robust
  ones, the opposite pattern to ROAD). An ablation removing the light-duplication
  convergence crutch shows the inversion survives (r = −0.315) — same sign, reduced
  magnitude — so it is a real, if amplified-by-duplication, data-structure effect rather
  than an artifact of the padding. Treat it as exploratory: at n=4–5 classes neither
  correlation approaches significance.

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
- **Both defence variants tested are still PGD-based adversarial training** (static
  and Madry-style), at one epsilon (CICIoV2024) or two epsilon ranges (ROAD). Neither
  has been checked against other defence families (TRADES, randomized smoothing,
  certified defences), so "adversarial training in general" would still be an
  overreach — what's been shown is specifically that *how* PGD-based AT is
  implemented (static vs iterative) changes the result substantially, not that this
  generalises to every possible defence.
- *(resolved — pending a re-run of notebook 05's significance cells)* **Madry AT's
  transfer-attack robustness** now has the same paired test as its white-box robustness
  (`madry_transfer_vs_undefended`), computed from the saved runs with no retraining. On
  the committed runs it comes out significant on both datasets (CICIoV2024 +0.185,
  p=0.0009; ROAD +0.227, p=0.0007), but that key only appears in
  `<name>_defence_results.json` once those cells are re-run.
- **The envelope-aware adaptive attack is not checked for goal preservation.** It clips
  payload bytes into the target ID's benign range, which is what lets it pass the
  envelope — but a `max-speedometer` frame whose bytes have been clipped into the normal
  range may no longer *set the speedometer to max*. What is demonstrated is that
  protocol-valid frames exist which evade both the filter and the detector; whether they
  still achieve the attacker's physical objective on a running vehicle is untested and
  out of scope.
- **The envelope's CICIoV2024 detector precision rests on 9 attack frames.** The
  clean-frame control reports precision 0.281 / recall 1.000 there, but the test split
  contains only 9 attack frames in total against 709 benign, so precision in particular
  is a very thin measurement. ROAD's version (3,734 attack frames) is the one to trust.
- **The accuracy-trap contrast caps the leaky arm at 50,000 rows.** Row-level CV runs on
  a stratified subsample of the raw duplicated data (CICIoV2024's raw form is 1.4M rows),
  which preserves the duplication structure — 99.26% after subsampling vs 99.75% raw —
  but is not the full table. The honest arm uses every unique signature. CICIoV2024 also
  uses 2 folds (`spoofing-GAS` has only 2 signatures), so its honest CNN/RF ordering
  there (CNN 0.764 > RF 0.676) differs from the single held-out split in notebook 03
  (RF 0.776 > CNN 0.661); both are thin, and neither should be read as settling the
  simple-vs-deep question on that dataset.
- **`threat_sizing()` (and its adaptive/black-box extensions) is a single run, not
  cross-validated** — unlike the per-class robustness CV or the 10-repeat defence
  result. CUDA/cuDNN is not forced deterministic in this codebase, so re-running it
  produces somewhat different F1s run to run (observed swings of several hundredths
  between runs at the same epsilon). Treat single-run threat-sizing numbers as
  approximate; the qualitative pattern (rejection rate holds across epsilon, rounding
  doesn't recover F1, adaptive/black-box attacks evade far more than naive ones) has
  reproduced consistently across runs even though the exact decimals haven't.
- **HopSkipJump uses a deliberately reduced query budget** (`max_iter=15,
  max_eval=300` vs ART's defaults of `50`/`10000`) to stay tractable on more than a
  handful of samples, and runs on a stratified subsample (≤20 signatures/class per
  dataset — 29 total for CICIoV2024, since most of its classes have only 1-4 test
  signatures to sample from; 100 for ROAD), not the full test set. Its F1 is a lower
  bound on black-box attacker capability, not an upper bound — a well-resourced
  attacker with the full query budget would likely do at least as well. On CICIoV2024
  the robust-support subset of that subsample is roughly 26 frames, so quote those cells
  with their n; ROAD's 100 are better founded.
