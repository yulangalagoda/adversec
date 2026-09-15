# AdverSec — Project Summary

## 1. The Flow

This is the exact sequence every dataset goes through, start to finish. Both datasets
(CICIoV2024 and ROAD) go through the *identical* sequence below — only the very first
step differs (how the raw files are read), everything after that is the same code
treating both the same way.

### Step 1 — Load into one common shape
Each raw dataset (CICIoV2024's per-class CSVs in decimal form; ROAD's raw CAN-bus
capture logs plus an attack-metadata file describing which frames are injected) is
converted into one common table: an arbitration ID, 8 payload bytes, and a true class
label (benign or a named attack). Every row is checked against a strict contract
(byte values 0–255, ID within the dataset's legal range, no missing values, only known
class labels) so nothing malformed slips downstream.

### Step 2 — Measure and remove duplication
Raw CAN traffic is heavily repetitive — the same frame recurs thousands of times. The
duplication rate is measured first (this matters: CICIoV2024 is ~99.75% duplicated,
ROAD ~40%), then the data is reduced to one row per unique (features + class)
combination — the genuinely distinct signatures. This step exists because training or
testing on the raw, duplicated data massively inflates apparent accuracy without
teaching the model anything new.

### Step 3 — Split into train and test
The unique signatures (not the raw duplicated rows) are split into train and test sets,
stratified by class and seeded for reproducibility. This happens *before* any
padding/augmentation, so no synthetic copy of a training signature can ever leak into
the test set.

### Step 4 — Pad tiny classes for trainability (train only)
Some attack classes have only a handful of genuinely distinct signatures. To give a
neural network enough signal to learn from without literally inventing new data, the
smallest classes are padded up to a target row count by repeating their real signatures
extra times. This never touches the test set, never touches benign traffic, and adds
volume — not diversity.

### Step 5 — Encode and scale
Class labels are converted to integers. Each of the 9 features (ID + 8 payload bytes)
is scaled to a 0–1 range, fitted on the training data only. This scaled space is also
where every adversarial perturbation later is measured and bounded.

### Step 6 — Train clean baselines
Two models are trained on the processed data: a Random Forest and a small 1D
convolutional neural network. Both are evaluated on the untouched test set using
accuracy, macro-averaged F1, per-class precision/recall, and a confusion matrix —
accuracy alone is treated as untrustworthy given how imbalanced the classes are.

### Step 7 — Attack the models
The neural network is wrapped so its gradients can be computed, then adversarial
examples are crafted two ways:
- **Gradient-based** (FGSM — one perturbation step; PGD — many small steps, stronger)
  across a range of perturbation sizes (from barely-perceptible to large).
- **Gradient-free / black-box** (HopSkipJump) — an attack that only ever asks the model
  "what's your prediction here?" and never touches its internals, to check whether
  findings depend on the attacker having gradient access.

The same crafted adversarial frames are fed to *both* the neural network and the
Random Forest, so the Random Forest's robustness is being measured under a transfer
attack (perturbations it never influenced).

### Step 8 — Cross-validate per-class robustness
Rather than trusting one train/test split, the whole train → attack → evaluate cycle
is repeated across 5 folds, per class, to see how each individual attack type holds up
under increasing perturbation — and specifically whether classes with more distinct
signatures are inherently more robust (they are not, in general).

### Step 9 — Explain *why* some classes are more robust than others
For every attack class, its distance (in the same scaled feature space) to the nearest
real benign signature is measured, and compared against how robust that class turned
out to be. On one dataset, classes far from benign traffic are the robust ones; on the
other, that relationship runs backwards. That reversal is then tested for robustness
itself, by re-running the same measurement with the class-padding step (Step 4)
switched off, to check the reversal isn't just an artefact of the padding.

### Step 10 — Train a defended model, two different ways, and test both under two threat models
A second neural network is trained the same way as the baseline, then made "defended" —
tried two different ways:
- **Static**: additionally trained on a fixed batch of adversarial examples crafted
  *once*, from the *undefended* baseline, before training starts.
- **Iterative (Madry-style)**: adversarial examples are crafted fresh, every single
  training batch, against whatever the model's weights happen to be *at that moment* —
  a moving target, rather than a fixed one.

Both defended models are then measured under two different assumptions about the
attacker:
- **Transfer**: the attacker only ever sees the undefended baseline model.
- **White-box**: the attacker has full access to the defended model itself — the worst
  case.

This whole training-and-testing cycle is repeated many times over different random
seeds, and a paired statistical test is used to check whether each defended model's
apparent improvement (or harm) is real, rather than random noise from one lucky or
unlucky training run — including a direct statistical comparison between the two
defence recipes themselves, not just each against the undefended baseline.

### Step 11 — Test whether the attacks are physically real
An adversarial example only matters if it corresponds to a frame a real attacker could
actually send. So every crafted frame is rounded to the nearest legal integer value,
then checked against the range of byte values genuinely observed, historically, for
that specific arbitration ID in real benign traffic — a cheap, non-learned validity
check. This is tested three ways:
- A naive attacker who ignores this check entirely (how much does rounding alone undo
  the attack, and how often does the check catch it? — and, just as importantly, how
  often does the same check wrongly flag genuinely legitimate traffic?).
- An adaptive attacker who already knows about the check and deliberately keeps a real,
  legitimate ID and stays inside that ID's normal byte range.
- The black-box attacker from Step 7, to see whether this same defence still works
  against an attacker who never used gradients at all.

Crucially, the same check is also run on *clean, untouched* traffic first. Attack frames
are unusual by their nature, so a lot of them already fail the check before anyone
perturbs anything — without that control, "the check rejects almost every adversarial
frame" would be measuring the wrong thing. Running it both ways separates "the
perturbation made this frame illegal" from "this was an attack and was already illegal",
and as a by-product scores the cheap check on its own, as a detector with no machine
learning in it at all.

### Step 12 — Widen the attack, so the conclusion doesn't rest on one method
The defence comparison above uses one attack (the strong iterative gradient one). That
invites an obvious objection: maybe the answer is an artefact of that particular attack.
So the whole grid is re-run — one-step gradient, iterative gradient, and gradient-free
black-box — against every model, under both attacker assumptions, to check whether the
ranking of the defences actually depends on which attack you pick. The black-box attack
is additionally re-scored after rounding every crafted frame to legal whole numbers,
because an attack that only exists at fractional precision is not an attack anyone can
send down a real wire.

### Step 13 — Measure the accuracy trap instead of asserting it
Every step above takes it on faith that evaluating on repeated traffic inflates scores.
This step proves it, by scoring the same models two ways: once the naive way, letting
copies of the same frame fall on both sides of the train/test divide, and once properly,
where no copy of a frame can ever cross that line. The difference between the two numbers
is the inflation a careless evaluation invents, measured rather than assumed.

---

## 2. File Structure

### Top level
| Path | What it is |
|---|---|
| `adversec/` | The installable Python package — all the actual logic lives here |
| `configs/` | One YAML file per dataset — paths, class names, dataset-specific knobs |
| `datasets/raw/` | The original, unmodified data (gitignored — sourced separately) |
| `datasets/processed/` | Generated arrays/CSVs/scalers each stage produces (small, tracked in git) |
| `notebooks/` | Nine step-by-step notebooks, one per pipeline stage, for interactive use |
| `figures/` | Every figure for the write-up (PNG + PDF), generated from `results/` by notebook 09 |
| `results/` | The citable JSON output of every stage — the actual numbers cited anywhere |
| `tests/` | Scripts that re-run the pipeline and check the numbers still match |
| `README.md` | Project overview, reproduction instructions, findings, limitations |
| `pyproject.toml`, `requirements.txt` | How to install it and with which exact versions |

### Inside `adversec/` (the package)

| File | Contains | Connects to |
|---|---|---|
| `contract.py` | The definition of "a valid CAN table" (which columns, what type, legal ranges) and a `validate()` function that enforces it | Called by every dataset adapter before handing data downstream; imported by almost every other file for its column-name constants |
| `config.py` | Shared, dataset-*agnostic* settings: file paths, random seed, split/duplication/attack hyperparameters, plus a loader for each dataset's own YAML | Read by nearly everything: the CLI, the pipeline stages, the experiments, the dataset adapters |
| `datasets/base.py` | The one interface every dataset adapter must implement (`load()`, `meta()`) | Implemented by `ciciov.py` and `road.py` |
| `datasets/ciciov.py` | The *only* code that understands CICIoV2024's raw CSV format | Reads its settings from `configs/ciciov2024.yaml` via `config.py`; validates output via `contract.py` |
| `datasets/road.py` | The *only* code that understands ROAD's raw candump-log format and its attack metadata | Reads its settings from `configs/road.yaml`; validates via `contract.py` |
| `datasets/registry.py` | Looks up a dataset adapter by name (`"ciciov2024"` or `"road"`) | Used by `cli.py` and by the experiment scripts so they never import a dataset adapter directly |
| `pipeline/dedup.py` | Removes duplicate rows; audits how duplicated the raw data is | Used by `cli.py`'s `prep` command, notebook 01, and reused inside `experiments/crossval.py` |
| `pipeline/split.py` | Signature-level train/test split | Same callers as `dedup.py`, plus notebook 01 |
| `pipeline/augment.py` | Pads small train classes up to a target row count | Used by `cli.py` prep, notebook 02, and every cross-validation function in `experiments/crossval.py` |
| `pipeline/encode.py` | Label encoding + per-feature scaling | Same callers as `augment.py` |
| `models/rf.py` | Builds the Random Forest | Used by `experiments/baseline.py`, `crossval.py` |
| `models/cnn.py` | Defines the small 1D-CNN and its training loop | Used everywhere a CNN needs training: `baseline.py`, `adversarial.py`, `defense.py`, `crossval.py`, all notebooks |
| `experiments/attack.py` | Wraps a trained CNN so gradients are accessible; generates FGSM, PGD, and HopSkipJump adversarial examples | Used by `adversarial.py`, `defense.py`, `crossval.py`, notebooks 04/05/06 |
| `experiments/defense.py` | Builds an adversarially-augmented training set and trains the "defended" CNN | Used by `crossval.py` and notebook 05 |
| `experiments/crossval.py` | Every cross-validation scheme: leaky row-level vs honest signature-level, per-class robustness, defended-model comparison | Used by `adversarial.py`, `defended.py`, notebooks 04/08 (the leaky-vs-honest pair is what notebook 08 runs) |
| `experiments/realism.py` | Physical-plausibility checks: integer rounding, learning the per-ID "envelope" of legal byte values, checking/clipping frames against it, and scoring the envelope on clean frames as a standalone detector (the control for every rejection rate) | Used by `adversarial.py`'s threat-sizing function and notebooks 06/07 |
| `experiments/baseline.py` | Orchestrates Step 6 (train + evaluate both clean models) and saves `results/<name>_baseline_metrics.json` | Called by `cli.py`'s `baseline` command; mirrored (not called) by notebook 03 |
| `experiments/adversarial.py` | Orchestrates Steps 7–9 and 11 (attacks, cross-validated robustness, distance-to-benign mechanism, threat-sizing) and saves `results/<name>_adversarial_results.json` | Called by `cli.py`'s `attack` command; mirrored (not called) by notebooks 04/06 |
| `experiments/defended.py` | Orchestrates the *older* defence comparison shape (per `configs/*.yaml`) and saves `results/<name>_defence_results.json` | Called by `cli.py`'s `defend` command — superseded for reporting purposes by notebook 05's own newer methodology |
| `evaluation.py` | Shared metric functions (`evaluate_model`, `robust_support_f1`) used identically by every model so comparisons are fair | Used by `baseline.py` and notebooks 03/04 |
| `report.py` | Loads `results/*.json` and draws the matplotlib figures for the write-up — nine plots: signature diversity, baseline confusion, per-class robustness, distance mechanism, threat sizing, the two-threat-model defence, the attack grid, the accuracy trap, and the envelope control | Standalone — reads results, produces plots, nothing else depends on it |
| `cli.py` | The `adversec` command-line entry point (`prep`, `baseline`, `attack`, `defend`) | The "glue" — calls into `datasets/`, `pipeline/`, and `experiments/` based on which subcommand and `--dataset` flag is given |

### `configs/ciciov2024.yaml`, `configs/road.yaml`
Per-dataset settings only: raw file paths, class names, the legal arbitration-ID range,
whether the CNN needs class-weighting, and how the defence experiment is shaped for
that dataset. Read by `config.py`'s loader, consumed by the matching dataset adapter
and by `experiments/defended.py`.

### `notebooks/01`–`09`
Interactive, step-by-step versions of the same pipeline. `01` and `02` produce the
processed arrays everything else needs; `03`–`07` each read those processed arrays and
write their own `results/*.json` — they don't depend on each other, only on `02`'s
output, so any one of them can be run on its own. `04` and `06` both write into the
*same* results file (different sections), safely, in either order. `07` runs the wide
attack grid (Step 12) into its own file. `09` needs no models at all — it reads the saved
results and renders every figure for the write-up into `figures/`, so plots can be
regenerated on any machine in seconds. `08` (Step 13) is the one exception to the
"only needs `02`'s output" rule: it measures the accuracy trap, so it has to go back to
the original, still-duplicated raw data that de-duplication threw away.

### `results/*.json`
The actual output of the study — one JSON file per pipeline stage per dataset. This is
what `report.py` reads to make figures, what the tests check against, and what a later
notebook (06 into 04's file) merges into.

### `tests/*.py`
Re-run the pipeline fresh and check every number matches what's already saved in
`results/` and `datasets/processed/` — a migration gate, not a general test suite for
correctness in the abstract sense.

---

## 3. Summary — In Plain English

This project asks one question: **does training a model to resist adversarial
attacks actually help it, in a real intrusion-detection setting on a car's internal
network — and does the answer depend on how that training is actually done?**

To find out, two publicly available CAN-bus attack datasets are run through the exact
same code, so any difference in the result is a difference in the *data*, not in how
carefully each one was analysed. Both datasets have their raw traffic cleaned up (a lot
of it turns out to be exact repeats, which inflates how good a model looks unless it's
removed), split fairly into train/test, and used to train two kinds of detector: a
Random Forest and a small neural network.

Both detectors are then attacked with small, deliberately crafted changes to the traffic
— changes designed to fool the model into misclassifying an attack as normal traffic (or
vice versa) — first assuming the attacker has full knowledge of the model (the worst
case), and separately assuming the attacker only knows a *different* copy of the model
(a more realistic case) and even assuming the attacker has no internal access to the
model at all and can only watch what it predicts.

One dataset's attacks are numerous and varied; the other's are scarce, with some attack
types having only a handful of genuine real-world examples to learn from at all. Training
the model against adversarial examples ahead of time consistently helps a lot against an
attacker who doesn't have access to the exact deployed model, on both datasets — but the
picture under an attacker who *does* have that access (the worst case) turns out to
depend entirely on **how** the training was done, not on which dataset it was: trained
the cheap, common way (against a fixed batch of adversarial examples crafted once,
upfront), the defence gives no real benefit on the data-scarce dataset and actively makes
things worse on the data-rich one. Trained the more expensive, textbook-correct way
(regenerating fresh adversarial examples every single training batch, against whatever
the model currently is), the defence delivers a real, statistically confirmed
improvement against that worst-case attacker — **on both datasets.** So this particular
defence isn't unreliable because of what data it's given; it's unreliable when it's
implemented the cheap way, and that's fixable.

Separately, the project tests a much simpler, non-machine-learning defence: since a
car's internal network has fairly predictable, repetitive legitimate traffic, a cheap
check can flag any frame whose values fall outside what's normally ever been seen for
that specific message type. Running that check on untouched traffic first turns out to
matter a great deal. On the repetitive dataset it flags *every single attack frame*
before anyone perturbs anything, which means the headline "it catches almost all the
crafted attacks" was mostly measuring that these were attacks, not that they had been
tampered with. Read properly, the cheap check is a perfectly decent attack detector in
its own right — on one dataset it catches every attack at a small, measured cost in
false alarms on genuine traffic; on the other it catches two attack types completely and
is blind to two others, because those two impersonate a legitimate message using
perfectly normal-looking values.

It is also, on its own, not security. An attacker who already knows the check is there,
keeps a legitimate ID and stays inside that ID's normal byte range goes from being caught
every time to passing almost every time — while still degrading the detector. And an
attacker who never had access to the model's internals at all, only its predictions, can
still do serious damage — sometimes more than a "stronger", fully-informed attacker.
That last result needed one extra check of its own: because such an attacker searches for
the *smallest* effective change, it can land on a change smaller than the smallest step a
real CAN frame can actually take. Re-measuring after snapping every crafted frame to legal
whole numbers shows the threat is genuine but was overstated for the simpler model.

Finally, the project stops asserting its own founding premise and measures it. Scoring the
same models twice — once letting copies of the same frame fall on both sides of the
train/test divide, once forbidding it — the repetitive dataset reports a *flawless* score
the naive way and a mediocre one the honest way, while the varied dataset barely moves.
The inflation isn't a modelling subtlety; it is the entire difference between a headline
result and a real one.

Altogether, the project isn't just "here's a model, here's its accuracy" — it's a
structured investigation into *when* a particular defence works, *why* it works or
fails, and how much of what looks like a strong result is actually real once it's
checked properly rather than assumed.
