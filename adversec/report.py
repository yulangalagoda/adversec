"""
report.py
=========
Presentation helpers: load the results artifacts written by the `adversec` CLI and
render the figures/tables for the write-up. The thin notebooks import these so they
stay load-and-plot only. matplotlib is imported lazily, so the package does not
require it unless these functions are used.

Every function takes a dataset name ("ciciov2024" or "road"), reads
results/<name>_<kind>.json, and returns a matplotlib Figure (or None if the
artifact is absent, so a notebook cell degrades gracefully).
"""
from __future__ import annotations

import json

import numpy as np

from . import config


def load_results(name: str, kind: str):
    """kind in {baseline_metrics, adversarial_results, defence_results}. None if absent."""
    path = config.RESULTS_DIR / f"{name}_{kind}.json"
    return json.loads(path.read_text()) if path.exists() else None


def _eps_order(by_epsilon: dict) -> list[str]:
    """Epsilon keys in display order: 'clean' first, then numeric ascending."""
    keys = [k for k in by_epsilon if k != "clean"]
    keys.sort(key=float)
    return (["clean"] if "clean" in by_epsilon else []) + keys


def plot_diversity_gradient(name: str):
    """Bar chart of unique attack-signature counts per class (the diversity gradient)."""
    d = load_results(name, "adversarial_results")
    if d is None:
        print(f"[{name}] adversarial_results.json absent")
        return None
    import matplotlib.pyplot as plt

    cv = d["cross_validated_per_class_pgd"]
    items = sorted(cv.items(), key=lambda kv: -kv[1]["signatures"])
    labels = [k for k, _ in items]
    sigs = [v["signatures"] for _, v in items]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labels, sigs, color="#4C78A8")
    ax.set_ylabel("unique signatures")
    ax.set_title(f"{name}: attack-signature diversity")
    ax.set_yscale("log")
    for i, s in enumerate(sigs):
        ax.text(i, s, f"{s:,}", ha="center", va="bottom", fontsize=8)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    return fig


def plot_baseline_confusion(name: str):
    """Confusion-matrix heatmaps for the RF and CNN clean baselines, with macro-F1 titles."""
    d = load_results(name, "baseline_metrics")
    if d is None:
        print(f"[{name}] baseline_metrics.json absent")
        return None
    import matplotlib.pyplot as plt

    classes = d["class_names"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, key, title in [(axes[0], "random_forest", "Random Forest"),
                           (axes[1], "cnn_1d", "1D-CNN")]:
        cm = np.array(d[key]["confusion_matrix"])
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(f"{title}  (macro-F1 {d[key]['macro_f1']:.3f})")
        ax.set_xticks(range(len(classes)))
        ax.set_yticks(range(len(classes)))
        ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(classes, fontsize=7)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        thresh = cm.max() / 2 if cm.max() else 0
        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                        color="white" if cm[i, j] > thresh else "black")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"{name}: clean-data baselines")
    fig.tight_layout()
    return fig


def plot_perclass_robustness(name: str):
    """Per-class cross-validated F1 vs epsilon (mean +/- std), classes ordered by diversity."""
    d = load_results(name, "adversarial_results")
    if d is None:
        print(f"[{name}] adversarial_results.json absent")
        return None
    import matplotlib.pyplot as plt

    cv = d["cross_validated_per_class_pgd"]
    order = sorted(cv.items(), key=lambda kv: -kv[1]["signatures"])
    eps = _eps_order(next(iter(cv.values()))["by_epsilon"])
    x = range(len(eps))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for cls, cd in order:
        by = cd["by_epsilon"]
        means = [by[e]["mean"] for e in eps]
        stds = [by[e]["std"] for e in eps]
        ax.errorbar(x, means, yerr=stds, marker="o", capsize=3,
                    label=f"{cls} ({cd['signatures']:,})")
    ax.set_xticks(list(x))
    ax.set_xticklabels(eps)
    ax.set_xlabel("PGD epsilon")
    ax.set_ylabel("F1 (mean +/- std)")
    ax.set_title(f"{name}: per-class robustness under PGD")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8, title="class (signatures)")
    fig.tight_layout()
    return fig


def plot_distance_mechanism(name: str):
    """Scatter of per-class distance-to-benign vs robustness@0.01, with the Pearson r."""
    d = load_results(name, "adversarial_results")
    if d is None:
        print(f"[{name}] adversarial_results.json absent")
        return None
    import matplotlib.pyplot as plt

    m = d["mechanism_distance_to_benign"]
    pc = m["per_class"]
    dist = [v["mean_distance"] for v in pc.values()]
    rob = [v["robustness_at_0.01"] for v in pc.values()]

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.scatter(dist, rob, color="#E45756", zorder=3)
    for cls, v in pc.items():
        ax.annotate(cls, (v["mean_distance"], v["robustness_at_0.01"]),
                    fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("mean L2 distance to nearest benign")
    ax.set_ylabel("robustness @ eps=0.01")
    # The sign is the finding: it is POSITIVE on ROAD and NEGATIVE on CICIoV2024, so the
    # title must not assert one direction for both. n is 4-5 classes either way, so this
    # is labelled exploratory rather than as an established relationship.
    r = m["pearson_r"]
    direction = "robustness rises with distance" if r > 0 else "robustness falls with distance (inverted)"
    ax.set_title(f"{name}: {direction}\n(exploratory, r={r}, n={m['n_classes']} classes)", fontsize=10)
    fig.tight_layout()
    return fig


def plot_threat_sizing(name: str):
    """F1 (raw vs integer-rounded) and per-ID envelope rejection %, across epsilon."""
    d = load_results(name, "adversarial_results")
    if d is None:
        print(f"[{name}] adversarial_results.json absent")
        return None
    import matplotlib.pyplot as plt

    rows = d["threat_sizing"]["by_epsilon"]
    eps = [r["eps"] for r in rows]
    f1_raw = [r["f1_raw_adversarial"] for r in rows]
    f1_round = [r["f1_rounded_integer"] for r in rows]
    rej = [r["pct_rejected_by_envelope"] for r in rows]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(eps, f1_raw, marker="o", label="F1 (raw adversarial)")
    ax.plot(eps, f1_round, marker="s", label="F1 (integer-rounded)")
    ax.set_xlabel("PGD epsilon")
    ax.set_ylabel("macro-F1")
    ax.set_ylim(-0.05, 1.05)
    ax2 = ax.twinx()
    ax2.plot(eps, rej, marker="^", color="#54A24B", linestyle="--", label="% rejected by envelope")
    ax2.set_ylabel("% rejected by per-ID envelope")
    ax2.set_ylim(0, 105)
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [ln.get_label() for ln in lines], fontsize=8, loc="center right")
    ax.set_title(f"{name}: threat-sizing (per-feature integer clip)")
    fig.tight_layout()
    return fig


def plot_defence(name: str):
    """
    Defended-model results.

    Handles all three schemas that can appear in <name>_defence_results.json:
      - "two_threat_model_averaged" (notebook 05, the CURRENT decisive result): grouped
        bars for baseline / static AT / Madry AT / RF across clean, transfer, white-box.
      - "robust_support_cv" and "perclass_comparison" (the older CLI `adversec defend`
        shapes, kept for the migration-gate tests).

    The notebook-05 file has no "mode" key -- it has "method" -- so dispatch on whichever
    is present rather than assuming one. (Reading d["mode"] unconditionally is what used
    to raise KeyError on every current results file.)
    """
    d = load_results(name, "defence_results")
    if d is None:
        print(f"[{name}] defence_results.json absent (run notebook 05, or: adversec defend --dataset {name})")
        return None
    import matplotlib.pyplot as plt

    mode = d.get("mode") or d.get("method")

    if mode == "two_threat_model_averaged":
        res = d["results"]
        madry = d.get("defended_cnn_madry")
        # (label, clean, transfer/attack, white-box or None)
        series = [
            ("baseline CNN", res["baseline_cnn"]["clean"], res["baseline_cnn"]["attack"], res["baseline_cnn"]["attack"]),
            ("static AT", res["defended_cnn"]["clean"], res["defended_cnn"]["transfer"], res["defended_cnn"]["white_box"]),
        ]
        if madry:
            series.append(("Madry AT", madry["clean"], madry["transfer"], madry["white_box"]))
        series.append(("Random Forest", res["random_forest"]["clean"], res["random_forest"]["attack"], None))

        labels = [s[0] for s in series]
        x = np.arange(len(series))
        w = 0.27
        fig, ax = plt.subplots(figsize=(8.5, 4.6))
        for off, idx, lab, col in [(-w, 1, "clean", "#B0B0B0"),
                                   (0.0, 2, "transfer", "#4C78A8"),
                                   (w, 3, "white-box", "#E45756")]:
            means = [(s[idx]["mean"] if s[idx] else np.nan) for s in series]
            errs = [(s[idx]["std"] if s[idx] else 0.0) for s in series]
            ax.bar(x + off, means, w, yerr=errs, capsize=3, label=lab, color=col)
        # Mark the cells that do not exist (RF has no gradients -> no white-box).
        for i, s in enumerate(series):
            if s[3] is None:
                ax.text(i + w, 0.02, "n/a", ha="center", va="bottom", fontsize=8, color="#666")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel("robust-support macro-F1")
        ax.set_ylim(0, 1.05)
        ax.set_title(f"{name}: defence under two threat models "
                     f"(PGD eps={d['attack_eps']}, mean±std over {d['n_repeats']} repeats)", fontsize=10)
        ax.legend(fontsize=8)
        fig.tight_layout()
        return fig

    if mode == "robust_support_cv":
        res = d["results"]
        models = [("base", "base"), ("def PGD", "pgd"), ("def multi", "multi"), ("RF", "rf")]
        clean = [res[f"{k}_clean"]["mean"] for _, k in models]
        adv = [res[f"{k}_adv"]["mean"] for _, k in models]
        adv_e = [res[f"{k}_adv"]["std"] for _, k in models]
        x = np.arange(len(models))
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.bar(x - 0.2, clean, 0.4, label="clean", color="#B0B0B0")
        ax.bar(x + 0.2, adv, 0.4, yerr=adv_e, capsize=3,
               label=f"PGD @ {d['attack_eps']}", color="#4C78A8")
        ax.set_xticks(x)
        ax.set_xticklabels([m for m, _ in models])
        ax.set_ylabel("robust-support macro-F1")
        ax.set_ylim(0, 1.05)
        ax.set_title(f"{name}: defended-model robustness (CV)")
        ax.legend()
        fig.tight_layout()
        return fig

    if mode == "perclass_comparison":
        res = d["results"]
        classes = d["class_names"]
        eps = _eps_order(res["baseline"])
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for mkey, style in [("baseline", "-o"), ("def_meaning", "--s"), ("def_full", ":^")]:
            # mean over classes at each epsilon (overall defence effect)
            means = [np.mean([res[mkey][e][c]["mean"] for c in classes]) for e in eps]
            ax.plot(range(len(eps)), means, style, label=mkey)
        ax.set_xticks(range(len(eps)))
        ax.set_xticklabels(eps)
        ax.set_xlabel("PGD epsilon")
        ax.set_ylabel("mean per-class F1")
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(f"{name}: adversarial training does not help (per-class mean)")
        ax.legend()
        fig.tight_layout()
        return fig

    print(f"[{name}] unknown defence mode/method {mode!r}")
    return None


def plot_attack_grid(name: str):
    """
    The notebook-07 grid: every attack x model cell as grouped bars, so the question the
    grid exists to answer -- does the defence ranking depend on which attack is used? --
    is readable at a glance.

    HopSkipJump is drawn from `blackbox_hopskipjump_direct` (a black-box attack crafted
    directly against each model), not from `grid`, and is hatched because it is measured
    on a stratified subsample rather than the full test set.
    """
    d = load_results(name, "attack_grid_results")
    if d is None:
        print(f"[{name}] attack_grid_results.json absent (run notebook 07)")
        return None
    import matplotlib.pyplot as plt

    models = ["baseline", "static", "madry", "rf"]
    pretty = {"baseline": "baseline", "static": "static AT", "madry": "Madry AT", "rf": "RandForest"}
    rows = [("clean", "clean"), ("fgsm_transfer", "FGSM transfer"), ("fgsm_whitebox", "FGSM white-box"),
            ("pgd_transfer", "PGD transfer"), ("pgd_whitebox", "PGD white-box")]
    colors = {"baseline": "#8C8C8C", "static": "#4C78A8", "madry": "#54A24B", "rf": "#E45756"}

    hsj = d.get("blackbox_hopskipjump_direct", {}).get("per_model", {})
    labels = [lab for _, lab in rows] + (["HSJ black-box"] if hsj else [])
    x = np.arange(len(labels))
    w = 0.2

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    for i, m in enumerate(models):
        means, errs, hatches = [], [], []
        for key, _ in rows:
            cell = d["grid"].get(key, {}).get(m)
            means.append(cell["mean"] if cell else np.nan)
            errs.append(cell["std"] if cell else 0.0)
            hatches.append("")
        if hsj:
            cell = hsj.get(m, {}).get("f1")
            means.append(cell["mean"] if cell else np.nan)
            errs.append(cell["std"] if cell else 0.0)
        bars = ax.bar(x + (i - 1.5) * w, means, w, yerr=errs, capsize=2,
                      label=pretty[m], color=colors[m])
        if hsj:
            bars[-1].set_hatch("//")   # subsample, not full test set

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("robust-support macro-F1")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"{name}: attack x model grid (eps={d['attack_eps']}, "
                 f"mean±std over {d['n_repeats']} repeats; hatched = subsample)", fontsize=10)
    ax.legend(fontsize=8, ncol=4)
    fig.tight_layout()
    return fig
