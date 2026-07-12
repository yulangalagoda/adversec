"""
Baseline verification (step-4 migration gate).

Re-runs the RF + CNN baseline through the new package and checks it against the
committed metrics. Requires torch + prepared arrays, so it skips in the Windows
working copy and runs on the lab machine.

Random Forest is fully deterministic (fixed seed) -> exact macro-F1 match.
The CNN depends on the torch/CUDA numerics of the machine that produced the
committed run, so it is checked within a tolerance; on the lab machine (same
pinned CUDA torch) it should match closely.

Run:  python tests/test_baseline.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PROC = ROOT / "datasets" / "processed"
RES = ROOT / "results"


def test_road_baseline():
    if not (PROC / "road_stage2_arrays.npz").exists():
        print("SKIP road baseline: prepared arrays absent (run on the lab machine)")
        return
    if not (RES / "road_baseline_metrics.json").exists():
        print("SKIP road baseline: committed metrics absent")
        return

    try:
        from adversec.experiments.baseline import run_baseline
    except ModuleNotFoundError as e:
        print(f"SKIP road baseline: {e} (torch/ART not installed here)")
        return

    committed = json.load(open(RES / "road_baseline_metrics.json"))
    report = run_baseline("road", device="cpu", cnn_epochs=50, save=False)

    # Random Forest: deterministic -> exact.
    assert report["random_forest"]["macro_f1"] == committed["random_forest"]["macro_f1"], (
        report["random_forest"]["macro_f1"], committed["random_forest"]["macro_f1"])
    assert report["random_forest"]["confusion_matrix"] == committed["random_forest"]["confusion_matrix"]

    # CNN: within tolerance (device/torch-version sensitive).
    d = abs(report["cnn_1d"]["macro_f1"] - committed["cnn_1d"]["macro_f1"])
    assert d < 0.02, f"CNN macro-F1 drifted by {d:.4f}"

    assert report["test_size"] == committed["test_size"]
    assert report["class_names"] == committed["class_names"]
    print(f"road baseline ok: RF macro-F1 {report['random_forest']['macro_f1']:.4f} (exact), "
          f"CNN {report['cnn_1d']['macro_f1']:.4f} (|d|={d:.4f})")


if __name__ == "__main__":
    test_road_baseline()
    print("\nSTEP-4 BASELINE GATE PASSED (or skipped if torch/data absent)")
