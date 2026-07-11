"""
cli.py
======
Command-line entry point.

    adversec list                          # show available datasets
    adversec prep --dataset ciciov2024     # run the data-prep pipeline, save artifacts
    adversec prep --dataset road

The data-prep stage is dataset-agnostic and needs no GPU. Model/experiment stages
(baseline, attack, defence) live in adversec.experiments and require torch + ART;
they are driven from the thin notebooks or added here as further subcommands.
"""
from __future__ import annotations

import argparse

import joblib
import numpy as np

from . import config
from .contract import FEATURES
from .datasets.registry import available, get_dataset
from .pipeline import (
    duplicate_train_classes,
    encode_labels,
    scale_features,
    split_train_test,
    strict_dedup,
)


def _prep(name: str) -> None:
    ds = get_dataset(name)
    meta = ds.meta()
    raw = ds.load()
    strict = strict_dedup(raw, FEATURES)
    train, test = split_train_test(strict)
    train_dup = duplicate_train_classes(train, benign_class=meta.benign_label)
    y_train, y_test, enc = encode_labels(train_dup, test)
    X_train, X_test, scaler = scale_features(train_dup, test, FEATURES)

    out = config.PROCESSED_DIR
    out.mkdir(parents=True, exist_ok=True)
    strict.to_csv(out / f"{name}_strict.csv", index=False)
    test.to_csv(out / f"{name}_test.csv", index=False)
    train_dup.to_csv(out / f"{name}_train_dup.csv", index=False)
    np.savez(out / f"{name}_stage2_arrays.npz",
             X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    joblib.dump(scaler, out / f"{name}_feature_scaler.joblib")
    joblib.dump(enc, out / f"{name}_label_encoder.joblib")
    print(f"prep done: {name}  strict={len(strict)}  train_dup={len(train_dup)}  test={len(test)}")
    print(f"  classes: {list(enc.classes_)}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="adversec")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="list available datasets")
    prep = sub.add_parser("prep", help="run the data-prep pipeline for a dataset")
    prep.add_argument("--dataset", required=True, choices=available())
    bl = sub.add_parser("baseline", help="train + evaluate the RF and CNN baselines")
    bl.add_argument("--dataset", required=True, choices=available())
    bl.add_argument("--device", default="cpu", help="cpu or cuda")
    bl.add_argument("--epochs", type=int, default=50)
    at = sub.add_parser("attack", help="cross-validated robustness + distance + threat-sizing")
    at.add_argument("--dataset", required=True, choices=available())
    at.add_argument("--device", default="cpu", help="cpu or cuda")
    at.add_argument("--epochs", type=int, default=50)
    at.add_argument("--folds", type=int, default=5)
    df = sub.add_parser("defend", help="cross-validated defended-model experiment")
    df.add_argument("--dataset", required=True, choices=available())
    df.add_argument("--device", default="cpu", help="cpu or cuda")
    df.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args(argv)

    if args.cmd == "list":
        print("datasets:", available())
    elif args.cmd == "prep":
        _prep(args.dataset)
    elif args.cmd == "baseline":
        # imported lazily so `list` / `prep` stay torch-free
        from .experiments.baseline import run_baseline
        run_baseline(args.dataset, device=args.device, cnn_epochs=args.epochs)
    elif args.cmd == "attack":
        from .experiments.adversarial import run_adversarial
        run_adversarial(args.dataset, device=args.device, cnn_epochs=args.epochs, n_splits=args.folds)
    elif args.cmd == "defend":
        from .experiments.defended import run_defended
        run_defended(args.dataset, device=args.device, cnn_epochs=args.epochs)


if __name__ == "__main__":
    main()
