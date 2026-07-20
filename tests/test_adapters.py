"""
Adapter verification.

Each dataset adapter must emit a valid canonical table. Where the raw data and
the legacy src/ pipeline are BOTH present, the canonical table must also match
the legacy loader's output on the canonical columns -- this is the migration
gate for step 2 (loaders). Tests skip gracefully when data or legacy code is
absent, so the file runs on any machine.

Run:  python tests/test_adapters.py     (or)     python -m pytest tests/
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from adversec.contract import CANONICAL_COLUMNS, validate          # noqa: E402
from adversec.datasets.registry import available, get_dataset      # noqa: E402


def _has_ciciov_raw() -> bool:
    return (ROOT / "datasets" / "raw" / "ciciov2024_decimal" / "decimal_benign.csv").exists()


def _has_road_raw() -> bool:
    return (ROOT / "datasets" / "raw" / "road" / "attacks" / "capture_metadata.json").exists()


def test_registry_lists_both():
    assert set(available()) == {"ciciov2024", "road"}
    print("registry: ok ->", available())


def _cross_check_against_legacy(df_new, legacy_module, loader_name):
    """Compare the adapter's canonical table to the legacy loader's, on canonical columns."""
    legacy_src = ROOT / "src"
    if not (legacy_src / f"{legacy_module}.py").exists():
        print(f"  (legacy src/{legacy_module}.py absent -> structural check only)")
        return
    sys.path.insert(0, str(legacy_src))
    mod = importlib.import_module(legacy_module)
    df_old = getattr(mod, loader_name)()
    a = df_new.reset_index(drop=True)
    b = df_old[CANONICAL_COLUMNS].reset_index(drop=True)
    assert a.equals(b), "canonical table diverged from the legacy loader"
    print(f"  matches legacy {legacy_module}.{loader_name}()  ({len(a):,} rows)")


def test_ciciov():
    if not _has_ciciov_raw():
        print("SKIP ciciov: raw ciciov2024_decimal CSVs not present")
        return
    ds = get_dataset("ciciov2024")
    df = ds.load()
    validate(df, ds.meta())
    assert list(df.columns) == CANONICAL_COLUMNS
    print(f"ciciov: valid canonical table  ({len(df):,} rows, classes={ds.meta().class_names})")
    _cross_check_against_legacy(df, "cleaning", "load_raw_data")


def test_road():
    if not _has_road_raw():
        print("SKIP road: raw captures not present in this working copy (run on the lab machine)")
        return
    ds = get_dataset("road")
    df = ds.load()
    validate(df, ds.meta())
    assert list(df.columns) == CANONICAL_COLUMNS
    print(f"road: valid canonical table  ({len(df):,} rows, classes={ds.meta().class_names})")
    _cross_check_against_legacy(df, "road_cleaning", "load_road_dataset")


if __name__ == "__main__":
    test_registry_lists_both()
    test_ciciov()
    test_road()
    print("\nDONE")
