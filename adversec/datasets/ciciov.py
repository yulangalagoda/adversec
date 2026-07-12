"""
ciciov.py
=========
Adapter for CICIoV2024 (decimal representation).

Six per-class CSVs, each with columns ID, DATA_0..DATA_7 plus the dataset's own
label columns (label, category, specific_class). The label columns are redundant
with the canonical true_class and are dropped here.
"""
from __future__ import annotations

import pandas as pd

from .. import config
from ..contract import CANONICAL_COLUMNS, LABEL_COLUMN, DatasetMeta, validate
from .base import CANDataset


class CICIoVDataset(CANDataset):
    def __init__(self, cfg: dict | None = None):
        self._cfg = cfg if cfg is not None else config.load_dataset_config("ciciov2024")
        self._raw_dir = config.PROJECT_ROOT / self._cfg["raw_dir"]
        self._files = dict(self._cfg["files"])

    def meta(self) -> DatasetMeta:
        return DatasetMeta(
            name=self._cfg["name"],
            class_names=sorted(self._files.keys()),
            benign_label=self._cfg["benign_label"],
            id_max=int(self._cfg["id_max"]),
        )

    def load(self) -> pd.DataFrame:
        frames = []
        for class_name, filename in self._files.items():
            df = pd.read_csv(self._raw_dir / filename)
            df.columns = [c.strip() for c in df.columns]
            df[LABEL_COLUMN] = class_name
            # Keep only the canonical columns; the dataset's own label/category/
            # specific_class columns are redundant with true_class.
            frames.append(df[CANONICAL_COLUMNS])
        combined = pd.concat(frames, ignore_index=True)
        return validate(combined, self.meta())
