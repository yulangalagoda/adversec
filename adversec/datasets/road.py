"""
road.py
=======
Adapter for the ROAD (Real ORNL Automotive Dynamometer) CAN dataset.

Raw form is candump logs plus a capture_metadata.json describing each attack's
injection ID, payload mask and time interval. This adapter parses the logs,
labels the injected frames (targeted attacks by ID + interval + fixed bytes;
fuzzing by interval + all-FF payload), samples benign frames from ambient
captures, and emits the canonical table. It is the only ROAD-aware code.

Ported from the legacy src/road_cleaning.py with one addition: _parse_log_file
skips malformed / non-8-byte frames instead of assuming every frame is exactly
eight bytes. On the clean ROAD captures (all 8-byte) this is a no-op, so ported
output matches the legacy loader.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .. import config
from ..contract import (
    CANONICAL_COLUMNS,
    DATA_COLUMNS,
    ID_COLUMN,
    LABEL_COLUMN,
    DatasetMeta,
    validate,
)
from .base import CANDataset


def _parse_log_file(log_path: Path) -> pd.DataFrame:
    """
    Parse one candump .log into timestamp + ID + DATA_0..7.

    Each raw line looks like: (1030000000.000000) can0 354#200A000000027480
    Malformed lines and frames without exactly eight payload bytes are skipped
    (and counted), since only 8-byte data frames map onto the canonical layout.
    """
    records = []
    skipped = 0
    with open(log_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 3 or "#" not in parts[2]:
                skipped += 1
                continue
            id_hex, payload_hex = parts[2].split("#")
            if len(payload_hex) != 16:  # 16 hex chars == 8 bytes
                skipped += 1
                continue
            timestamp = float(parts[0].strip("()"))
            can_id = int(id_hex, 16)
            data_bytes = [int(payload_hex[i:i + 2], 16) for i in range(0, 16, 2)]
            record = {"timestamp": timestamp, ID_COLUMN: can_id}
            for i in range(8):
                record[f"DATA_{i}"] = data_bytes[i]
            records.append(record)
    if skipped:
        print(f"    {log_path.name}: skipped {skipped} malformed / non-8-byte frames")
    return pd.DataFrame(records)


def _parse_injection_mask(injection_data_str):
    """Turn a 16-char payload mask into [(byte_index, value)] for the fixed (non-X) bytes."""
    if injection_data_str is None:
        return []
    constraints = []
    for byte_index in range(8):
        chunk = injection_data_str[byte_index * 2: byte_index * 2 + 2]
        if chunk.upper() == "XX":
            continue
        constraints.append((byte_index, int(chunk, 16)))
    return constraints


def _label_attack_capture(df, capture_name, meta):
    """Label targeted-attack frames: ID match AND inside interval AND fixed bytes match."""
    entry = meta[capture_name]
    start_time = df["timestamp"].iloc[0]
    elapsed = df["timestamp"] - start_time
    injection_id = int(entry["injection_id"], 16)
    interval_start, interval_end = entry["injection_interval"]
    constraints = _parse_injection_mask(entry["injection_data_str"])

    is_attack = df[ID_COLUMN] == injection_id
    is_attack = is_attack & (elapsed >= interval_start) & (elapsed <= interval_end)
    for byte_index, value in constraints:
        is_attack = is_attack & (df[f"DATA_{byte_index}"] == value)

    df[LABEL_COLUMN] = "benign"
    df.loc[is_attack, LABEL_COLUMN] = capture_name
    return df


def _label_fuzzing_capture(df, capture_name, meta):
    """Label fuzzing frames: inside interval AND all-FF payload (isolates true injections)."""
    entry = meta[capture_name]
    start_time = df["timestamp"].iloc[0]
    elapsed = df["timestamp"] - start_time
    interval_start, interval_end = entry["injection_interval"]

    in_window = (elapsed >= interval_start) & (elapsed <= interval_end)
    all_ff = (df[DATA_COLUMNS] == 255).all(axis=1)
    is_attack = in_window & all_ff

    df[LABEL_COLUMN] = "benign"
    df.loc[is_attack, LABEL_COLUMN] = capture_name
    return df


class ROADDataset(CANDataset):
    def __init__(self, cfg: dict | None = None):
        self._cfg = cfg if cfg is not None else config.load_dataset_config("road")
        root = config.PROJECT_ROOT / self._cfg["raw_dir"]
        self._attacks_dir = root / self._cfg["attacks_subdir"]
        self._ambient_dir = root / self._cfg["ambient_subdir"]
        self._metadata_path = self._attacks_dir / "capture_metadata.json"
        self._attack_captures = dict(self._cfg["attack_captures"])
        self._fuzzing_classes = set(self._cfg["fuzzing_classes"])
        self._ambient_captures = list(self._cfg["ambient_captures"])
        self._sample_per_capture = int(self._cfg["ambient_sample_per_capture"])
        self._seed = config.RANDOM_SEED

    def meta(self) -> DatasetMeta:
        classes = sorted(set(self._attack_captures.keys()) | {self._cfg["benign_label"]})
        return DatasetMeta(
            name=self._cfg["name"],
            class_names=classes,
            benign_label=self._cfg["benign_label"],
            id_max=int(self._cfg["id_max"]),
        )

    def _load_metadata(self):
        with open(self._metadata_path, "r") as f:
            return json.load(f)

    def _load_attacks(self):
        meta = self._load_metadata()
        class_frames = []
        for clean_name, capture_list in self._attack_captures.items():
            is_fuzzing = clean_name in self._fuzzing_classes
            per_class = []
            for capture_name in capture_list:
                d = _parse_log_file(self._attacks_dir / f"{capture_name}.log")
                if is_fuzzing:
                    d = _label_fuzzing_capture(d, capture_name, meta)
                else:
                    d = _label_attack_capture(d, capture_name, meta)
                atk = d[d[LABEL_COLUMN] == capture_name].copy()
                atk[LABEL_COLUMN] = clean_name
                per_class.append(atk)
            class_frames.append(pd.concat(per_class, ignore_index=True))
        return pd.concat(class_frames, ignore_index=True)

    def _load_benign(self):
        pieces = []
        for capture_name in self._ambient_captures:
            d = _parse_log_file(self._ambient_dir / f"{capture_name}.log")
            if len(d) > self._sample_per_capture:
                d = d.sample(n=self._sample_per_capture, random_state=self._seed)
            d[LABEL_COLUMN] = self._cfg["benign_label"]
            pieces.append(d)
        return pd.concat(pieces, ignore_index=True)

    def load(self) -> pd.DataFrame:
        attacks_df = self._load_attacks()
        benign_df = self._load_benign()
        combined = pd.concat([attacks_df, benign_df], ignore_index=True)
        # Drop timestamp (only needed for interval labelling) by keeping canonical columns.
        combined = combined[CANONICAL_COLUMNS]
        return validate(combined, self.meta())
