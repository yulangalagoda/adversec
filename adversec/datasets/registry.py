"""
registry.py
===========
Name -> adapter lookup, so the CLI and experiments can select a dataset by string
without importing its module directly.
"""
from __future__ import annotations

from .base import CANDataset
from .ciciov import CICIoVDataset
from .road import ROADDataset


_REGISTRY: dict[str, type[CANDataset]] = {
    "ciciov2024": CICIoVDataset,
    "road": ROADDataset,
}


def get_dataset(name: str) -> CANDataset:
    if name not in _REGISTRY:
        raise KeyError(f"unknown dataset '{name}'; known: {sorted(_REGISTRY)}")
    return _REGISTRY[name]()


def available() -> list[str]:
    return sorted(_REGISTRY)
