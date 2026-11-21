"""CSV, JSON and mesh output."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


def read_csv(path):
    with Path(path).open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_plain), encoding='utf-8')


def _plain(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f'not JSON serialisable: {type(value)}')


def write_mesh(path, mesh, face_colors=None):
    """Binary PLY in world millimetres; optional per-face RGBA colours (uint8, one row per face)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh = mesh.copy()
    if face_colors is not None:
        mesh.visual.face_colors = face_colors
    mesh.export(path)
