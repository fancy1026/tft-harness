"""Champion appearance: a DINOv2-small image embedding of each unit (data/models/dinov2-small).

Two uses:
  - same unit?  a unit's crop now against its own crop from when its name was confirmed (a much
    easier question than naming it among ~60 champions; no training data needed)
  - which champion?  nearest labelled crops in data/champ-library (buys: shop text + the slot the
    unit landed in; info-panel reads confirmed by the selection ring) - answers only when clear.

A unit's crop is cut around its health bar (the model stands under it), so the same code works
for bench and board units.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / 'data' / 'models' / 'dinov2-small' / 'model.onnx'
LIBRARY = ROOT / 'data' / 'champ-library'
SIZE = (112, 126)                      # w, h: multiples of the 14-px patch
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
_sess = None


def session():
    global _sess
    if _sess is None:
        import onnxruntime as ort
        o = ort.SessionOptions()
        o.intra_op_num_threads = 4
        o.add_session_config_entry('session.intra_op.allow_spinning', '0')   # no busy-wait beside the OCR (screen._tune_onnx)
        _sess = ort.InferenceSession(str(MODEL), o, providers=['CPUExecutionProvider'])
    return _sess


def unit_crop(frame, bar):
    """RGB crop of the unit under the health bar at bar = (x, y) in 1080p coordinates."""
    h, w = frame.shape[:2]
    s = h / 1080
    cx, cy = w / 2 + (bar[0] - 960) * s, bar[1] * s
    c = frame[max(0, round(cy - 4 * s)):round(cy + 126 * s), max(0, round(cx - 60 * s)):round(cx + 60 * s)]
    if not c.size:
        return None
    c = cv2.resize(c, (96, 104), interpolation=cv2.INTER_AREA)
    c[:32, :60] = 128          # the bar and the item icons under it: items must not change the look
    return c


def embed(crops):
    """[RGB crop] -> (n, 384) unit vectors (CLS token)."""
    if not crops:
        return np.zeros((0, 384), np.float32)
    x = np.stack([cv2.resize(c, SIZE, interpolation=cv2.INTER_CUBIC) for c in crops]).astype(np.float32) / 255
    x = ((x - MEAN) / STD).transpose(0, 3, 1, 2)
    out = session().run(None, {'pixel_values': x})[0]
    v = out[:, 0]
    return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-6)


def same_unit(a, b):
    """Cosine similarity of two embeddings."""
    return float(np.dot(a, b))


_lib = None


def library():
    """(names, matrix) of labelled crops' embeddings."""
    global _lib
    if _lib is None:
        names, vecs = [], []
        try:
            for f, name in json.loads((LIBRARY / 'labels.json').read_text(encoding='utf-8')).items():
                img = cv2.imread(str(LIBRARY / f))
                if img is not None:
                    names.append(name)
                    vecs.append(img[..., ::-1])
        except (OSError, ValueError):
            pass
        _lib = (names, embed(vecs) if vecs else np.zeros((0, 384), np.float32))
    return _lib


NAME_SIM, NAME_LEAD = 0.80, 0.05      # set from the held-out test (see tools/eval_champs.py)


def name_unit(vec):
    """(name or None, similarity, lead over the best other champion)."""
    names, mat = library()
    if not names:
        return None, 0.0, 0.0
    s = mat @ vec
    best = {}
    for n, v in zip(names, s):
        best[n] = max(best.get(n, -1.0), float(v))
    top = sorted(best.items(), key=lambda x: -x[1])
    lead = top[0][1] - (top[1][1] if len(top) > 1 else 0.0)
    ok = top[0][1] >= NAME_SIM and lead >= NAME_LEAD
    return (top[0][0] if ok else None), top[0][1], lead


_labels = None


def collecting():
    """Crops are collected in our test games only (TFT_TEST_MODE): never for players - a crop can show the
    player's in-game name over their tactician (the user, 2026-10-15), and play never reads the library."""
    import os
    return os.environ.get('TFT_TEST_MODE') == '1'


def add_to_library(crop, name, star=1, burst=False, flush=True, vec=None):
    """Keep a crop whose name was confirmed (info-panel text, or a buy seen landing). burst=True:
    one of many frames of the same unit (named apart so near-duplicates can be told apart).
    flush=False: the label file is written later by flush_labels() (a burst saves ~70 frames)."""
    if not collecting():
        return
    import hashlib
    import re
    global _lib, _labels
    LIBRARY.mkdir(parents=True, exist_ok=True)
    f = f"{re.sub(r'[^A-Za-z0-9]+', '_', name)}_{star}{'_b' if burst else ''}_{hashlib.md5(crop.tobytes()).hexdigest()[:8]}.png"
    cv2.imwrite(str(LIBRARY / f), np.ascontiguousarray(crop[..., ::-1]))
    if _labels is None:
        try:
            _labels = json.loads((LIBRARY / 'labels.json').read_text(encoding='utf-8'))
        except (OSError, ValueError):
            _labels = {}
    _labels[f] = name
    if vec is not None and _lib is not None:
        _lib = (_lib[0] + [name], np.vstack([_lib[1], vec[None]]))   # no re-embedding mid-game
    else:
        _lib = None
    if flush:
        flush_labels()


def flush_labels():
    if _labels is not None and collecting():
        (LIBRARY / 'labels.json').write_text(json.dumps(_labels, indent=1), encoding='utf-8')


PROGRESS = LIBRARY / 'progress.json'
NEED = 50                                     # frames per place before a champion is crossed off


def progress():
    try:
        return json.loads(PROGRESS.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return {}


def add_progress(name, bench, board):
    if not collecting():
        return
    p = progress()
    e = p.setdefault(name, {'bench': 0, 'board': 0})
    e['bench'] += bench
    e['board'] += board
    LIBRARY.mkdir(parents=True, exist_ok=True)
    PROGRESS.write_text(json.dumps(p, indent=1), encoding='utf-8')


def wants(name, place):
    """Does the library still need crops of this champion at this place ('bench' / 'board')?"""
    return progress().get(name, {}).get(place, 0) < NEED


def still_needed():
    """Champions not yet crossed off the list (fewer than NEED bench or board frames)."""
    from tft_player.catalog import CATALOG
    us = CATALOG.units
    names = [u.name for u in (us.values() if isinstance(us, dict) else us)]
    p = progress()
    return {n for n in names if p.get(n, {}).get('bench', 0) < NEED}     # bench only (the user, 2026-10-05)
