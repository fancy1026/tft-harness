"""Fingerprints of reference pictures: what the release ships instead of pictures cut from the game.

A fingerprint is a picture's zero-mean, unit-length sample times a fixed random k x n matrix (seeded, so the code
rebuilds the matrix). Similarities between pictures are kept (items.py, bench.py compare fingerprints directly); the
picture is not: k numbers for n > k pixel values. A template matcher (the carousel marker) rebuilds a template from a
fingerprint - the smoothest picture with that fingerprint, a blurred version of the original. The loot orbs are still
matched with their pictures: no fingerprint kept their detection (experiments/README.md section 6).
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

SEED = 18
_mats: dict = {}


def matrix(k, n, seed=SEED):
    if (k, n, seed) not in _mats:
        _mats[(k, n, seed)] = np.random.default_rng(seed).standard_normal((k, n)).astype(np.float32) / np.sqrt(k)
    return _mats[(k, n, seed)]


def sample(img):
    v = np.asarray(img, np.float32).ravel()
    v = v - v.mean()
    return v / (np.linalg.norm(v) + 1e-6)


def fingerprint(img, k=256):
    """A picture (any shape) -> k numbers, unit length."""
    v = sample(img)
    f = matrix(k, v.size) @ v
    return f / (np.linalg.norm(f) + 1e-6)


def _dct(n):
    c = np.array([[np.cos(np.pi * (2 * x + 1) * u / (2 * n)) * (np.sqrt(1 / n) if u == 0 else np.sqrt(2 / n))
                   for x in range(n)] for u in range(n)])
    return c                                                    # rows: frequencies


def rebuild(fp, shape):
    """The smoothest picture of this shape (h, w) or (h, w, c) whose fingerprint is fp: among all pictures with that
    fingerprint, the one with the least energy at high spatial frequencies."""
    h, w = shape[:2]
    ch = shape[2] if len(shape) > 2 else 1
    ch_, cw_ = _dct(h), _dct(w)
    # basis[i] = one 2-D DCT image (frequency u, v) of one channel; weight grows with frequency
    B, wt = [], []
    for u in range(h):
        for v in range(w):
            img = np.outer(ch_[u], cw_[v])
            for c in range(ch):
                z = np.zeros((h, w, ch))
                z[..., c] = img
                B.append(z.ravel())
                wt.append((1 + u + v) ** 2)
    B = np.array(B).T                                           # n x n
    winv = 1.0 / np.array(wt, np.float64)
    A = matrix(len(fp), B.shape[0]).astype(np.float64) @ B
    c = winv * (A.T @ np.linalg.solve((A * winv) @ A.T, np.asarray(fp, np.float64)))
    img = (B @ c).reshape(shape)
    img = (img - img.min()) / (img.max() - img.min() + 1e-9) * 255
    return img.astype(np.uint8)


# ---- reference pictures for template matchers (data/loot: the carousel marker) ---------------------------------------

REF_SIZE = 32           # references are fingerprinted as 32x32 grey pictures


def ref_print(gray):
    return fingerprint(cv2.resize(gray, (REF_SIZE, REF_SIZE), interpolation=cv2.INTER_AREA))


def window_prints(windows):
    """Fingerprints of grey windows (any equal size), as rows."""
    V = np.stack([sample(cv2.resize(w, (REF_SIZE, REF_SIZE), interpolation=cv2.INTER_LINEAR)) for w in windows])
    F = V @ matrix(256, REF_SIZE * REF_SIZE).T
    return F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-6)


def references(folder, prints_file):
    """{name: (fingerprint, (h, w))}: the shipped fingerprints (prints_file) and the pictures folder/<name>.png."""
    out = {}
    if prints_file.exists():
        z = np.load(prints_file)
        for n, v, s in zip(z['names'], z['vecs'], z['shapes']):
            out[str(n)] = (v.astype(np.float32), (int(s[0]), int(s[1])))
    for p in sorted(folder.glob('*.png')):
        g = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2GRAY)
        out[p.stem] = (ref_print(g), g.shape[:2])
    return out


def save_references(folder, path, names=None):
    """The folder's pictures (or those named) as fingerprints only (the release)."""
    refs = {n: r for n, r in references(folder, Path('nonexistent')).items() if names is None or n in names}
    np.savez_compressed(path, names=np.array(list(refs)), vecs=np.stack([f for f, _ in refs.values()]),
                        shapes=np.array([s for _, s in refs.values()]))
    return len(refs)


def template(fp, shape):
    """A grey template of shape (h, w) rebuilt from a reference's fingerprint."""
    return cv2.resize(rebuild(fp, (REF_SIZE, REF_SIZE)), (shape[1], shape[0]), interpolation=cv2.INTER_CUBIC)
