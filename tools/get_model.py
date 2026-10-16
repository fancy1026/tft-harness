"""Download the DINOv2-small image model (data/models/dinov2-small/model.onnx) and check its hash."""
import hashlib
import sys
import urllib.request
from pathlib import Path

URL = 'https://huggingface.co/onnx-community/dinov2-small/resolve/main/onnx/model.onnx'
SHA256 = 'f22797eabf810a75e41de68d378541ebea372122b25c4ce3ef25ff618250c20a'
OUT = Path(__file__).resolve().parent.parent / 'data' / 'models' / 'dinov2-small' / 'model.onnx'


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


if __name__ == '__main__':
    if OUT.exists() and sha256(OUT) == SHA256:
        sys.exit(print('image model: already there'))
    print('image model: downloading 88 MB from huggingface.co ...', flush=True)
    tmp = OUT.with_suffix('.part')
    OUT.parent.mkdir(parents=True, exist_ok=True)       # a fresh clone has no data/models folder (git keeps no empty folders)
    urllib.request.urlretrieve(URL, tmp)
    if sha256(tmp) != SHA256:
        tmp.unlink()
        sys.exit('image model: hash mismatch, download rejected')
    tmp.replace(OUT)
    print('image model: ok')
