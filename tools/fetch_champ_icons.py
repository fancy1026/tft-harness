"""Champion portraits for the overlay: every Set champion's square tile from CommunityDragon,
saved at 64 px in data/champ-icons/<Name>.png (the overlay draws them; perception never uses them).

    python tools/fetch_champ_icons.py
"""
import io
import json
import sys
import urllib.request
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'data' / 'champ-icons'
CDRAGON = 'https://raw.communitydragon.org/latest/game/'


def main():
    src = sorted((ROOT / 'data' / 'sources').glob('cdragon-tft-*.json'))[-1]
    d = json.loads(src.read_text(encoding='utf-8'))
    s = d['sets'][max(d['sets'], key=int)]
    OUT.mkdir(parents=True, exist_ok=True)
    got = 0
    for c in s['champions']:
        name, tile = c.get('name'), c.get('squareIcon') or ''
        if not name or not tile or not c.get('traits'):
            continue                                  # summons, dummies: no portrait needed
        path = OUT / f'{name}.png'
        if path.exists():
            continue
        url = CDRAGON + tile.lower().replace('.tex', '.png')
        try:
            raw = urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'}),
                                         timeout=30).read()
        except Exception as e:
            print(f'{name}: {e}')
            continue
        im = Image.open(io.BytesIO(raw)).convert('RGB')
        im.thumbnail((64, 64))
        im.save(path)
        got += 1
    print(f'{OUT}: {got} new, {len(list(OUT.glob("*.png")))} in all')
    # item icons: components and every completed item (recipes), for the comp plan's item row
    sys.path.insert(0, str(ROOT))
    from tft_player.items import COMPONENTS, RECIPES
    want = set(COMPONENTS) | set(RECIPES)
    icons = {}
    for it in d['items']:
        if it.get('name') in want and it.get('icon'):
            icons.setdefault(it['name'], it['icon'])
    out = ROOT / 'data' / 'item-icons-overlay'
    out.mkdir(parents=True, exist_ok=True)
    got = 0
    for name, icon in icons.items():
        path = out / (''.join(ch for ch in name if ch.isalnum()) + '.png')
        if path.exists():
            continue
        try:
            raw = urllib.request.urlopen(urllib.request.Request(CDRAGON + icon.lower().replace('.tex', '.png'),
                                                                headers={'User-Agent': 'Mozilla/5.0'}), timeout=30).read()
        except Exception as e:
            print(f'{name}: {e}')
            continue
        im = Image.open(io.BytesIO(raw)).convert('RGB')
        im.thumbnail((40, 40))
        im.save(path)
        got += 1
    print(f'{out}: {got} new, {len(list(out.glob("*.png")))} in all')


if __name__ == '__main__':
    sys.exit(main())
